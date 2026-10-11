using System.Runtime.CompilerServices;
using System.Text;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Tools;

/// <summary>One run of the agent loop: streams each model round, dispatches the calls it ends on, and feeds the results
/// back. Runs once; after <see cref="RunAsync"/> completes, <see cref="Conversation"/>, <see cref="ToolResults"/> and
/// the other summaries describe what happened.</summary>
public sealed class ToolLoopRun
{
    private readonly Func<TextRequest, CancellationToken, IAsyncEnumerable<TextChunk>> _stream;
    private readonly TextRequest _request;
    private readonly IToolDispatcher _tools;
    private readonly ToolLoopOptions _options;
    private readonly List<TextMessage> _conversation;
    private readonly List<(NativeToolCall Call, string Result)> _results = [];
    private readonly HashSet<string> _usedIds = new(StringComparer.Ordinal);
    private bool _started;

    internal ToolLoopRun(Func<TextRequest, CancellationToken, IAsyncEnumerable<TextChunk>> stream,
        TextRequest request, IToolDispatcher tools, ToolLoopOptions options)
    {
        _stream = stream;
        _request = request;
        _tools = tools;
        _options = options;
        _conversation = [.. request.Messages];
    }

    /// <summary>The conversation as the loop left it: the request's messages, then each round's assistant turn and its
    /// tool results. Like the other summaries, read it after <see cref="RunAsync"/> completes; it is not synchronized.</summary>
    public IReadOnlyList<TextMessage> Conversation => _conversation;

    /// <summary>Model invocations made so far.</summary>
    public int Rounds { get; private set; }

    /// <summary>Why the last round stopped.</summary>
    public StopReason Stop { get; private set; }

    /// <summary>The visible text of every round, concatenated.</summary>
    public string VisibleText { get; private set; } = "";

    /// <summary>Every dispatched (or denied) call with the result the model received for it, in order.</summary>
    public IReadOnlyList<(NativeToolCall Call, string Result)> ToolResults => _results;

    /// <summary>Streams the run. The request's <see cref="TextRequest.ForceToolId"/> applies to the first round only.</summary>
    /// <remarks>Chunk shape: each round's chunks stream through (text, calls, reasoning, status, usage); each round's
    /// own result and call-stop are suppressed; the run ends with one <see cref="TextChunkKind.Result"/> holding <see
    /// cref="VisibleText"/> and one <see cref="TextChunkKind.StopReason"/>. A dispatched call streams as <see
    /// cref="TextChunkKind.ToolResult"/>. Hitting the round limit while the model still asks for a tool streams a <see
    /// cref="ToolLoop.RoundLimitPhase"/> status, the final result and <see cref="StopReason.ToolCall"/>; that last call
    /// is not dispatched, and that round's text (without the call) ends <see cref="Conversation"/>. A round's own
    /// <see cref="TextChunkKind.Result"/> replaces its already-streamed deltas in <see cref="VisibleText"/> and the final
    /// result when the two differ. A gate or dispatcher failure becomes that call's error result. An <see
    /// cref="StopReason.Error"/> or <see cref="StopReason.Cancelled"/> stop is relayed and
    /// ends the run without a final result; that round's partial text stays out of <see cref="Conversation"/>, since
    /// the turn did not complete.</remarks>
    public async IAsyncEnumerable<TextChunk> RunAsync([EnumeratorCancellation] CancellationToken cancel = default)
    {
        if (_started) throw new InvalidOperationException("A tool loop runs once; create a new one for another run.");
        _started = true;
        IReadOnlyList<ToolDefinition> tools = _request.Tools is { Count: > 0 } ? _request.Tools : _tools.Definitions;
        StringBuilder visible = new();
        int resultIndex = 0;
        for (int round = 1; round <= _options.MaxRounds; round++)
        {
            Rounds = round;
            TextRequest roundRequest = _request with
            {
                Messages = [.. _conversation],
                Tools = tools,
                ForceToolId = round == 1 ? _request.ForceToolId : null,
            };
            StringBuilder roundText = new();
            int roundStart = visible.Length;
            List<NativeToolCall> calls = [];
            // A call's id is settled when its first delta arrives, so the deltas, the completed call and its result share it.
            Dictionary<int, string> idsByIndex = [];
            // Index-less calls match their delta by the id the delta arrived with (empty ids in arrival order).
            Dictionary<string, Queue<string>> idsByOriginal = new(StringComparer.Ordinal);
            StopReason stop = StopReason.Stop;
            await foreach (TextChunk chunk in _stream(roundRequest, cancel).WithCancellation(cancel).ConfigureAwait(false))
            {
                switch (chunk.Kind)
                {
                    case TextChunkKind.Chunk:
                        roundText.Append(chunk.Text);
                        visible.Append(chunk.Text);
                        yield return chunk;
                        break;
                    case TextChunkKind.ToolCallDelta when chunk.ToolCall is { } begin:
                    {
                        NativeToolCall repairedBegin = EnsureUsableId(begin);
                        string original = begin.Id ?? "";
                        if (!idsByOriginal.TryGetValue(original, out Queue<string>? pending)) idsByOriginal[original] = pending = new();
                        pending.Enqueue(repairedBegin.Id);
                        if (chunk.ToolCallIndex is int beginIndex) idsByIndex[beginIndex] = repairedBegin.Id;
                        yield return chunk with { ToolCall = repairedBegin };
                        break;
                    }
                    case TextChunkKind.NativeToolCall:
                        if (chunk.ToolCall is { } call)
                        {
                            NativeToolCall repaired = SettledId(chunk.ToolCallIndex, call, idsByIndex, idsByOriginal) is { } settled
                                ? call with { Id = settled }
                                : EnsureUsableId(call);
                            calls.Add(repaired);
                            yield return chunk with { ToolCall = repaired };
                        }
                        else
                        {
                            yield return chunk;
                        }
                        break;
                    case TextChunkKind.Result:
                        if (string.IsNullOrEmpty(chunk.Text) || chunk.Text == roundText.ToString()) break;
                        if (roundText.Length == 0)
                        {
                            // A stream that reports its text only at the end still gets that text into the turn and the stream.
                            yield return new TextChunk { Kind = TextChunkKind.Chunk, Text = chunk.Text };
                        }
                        // The round's result is authoritative over its deltas (a provider may correct them), so it replaces them.
                        roundText.Clear().Append(chunk.Text);
                        visible.Length = roundStart;
                        visible.Append(chunk.Text);
                        break;
                    case TextChunkKind.StopReason:
                        stop = chunk.Stop ?? StopReason.Stop;
                        if (stop is StopReason.Error or StopReason.Cancelled)
                        {
                            Stop = stop;
                            VisibleText = visible.ToString();
                            yield return chunk;
                            yield break;
                        }
                        break;
                    default:
                        yield return chunk;
                        break;
                }
            }
            Stop = stop;
            VisibleText = visible.ToString();
            if (calls.Count == 0)
            {
                // The final answer belongs to the conversation a host reuses for the next turn.
                if (roundText.Length > 0) _conversation.Add(new TextMessage { Role = TextRole.Assistant, Content = roundText.ToString() });
                yield return new TextChunk { Kind = TextChunkKind.Result, Text = VisibleText };
                yield return new TextChunk { Kind = TextChunkKind.StopReason, Stop = stop };
                yield break;
            }
            if (round == _options.MaxRounds)
            {
                // The turn's text stays in the conversation; its calls do not, since they were never dispatched and have no results.
                if (roundText.Length > 0) _conversation.Add(new TextMessage { Role = TextRole.Assistant, Content = roundText.ToString() });
                Stop = StopReason.ToolCall;
                yield return new TextChunk
                {
                    Kind = TextChunkKind.Status,
                    Text = ToolLoop.RoundLimitPrefix + "max_rounds=" + _options.MaxRounds.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    Status = new TextStatus(ToolLoop.RoundLimitPhase),
                };
                yield return new TextChunk { Kind = TextChunkKind.Result, Text = VisibleText };
                yield return new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.ToolCall };
                yield break;
            }
            _conversation.Add(new TextMessage { Role = TextRole.Assistant, Content = roundText.ToString(), ToolCalls = calls });
            foreach (NativeToolCall call in calls)
            {
                string result = await DispatchAsync(call, cancel).ConfigureAwait(false);
                _conversation.Add(new TextMessage { Role = TextRole.Tool, Content = result, ToolCallId = call.Id, Name = call.Name });
                _results.Add((call, result));
                yield return new TextChunk { Kind = TextChunkKind.ToolResult, Text = result, ToolCall = call, ToolCallIndex = resultIndex++ };
            }
        }
    }

    /// <summary>The id a call's first delta settled: by index, else by the id the delta arrived with; null when no delta
    /// came.</summary>
    private static string? SettledId(int? index, NativeToolCall call, Dictionary<int, string> idsByIndex,
        Dictionary<string, Queue<string>> idsByOriginal)
    {
        if (index is int i && idsByIndex.TryGetValue(i, out string? byIndex)) return byIndex;
        return idsByOriginal.TryGetValue(call.Id ?? "", out Queue<string>? pending) && pending.Count > 0 ? pending.Dequeue() : null;
    }

    /// <summary>Runs the gate and the dispatcher for one call. A failure in either becomes the call's error result, so every
    /// call in the conversation has a result; cancellation still ends the run.</summary>
    private async Task<string> DispatchAsync(NativeToolCall call, CancellationToken cancel)
    {
        try
        {
            ToolCallDecision decision = _options.OnBeforeToolCall is { } gate
                ? await gate(call, cancel).ConfigureAwait(false) : ToolCallDecision.Allow;
            return decision.Allowed
                ? await _tools.InvokeAsync(call, cancel).ConfigureAwait(false)
                : decision.DenialResult ?? ToolRegistry.ErrorJson("The host denied this tool call.");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return ToolRegistry.ErrorJson(ex.Message);
        }
    }

    /// <summary>A call keeps its id when the id is non-empty and unused in this run; otherwise it gets a fresh one
    /// under <see cref="ToolLoopOptions.IdPrefix"/>.</summary>
    private NativeToolCall EnsureUsableId(NativeToolCall call)
    {
        if (!string.IsNullOrEmpty(call.Id) && _usedIds.Add(call.Id)) return call;
        string id;
        do
        {
            id = _options.IdPrefix + Guid.NewGuid().ToString("N")[..12];
        }
        while (!_usedIds.Add(id));
        return call with { Id = id };
    }
}
