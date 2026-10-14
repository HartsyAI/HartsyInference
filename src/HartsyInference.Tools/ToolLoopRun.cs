using System.Runtime.CompilerServices;
using System.Text;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Tools;

/// <summary>One run of the agent loop: streams each model round, dispatches the calls it ends on, and feeds the results
/// back. A call the round's stream already resolved, by relaying its <see cref="TextChunkKind.ToolResult"/>, is
/// recorded with that result and not dispatched. Runs once; after <see cref="RunAsync"/> completes, <see
/// cref="Conversation"/>, <see cref="ToolResults"/> and the other summaries describe what happened.</summary>
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

    /// <summary>Every dispatched, denied or pre-resolved call with the result the model received for it, in order.</summary>
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
    /// ends the run without a final result; that round's partial text and relayed results stay out of <see
    /// cref="Conversation"/> and <see cref="ToolResults"/>, since the turn did not complete.
    /// <para>Pre-resolved calls: a stream that runs its tools itself (a remote assistant) relays each result as a <see
    /// cref="TextChunkKind.ToolResult"/> after its call. The result belongs to the oldest call of the round that arrived
    /// with the same id and has no result yet, so it follows that call's repaired id; it streams once, where it arrived,
    /// under the loop's own result index. A relayed result that matches no such call is dropped. Each relayed result
    /// closes the assistant turn before it: the conversation gets that turn's text with its resolved calls, then their
    /// results, and the text after them starts the next turn, the shape the stream itself went through. A round whose
    /// calls were all pre-resolved already holds its final answer, so it ends the run like a round without calls, with
    /// the stream's own stop reason. Calls left without a result are dispatched as usual, recorded with the round's
    /// trailing text, and the model is asked again. A round <see cref="TextChunkKind.Result"/> that does not begin with
    /// the text already recorded ahead of a relayed result becomes the whole trailing turn.</para></remarks>
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
            RoundCalls calls = new();
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
                        if (calls.HasRelayed) calls.CloseTurn(roundText);
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
                            if (calls.HasRelayed) calls.CloseTurn(roundText);
                            calls.Add(call.Id ?? "", repaired);
                            yield return chunk with { ToolCall = repaired };
                        }
                        else
                        {
                            yield return chunk;
                        }
                        break;
                    case TextChunkKind.ToolResult:
                        // A result the stream relays for a call it ran itself; one naming no open call has nowhere to belong.
                        if (chunk.ToolCall is { } relayed && calls.Resolve(relayed.Id ?? "", chunk.Text ?? "") is { } resolved)
                        {
                            yield return chunk with { ToolCall = resolved, ToolCallIndex = resultIndex++ };
                        }
                        break;
                    case TextChunkKind.Result:
                        if (string.IsNullOrEmpty(chunk.Text) || chunk.Text == roundText.ToString()) break;
                        // Text reported only by the result comes after the relayed results it follows.
                        if (calls.HasRelayed && roundText.Length == calls.TextStart) calls.CloseTurn(roundText);
                        if (roundText.Length == 0)
                        {
                            // A stream that reports its text only at the end still gets that text into the turn and the stream.
                            yield return new TextChunk { Kind = TextChunkKind.Chunk, Text = chunk.Text };
                        }
                        // The round's result is authoritative over its deltas (a provider may correct them), so it replaces them.
                        roundText.Clear().Append(chunk.Text);
                        calls.OnTextReplaced(chunk.Text);
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
            if (calls.HasRelayed) calls.CloseTurn(roundText);
            _conversation.AddRange(calls.Closed);
            _results.AddRange(calls.ClosedResults);
            string tail = roundText.ToString(calls.TextStart, roundText.Length - calls.TextStart);
            List<NativeToolCall> open = calls.Open;
            if (open.Count == 0)
            {
                // The final answer belongs to the conversation a host reuses for the next turn.
                if (tail.Length > 0) _conversation.Add(new TextMessage { Role = TextRole.Assistant, Content = tail });
                yield return new TextChunk { Kind = TextChunkKind.Result, Text = VisibleText };
                yield return new TextChunk { Kind = TextChunkKind.StopReason, Stop = stop };
                yield break;
            }
            if (round == _options.MaxRounds)
            {
                // The turn's text stays in the conversation; its calls do not, since they were never dispatched and have no results.
                if (tail.Length > 0) _conversation.Add(new TextMessage { Role = TextRole.Assistant, Content = tail });
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
            _conversation.Add(new TextMessage { Role = TextRole.Assistant, Content = tail, ToolCalls = open });
            foreach (NativeToolCall call in open)
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

    /// <summary>One round's calls, split into assistant turns by the results the stream relays for calls it ran itself.</summary>
    private sealed class RoundCalls
    {
        private readonly Dictionary<string, Queue<NativeToolCall>> _openByWireId = new(StringComparer.Ordinal);
        private readonly List<(NativeToolCall Call, string Result)> _relayed = [];
        private string _closedText = "";

        /// <summary>Calls without a result yet, in arrival order.</summary>
        public List<NativeToolCall> Open { get; } = [];

        /// <summary>The closed turns: each one's assistant message, then its relayed results.</summary>
        public List<TextMessage> Closed { get; } = [];

        /// <summary>The closed turns' calls with their relayed results, in arrival order.</summary>
        public List<(NativeToolCall Call, string Result)> ClosedResults { get; } = [];

        /// <summary>Where the open turn's text starts in the round's text.</summary>
        public int TextStart { get; private set; }

        /// <summary>The open turn has relayed results, so the next text or call starts a new turn.</summary>
        public bool HasRelayed => _relayed.Count > 0;

        /// <summary>Opens <paramref name="call"/>, repaired from the id it arrived with.</summary>
        public void Add(string wireId, NativeToolCall call)
        {
            Open.Add(call);
            if (!_openByWireId.TryGetValue(wireId, out Queue<NativeToolCall>? queue)) _openByWireId[wireId] = queue = new();
            queue.Enqueue(call);
        }

        /// <summary>Resolves the oldest open call that arrived with <paramref name="wireId"/>; null when there is none.</summary>
        public NativeToolCall? Resolve(string wireId, string result)
        {
            if (!_openByWireId.TryGetValue(wireId, out Queue<NativeToolCall>? queue) || queue.Count == 0) return null;
            NativeToolCall call = queue.Dequeue();
            Open.Remove(call);
            _relayed.Add((call, result));
            return call;
        }

        /// <summary>Records the open turn's text with its resolved calls, then their results; open calls stay open.</summary>
        public void CloseTurn(StringBuilder roundText)
        {
            List<NativeToolCall> resolved = new(_relayed.Count);
            foreach ((NativeToolCall call, string _) in _relayed) resolved.Add(call);
            string text = roundText.ToString(TextStart, roundText.Length - TextStart);
            Closed.Add(new TextMessage { Role = TextRole.Assistant, Content = text, ToolCalls = resolved });
            foreach ((NativeToolCall call, string result) in _relayed)
            {
                Closed.Add(new TextMessage { Role = TextRole.Tool, Content = result, ToolCallId = call.Id, Name = call.Name });
                ClosedResults.Add((call, result));
            }
            _relayed.Clear();
            TextStart = roundText.Length;
            _closedText = roundText.ToString();
        }

        /// <summary>The round's text was replaced by its result; a result that no longer extends the recorded turns is
        /// all trailing text.</summary>
        public void OnTextReplaced(string text)
        {
            if (!text.StartsWith(_closedText, StringComparison.Ordinal)) TextStart = 0;
        }
    }
}
