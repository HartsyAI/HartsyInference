using System.Runtime.CompilerServices;
using System.Text;
using HartsyInference.Core.Logging;
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
    /// ends the run without a final result; that round's partial text stays out of <see cref="Conversation"/>, since
    /// the turn did not complete, but the results it relayed stay in (see below).
    /// <para>Pre-resolved calls: a stream that runs its tools itself (a remote assistant) relays each result as a <see
    /// cref="TextChunkKind.ToolResult"/> after its call. The result belongs to the oldest call of the round that arrived
    /// with the same id and has no result yet, so it follows that call's repaired id; it streams once, where it arrived,
    /// under the loop's own result index. A relayed result that matches no such call is logged and dropped. A relayed
    /// result for a call of the open assistant turn closes that turn at the next text or call, so the conversation reads
    /// each turn's text with every call issued in it, in arrival order, then those calls' results, then the next turn:
    /// the shape the stream went through, and the one <c>VoiceAgentSession</c> records from the same chunks. A call
    /// without a relayed result is dispatched after the round and recorded in the turn that issued it, then the model is
    /// asked again. A round whose calls were all pre-resolved already holds its final answer, so it ends the run like a
    /// round without calls, reporting <see cref="StopReason.Stop"/> where the stream said <see cref="StopReason.ToolCall"/>.
    /// The tool exchanges of an <see cref="StopReason.Error"/> or <see cref="StopReason.Cancelled"/> round that relayed
    /// results are recorded through its last relayed result, since those calls did run and their results were streamed;
    /// its text after that and its unresolved calls are not. A round <see cref="TextChunkKind.Result"/> replaces the text
    /// of the closed turns only where it extends them; otherwise those turns keep their streamed text, the open turn keeps
    /// its streamed text too, and only <see cref="VisibleText"/> takes the result.</para></remarks>
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
                        // A result the stream relays for a call it ran itself.
                        string wireId = chunk.ToolCall?.Id ?? "";
                        if (calls.Resolve(wireId, chunk.Text ?? "") is { } resolved)
                        {
                            yield return chunk with { ToolCall = resolved, ToolCallIndex = resultIndex++ };
                        }
                        else
                        {
                            Logs.Warning("[ToolLoop] Dropped a relayed tool result for id '" + wireId + "' (" + (chunk.ToolCall?.Name ?? "no call")
                                + "): no call of this round with that id is waiting for a result.");
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
                        // The round's result is authoritative over its deltas (a provider may correct them), so it replaces them;
                        // text already recorded in closed turns is replaced only where the result extends it.
                        if (calls.Extends(chunk.Text)) roundText.Clear().Append(chunk.Text);
                        visible.Length = roundStart;
                        visible.Append(chunk.Text);
                        break;
                    case TextChunkKind.StopReason:
                        stop = chunk.Stop ?? StopReason.Stop;
                        if (stop is StopReason.Error or StopReason.Cancelled)
                        {
                            // Relayed calls did run, and their results already streamed.
                            _conversation.AddRange(calls.Messages(roundText, throughLastRelayed: true));
                            _results.AddRange(calls.Relayed);
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
            VisibleText = visible.ToString();
            _results.AddRange(calls.Relayed);
            List<NativeToolCall> open = calls.Unresolved();
            if (open.Count == 0)
            {
                // The final answer belongs to the conversation a host reuses for the next turn. Pre-resolved calls leave
                // nothing to run, so a tool-call stop is a plain one.
                _conversation.AddRange(calls.Messages(roundText, throughLastRelayed: false));
                if (stop == StopReason.ToolCall && calls.Relayed.Count > 0) stop = StopReason.Stop;
                Stop = stop;
                yield return new TextChunk { Kind = TextChunkKind.Result, Text = VisibleText };
                yield return new TextChunk { Kind = TextChunkKind.StopReason, Stop = stop };
                yield break;
            }
            if (round == _options.MaxRounds)
            {
                // The turn's text stays in the conversation; its open calls do not, since they were never dispatched.
                _conversation.AddRange(calls.Messages(roundText, throughLastRelayed: false));
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
            Stop = stop;
            foreach (NativeToolCall call in open)
            {
                string result = await DispatchAsync(call, cancel).ConfigureAwait(false);
                calls.Complete(call, result);
                _results.Add((call, result));
                yield return new TextChunk { Kind = TextChunkKind.ToolResult, Text = result, ToolCall = call, ToolCallIndex = resultIndex++ };
            }
            _conversation.AddRange(calls.Messages(roundText, throughLastRelayed: false));
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
        private readonly Dictionary<string, Queue<Slot>> _waitingByWireId = new(StringComparer.Ordinal);
        private readonly List<Turn> _turns = [new Turn(0)];
        private string _closedText = "";

        /// <summary>The relayed results with their calls, in arrival order.</summary>
        public List<(NativeToolCall Call, string Result)> Relayed { get; } = [];

        /// <summary>Where the open turn's text starts in the round's text.</summary>
        public int TextStart => _turns[^1].TextStart;

        /// <summary>A call of the open turn has a relayed result, so the next text or call starts a new turn.</summary>
        public bool HasRelayed => _turns[^1].Relayed;

        /// <summary>Records <paramref name="call"/>, repaired from the id it arrived with, in the open turn.</summary>
        public void Add(string wireId, NativeToolCall call)
        {
            Slot slot = new(call);
            _turns[^1].Calls.Add(slot);
            if (!_waitingByWireId.TryGetValue(wireId, out Queue<Slot>? queue)) _waitingByWireId[wireId] = queue = new();
            queue.Enqueue(slot);
        }

        /// <summary>Resolves the oldest waiting call that arrived with <paramref name="wireId"/>; null when there is none.</summary>
        public NativeToolCall? Resolve(string wireId, string result)
        {
            if (!_waitingByWireId.TryGetValue(wireId, out Queue<Slot>? queue) || queue.Count == 0) return null;
            Slot slot = queue.Dequeue();
            slot.Result = result;
            Relayed.Add((slot.Call, result));
            Turn open = _turns[^1];
            if (open.Calls.Contains(slot)) open.Relayed = true;
            return slot.Call;
        }

        /// <summary>Records a dispatched call's result.</summary>
        public void Complete(NativeToolCall call, string result)
        {
            foreach (Turn turn in _turns)
            {
                foreach (Slot slot in turn.Calls)
                {
                    if (ReferenceEquals(slot.Call, call)) slot.Result = result;
                }
            }
        }

        /// <summary>Calls without a result, in arrival order.</summary>
        public List<NativeToolCall> Unresolved()
        {
            List<NativeToolCall> open = [];
            foreach (Turn turn in _turns)
            {
                foreach (Slot slot in turn.Calls)
                {
                    if (slot.Result is null) open.Add(slot.Call);
                }
            }
            return open;
        }

        /// <summary>Ends the open turn at the round's current text and opens the next.</summary>
        public void CloseTurn(StringBuilder roundText)
        {
            Turn closing = _turns[^1];
            closing.Text = roundText.ToString(closing.TextStart, roundText.Length - closing.TextStart);
            _closedText = roundText.ToString();
            _turns.Add(new Turn(roundText.Length));
        }

        /// <summary>Whether <paramref name="text"/> keeps the closed turns' text as its start.</summary>
        public bool Extends(string text) => text.StartsWith(_closedText, StringComparison.Ordinal);

        /// <summary>Each turn's assistant message with its calls that have a result, then those results; with <paramref
        /// name="throughLastRelayed"/>, only the turns up to the last one holding a relayed result.</summary>
        public List<TextMessage> Messages(StringBuilder roundText, bool throughLastRelayed)
        {
            int last = _turns.Count - 1;
            if (throughLastRelayed)
            {
                while (last >= 0 && !_turns[last].Calls.Exists(slot => slot.Result is not null)) last--;
            }
            List<TextMessage> messages = [];
            for (int t = 0; t <= last; t++)
            {
                Turn turn = _turns[t];
                string text = turn.Text ?? roundText.ToString(turn.TextStart, roundText.Length - turn.TextStart);
                List<NativeToolCall> resolved = [];
                foreach (Slot slot in turn.Calls)
                {
                    if (slot.Result is not null) resolved.Add(slot.Call);
                }
                if (text.Length == 0 && resolved.Count == 0) continue;
                messages.Add(new TextMessage { Role = TextRole.Assistant, Content = text, ToolCalls = resolved.Count > 0 ? resolved : null });
                foreach (Slot slot in turn.Calls)
                {
                    if (slot.Result is { } result)
                    {
                        messages.Add(new TextMessage { Role = TextRole.Tool, Content = result, ToolCallId = slot.Call.Id, Name = slot.Call.Name });
                    }
                }
            }
            return messages;
        }

        /// <summary>One call and, once it has one, its result.</summary>
        private sealed class Slot(NativeToolCall call)
        {
            public NativeToolCall Call { get; } = call;

            public string? Result { get; set; }
        }

        /// <summary>One assistant turn: where its text starts, its text once closed, and the calls issued in it.</summary>
        private sealed class Turn(int textStart)
        {
            public int TextStart { get; } = textStart;

            public string? Text { get; set; }

            public List<Slot> Calls { get; } = [];

            public bool Relayed { get; set; }
        }
    }
}
