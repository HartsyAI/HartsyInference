using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Turns the native text stream into OpenAI <c>chat.completion.chunk</c> frames. The assistant role comes first, then content and reasoning deltas. A tool call
/// streams its <c>id</c> and name in its first fragment and its argument text after that, so the client concatenates the arguments. A call the parser assembles without
/// streamed fragments goes out once, whole. The finish frame carries the finish reason, and a usage frame (empty <c>choices</c>) follows it when the client asked for
/// <c>stream_options.include_usage</c>.</summary>
/// <remarks>A call that streamed its first fragment and was then aborted by the parser cannot be retracted: its id and name have already gone out. A whole call that
/// arrives later under the aborted call's index is sent in full.</remarks>
internal sealed class ChatStreamTranslator
{
    private readonly string _id;
    private readonly long _created;
    private readonly string _model;
    private readonly bool _includeUsage;
    private readonly HashSet<int> _callsStreamed = [];
    private int _nextCallIndex;
    private string _finishReason = "stop";
    private ChatUsage? _usage;

    /// <summary>A translator for one reply, stamped with its id, creation time and model.</summary>
    public ChatStreamTranslator(string id, long created, string model, bool includeUsage)
    {
        _id = id;
        _created = created;
        _model = model;
        _includeUsage = includeUsage;
    }

    /// <summary>The first frame: the assistant role.</summary>
    public ChatCompletionChunk Start() => Frame(new ChatCompletionDelta { Role = "assistant" });

    /// <summary>The frames for one native chunk. A chunk with nothing for the client yields none.</summary>
    public IEnumerable<ChatCompletionChunk> Handle(TextChunk chunk)
    {
        switch (chunk.Kind)
        {
            case TextChunkKind.Chunk when chunk.Text is not null:
                yield return Frame(new ChatCompletionDelta { Content = chunk.Text });
                break;
            case TextChunkKind.Reasoning when chunk.Text is not null:
                yield return Frame(new ChatCompletionDelta { ReasoningContent = chunk.Text });
                break;
            case TextChunkKind.ToolCallDelta when chunk.ToolCall is { } begin:
                int started = CallIndex(chunk);
                _callsStreamed.Add(started);
                yield return ToolFrame(new ChatToolCallDeltaDto
                {
                    Index = started,
                    Id = begin.Id,
                    Type = "function",
                    Function = new ChatToolCallDeltaFunctionDto { Name = begin.Name, Arguments = "" },
                });
                break;
            case TextChunkKind.ToolCallDelta when chunk.Text is not null:
                yield return ToolFrame(new ChatToolCallDeltaDto
                {
                    Index = CallIndex(chunk),
                    Function = new ChatToolCallDeltaFunctionDto { Arguments = chunk.Text },
                });
                break;
            case TextChunkKind.NativeToolCall when chunk.ToolCall is { } call:
                int whole = CallIndex(chunk);
                // A call whose fragments already streamed carries its arguments in them; a whole call (from a filter) is sent once.
                if (_callsStreamed.Add(whole))
                {
                    yield return ToolFrame(new ChatToolCallDeltaDto
                    {
                        Index = whole,
                        Id = call.Id,
                        Type = "function",
                        Function = new ChatToolCallDeltaFunctionDto { Name = call.Name, Arguments = call.Arguments },
                    });
                }
                break;
            case TextChunkKind.ToolCallAbort when chunk.ToolCallIndex is { } aborted:
                // Its fragments cannot be taken back, so a whole call that later reuses the index must still reach the client, and this allows it. That is safe
                // because the one producer, DsmlCallsParser, faults terminally: nothing follows an abort in the same reply, so no streamed call reuses the index.
                // A producer that carried on after an abort would have to number its next call past the aborted one.
                _callsStreamed.Remove(aborted);
                break;
            case TextChunkKind.StopReason when chunk.Stop is { } stop:
                _finishReason = CompatEndpoints.ToFinishReason(stop);
                break;
            case TextChunkKind.Usage when chunk.Usage is { } usage:
                _usage = new ChatUsage { PromptTokens = usage.PromptTokens, CompletionTokens = usage.CompletionTokens };
                break;
        }
    }

    /// <summary>The frames that close the reply: the finish reason, then the usage frame when the client asked for it.</summary>
    public IEnumerable<ChatCompletionChunk> End()
    {
        yield return Frame(new ChatCompletionDelta(), _finishReason);
        if (_includeUsage && _usage is { } usage)
        {
            yield return new ChatCompletionChunk { Id = _id, Created = _created, Model = _model, Choices = [], Usage = usage };
        }
    }

    /// <summary>The call's index in the reply: the parser's when it gave one, else the next free index (a call from a filter arrives without one).</summary>
    /// <remarks>Assumes every <see cref="TextChunkKind.ToolCallDelta"/> fragment carries its call's index, as <c>ParsedEventTranslator</c> sets it today. An unindexed
    /// fragment would take a new index each time and split its call across several.</remarks>
    private int CallIndex(TextChunk chunk)
    {
        int index = chunk.ToolCallIndex ?? _nextCallIndex;
        _nextCallIndex = Math.Max(_nextCallIndex, index + 1);
        return index;
    }

    private ChatCompletionChunk ToolFrame(ChatToolCallDeltaDto call) => Frame(new ChatCompletionDelta { ToolCalls = [call] });

    private ChatCompletionChunk Frame(ChatCompletionDelta delta, string? finishReason = null) => new()
    {
        Id = _id,
        Created = _created,
        Model = _model,
        Choices = [new ChatCompletionChunkChoice { Delta = delta, FinishReason = finishReason }],
    };
}
