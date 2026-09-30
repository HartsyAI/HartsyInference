using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Tools;

/// <summary>The agent loop: streams a model turn, dispatches every <see cref="NativeToolCall"/> it ends on through the <see cref="ToolRegistry"/>, appends the assistant turn (with <see cref="TextMessage.ToolCalls"/>) and one <see cref="TextRole.Tool"/> message per result, and asks the model again, up to <c>maxRounds</c> model invocations.</summary>
/// <remarks>Chunk shape: every round's <see cref="TextChunkKind.Chunk"/>, <see cref="TextChunkKind.NativeToolCall"/> and pass-through chunks are yielded as they stream; each round's own <see cref="TextChunkKind.Result"/> and <see cref="StopReason.ToolCall"/> stop are suppressed, and the loop ends with one <see cref="TextChunkKind.Result"/> holding the visible text of all rounds followed by one <see cref="TextChunkKind.StopReason"/>. Tool results are carried as <see cref="TextChunkKind.Status"/> chunks (no Engine kind fits and adding one is out of this package's reach): <see cref="TextChunk.Text"/> is <see cref="ToolResultPrefix"/> followed by the result, <see cref="TextChunk.Status"/> has phase <see cref="ToolResultPhase"/>, and <see cref="TextChunk.ToolCall"/> / <see cref="TextChunk.ToolCallIndex"/> identify the call. Hitting the round limit while the model still asks for a tool yields a <see cref="TextChunkKind.Status"/> chunk prefixed <see cref="RoundLimitPrefix"/>, the final result, and <see cref="StopReason.ToolCall"/>; that last call is not dispatched. An <see cref="StopReason.Error"/> or <see cref="StopReason.Cancelled"/> stop from the service is relayed as-is and ends the loop without a final <see cref="TextChunkKind.Result"/>.</remarks>
public static class ToolLoop
{
    /// <summary>Prefix of a tool-result <see cref="TextChunkKind.Status"/> chunk's text; the rest is the result.</summary>
    public const string ToolResultPrefix = "tool_result:";

    /// <summary><see cref="TextStatus.Phase"/> of a tool-result chunk.</summary>
    public const string ToolResultPhase = "tool_result";

    /// <summary>Prefix of the terminal <see cref="TextChunkKind.Status"/> chunk emitted when the round limit is hit.</summary>
    public const string RoundLimitPrefix = "tool_loop:";

    /// <summary><see cref="TextStatus.Phase"/> of the round-limit chunk.</summary>
    public const string RoundLimitPhase = "tool_loop";

    /// <summary>Default model invocations per <see cref="RunAsync"/>.</summary>
    public const int DefaultMaxRounds = 8;

    /// <summary>Runs the loop over <paramref name="text"/> for <paramref name="request"/>; <paramref name="request"/>.Tools, when set, is offered as-is, else <paramref name="registry"/>.Definitions.</summary>
    public static async IAsyncEnumerable<TextChunk> RunAsync(ITextService text, ModelSpec spec, TextRequest request, ToolRegistry registry,
        int maxRounds = DefaultMaxRounds, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(registry);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxRounds, 1);
        List<TextMessage> conversation = [.. request.Messages];
        IReadOnlyList<ToolDefinition> tools = request.Tools is { Count: > 0 } ? request.Tools : registry.Definitions;
        StringBuilder visible = new();
        int resultIndex = 0;
        for (int round = 1; round <= maxRounds; round++)
        {
            TextRequest roundRequest = request with { Messages = [.. conversation], Tools = tools };
            StringBuilder roundText = new();
            List<NativeToolCall> calls = [];
            StopReason stop = StopReason.Stop;
            await foreach (TextChunk chunk in text.StreamAsync(spec, roundRequest, cancel).WithCancellation(cancel).ConfigureAwait(false))
            {
                switch (chunk.Kind)
                {
                    case TextChunkKind.Chunk:
                        roundText.Append(chunk.Text);
                        visible.Append(chunk.Text);
                        yield return chunk;
                        break;
                    case TextChunkKind.NativeToolCall:
                        if (chunk.ToolCall is { } call) calls.Add(call);
                        yield return chunk;
                        break;
                    case TextChunkKind.Result:
                        break;
                    case TextChunkKind.StopReason:
                        stop = chunk.Stop ?? StopReason.Stop;
                        if (stop is StopReason.Error or StopReason.Cancelled)
                        {
                            yield return chunk;
                            yield break;
                        }
                        break;
                    default:
                        yield return chunk;
                        break;
                }
            }
            if (stop != StopReason.ToolCall || calls.Count == 0)
            {
                yield return new TextChunk { Kind = TextChunkKind.Result, Text = visible.ToString() };
                yield return new TextChunk { Kind = TextChunkKind.StopReason, Stop = stop };
                yield break;
            }
            if (round == maxRounds)
            {
                yield return new TextChunk
                {
                    Kind = TextChunkKind.Status,
                    Text = RoundLimitPrefix + "max_rounds=" + maxRounds.ToString(CultureInfo.InvariantCulture),
                    Status = new TextStatus(RoundLimitPhase),
                };
                yield return new TextChunk { Kind = TextChunkKind.Result, Text = visible.ToString() };
                yield return new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.ToolCall };
                yield break;
            }
            conversation.Add(new TextMessage { Role = TextRole.Assistant, Content = roundText.ToString(), ToolCalls = calls });
            foreach (NativeToolCall call in calls)
            {
                string result = await registry.InvokeAsync(call, cancel).ConfigureAwait(false);
                conversation.Add(new TextMessage { Role = TextRole.Tool, Content = result, ToolCallId = call.Id, Name = call.Name });
                yield return new TextChunk
                {
                    Kind = TextChunkKind.Status,
                    Text = ToolResultPrefix + result,
                    Status = new TextStatus(ToolResultPhase),
                    ToolCall = call,
                    ToolCallIndex = resultIndex++,
                };
            }
        }
    }
}
