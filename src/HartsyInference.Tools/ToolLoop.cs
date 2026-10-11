using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Tools;

/// <summary>The agent loop: streams a model turn, dispatches every call it ends on through an <see cref="IToolDispatcher"/>, appends the assistant turn and one <see cref="TextRole.Tool"/> message per result, and asks the model again, up to <see cref="ToolLoopOptions.MaxRounds"/> model invocations. Use <see cref="Create(ITextService, ModelSpec, TextRequest, IToolDispatcher, ToolLoopOptions?)"/> over an engine, or <see cref="Create(Func{TextRequest, CancellationToken, IAsyncEnumerable{TextChunk}}, TextRequest, IToolDispatcher, ToolLoopOptions?)"/> over any provider's stream.</summary>
public static class ToolLoop
{
    /// <summary>Default model invocations per run.</summary>
    public const int DefaultMaxRounds = 8;

    /// <summary>Prefix of the round-limit status text.</summary>
    public const string RoundLimitPrefix = "tool_loop:";

    /// <summary><see cref="TextStatus.Phase"/> of the round-limit chunk.</summary>
    public const string RoundLimitPhase = "tool_loop";

    /// <summary>A run over the engine's text service for one model.</summary>
    public static ToolLoopRun Create(ITextService text, ModelSpec spec, TextRequest request, IToolDispatcher tools, ToolLoopOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(spec);
        return Create((r, ct) => text.StreamAsync(spec, r, ct), request, tools, options);
    }

    /// <summary>A run over any model turn: <paramref name="stream"/> streams one round for the request it is given, with the conversation so far.</summary>
    public static ToolLoopRun Create(Func<TextRequest, CancellationToken, IAsyncEnumerable<TextChunk>> stream, TextRequest request, IToolDispatcher tools, ToolLoopOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(tools);
        ToolLoopOptions resolved = options ?? new ToolLoopOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(resolved.MaxRounds, 1);
        ArgumentNullException.ThrowIfNull(resolved.IdPrefix);
        return new ToolLoopRun(stream, request, tools, resolved);
    }

    /// <summary>Runs the loop over <paramref name="text"/> for <paramref name="request"/>; <paramref name="request"/>.Tools, when set, is offered as-is, else <paramref name="registry"/>.Definitions.</summary>
    public static IAsyncEnumerable<TextChunk> RunAsync(ITextService text, ModelSpec spec, TextRequest request, ToolRegistry registry,
        int maxRounds = DefaultMaxRounds, CancellationToken cancel = default)
        => Create(text, spec, request, registry, new ToolLoopOptions { MaxRounds = maxRounds }).RunAsync(cancel);
}
