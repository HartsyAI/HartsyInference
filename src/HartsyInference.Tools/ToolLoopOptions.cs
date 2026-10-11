using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools;

/// <summary>Settings for one <see cref="ToolLoopRun"/>: how many model rounds, a gate in front of each call, and the prefix for ids the loop has to invent.</summary>
public sealed record ToolLoopOptions
{
    /// <summary>Model invocations allowed per run.</summary>
    public int MaxRounds { get; init; } = ToolLoop.DefaultMaxRounds;

    /// <summary>Asked before each call runs; a denial is not dispatched and its result is fed back to the model. Null allows every call.</summary>
    public Func<NativeToolCall, CancellationToken, ValueTask<ToolCallDecision>>? OnBeforeToolCall { get; init; }

    /// <summary>Prefix for an id the loop assigns to a call that arrives without one, or with one already used in the run.</summary>
    public string IdPrefix { get; init; } = "call_";
}

/// <summary>What <see cref="ToolLoopOptions.OnBeforeToolCall"/> decided: run the call, or refuse it and feed <see cref="DenialResult"/> back as its result.</summary>
public readonly record struct ToolCallDecision(bool Allowed, string? DenialResult)
{
    /// <summary>Run the call.</summary>
    public static ToolCallDecision Allow => new(true, null);

    /// <summary>Refuse the call; <paramref name="result"/> is the tool result the model reads.</summary>
    public static ToolCallDecision Deny(string result) => new(false, result);
}
