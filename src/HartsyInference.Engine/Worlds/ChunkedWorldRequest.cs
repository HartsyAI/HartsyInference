namespace HartsyInference.Engine.Worlds;

/// <summary>What a stream is opened with: the resolved plan, the actions sent before the stream started, and the phase that
/// owns what the stream's open acquires.</summary>
public sealed record ChunkedWorldRequest
{
    /// <summary>The resolved plan; <see cref="WorldPlan.Request"/> carries the seed frame and the opening camera state.</summary>
    public required WorldPlan Plan { get; init; }

    /// <summary>The actions sent before the stream started, in the order they were sent. The stream applies them before its
    /// first chunk.</summary>
    public required IReadOnlyList<string> Actions { get; init; }

    /// <summary>Owns the intermediates of the open that outlive it. The pipeline registers each one here and never disposes
    /// the phase; it is released when the session ends, whether the stream finished, failed or was cancelled.</summary>
    public required WorldPhase Phase { get; init; }
}
