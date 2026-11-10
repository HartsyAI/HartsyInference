namespace HartsyInference.Engine.Worlds;

/// <summary>What one batch run is given: the resolved plan, the actions queued before the run, and the phase that owns the run's intermediates.</summary>
public sealed record OneShotClipRequest
{
    /// <summary>The resolved plan the clip renders; <see cref="WorldPlan.Request"/> carries the seed frame.</summary>
    public required WorldPlan Plan { get; init; }

    /// <summary>The actions queued before the stream started, in the order they were sent.</summary>
    public required IReadOnlyList<string> Actions { get; init; }

    /// <summary>Owns the run's intermediates. The pipeline registers each one here and never disposes the phase; it is released
    /// when the run ends, whether it succeeded, failed or was cancelled.</summary>
    public required WorldPhase Phase { get; init; }
}
