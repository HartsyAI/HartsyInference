namespace HartsyInference.Core.MemoryManagement;

/// <summary>Per-request VRAM overrides. Every member is nullable, and null means "follow the backend's policy" — so an advanced caller can pin one lever without restating the rest.</summary>
/// <remarks>Deliberately carries only the runtime-scoped levers. <see cref="VramPolicy.QuantizedCompute"/> and
/// <see cref="VramPolicy.MultiGpuSpill"/> are baked at model construction, so varying them per request would force a
/// cached-pipeline rebuild and a multi-GB re-upload; they stay backend-scoped.</remarks>
public sealed record VramOverrides
{
    /// <summary>Replaces the whole tier for this request; individual members below still win over it.</summary>
    public VramTier? Tier { get; init; }

    /// <inheritdoc cref="VramPolicy.KeepResident"/>
    public LeverState? KeepResident { get; init; }

    /// <inheritdoc cref="VramPolicy.PhaseUnload"/>
    public LeverState? PhaseUnload { get; init; }

    /// <inheritdoc cref="VramPolicy.WeightStreaming"/>
    public LeverState? WeightStreaming { get; init; }

    /// <inheritdoc cref="VramPolicy.PrefetchAhead"/>
    public int? PrefetchAhead { get; init; }

    /// <inheritdoc cref="VramPolicy.HeadroomBytes"/>
    public long? HeadroomBytes { get; init; }

    /// <inheritdoc cref="VramPolicy.Caches"/>
    public CachePrecision? Caches { get; init; }

    /// <inheritdoc cref="VramPolicy.ActivationOffload"/>
    public LeverState? ActivationOffload { get; init; }

    /// <inheritdoc cref="VramPolicy.ChunkScale"/>
    public float? ChunkScale { get; init; }

    /// <inheritdoc cref="VramPolicy.FreeAfterGeneration"/>
    public LeverState? FreeAfterGeneration { get; init; }

    /// <summary>True when nothing is pinned, so the caller can skip re-resolving entirely.</summary>
    public bool IsEmpty => Tier is null && KeepResident is null && PhaseUnload is null && WeightStreaming is null
        && PrefetchAhead is null && HeadroomBytes is null && Caches is null && ActivationOffload is null
        && ChunkScale is null && FreeAfterGeneration is null;
}
