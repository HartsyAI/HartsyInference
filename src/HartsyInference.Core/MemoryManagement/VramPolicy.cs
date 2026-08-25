namespace HartsyInference.Core.MemoryManagement;

/// <summary>One generation's resolved VRAM behavior: a tier plus the fully-expanded lever values it implies. Produced by <see cref="VramPolicyResolver"/>; every modality reads this instead of its own environment variable.</summary>
/// <remarks>Levers split by when they bind. <see cref="KeepResident"/>, <see cref="PhaseUnload"/>,
/// <see cref="WeightStreaming"/>, <see cref="PrefetchAhead"/>, <see cref="HeadroomBytes"/>, <see cref="Caches"/>,
/// <see cref="ActivationOffload"/>, <see cref="ChunkScale"/> and <see cref="FreeAfterGeneration"/> are decided inside
/// the generation call, so they are safe to vary per request. <see cref="QuantizedCompute"/> and
/// <see cref="MultiGpuSpill"/> are baked at model construction — changing them means rebuilding a cached pipeline and
/// re-uploading its weights — so they are backend-scoped only.</remarks>
public sealed record VramPolicy
{
    /// <summary>The preset this policy expanded from.</summary>
    public required VramTier Tier { get; init; }

    /// <summary>Whether a pipeline keeps its weights on the device between generations.</summary>
    public LeverState KeepResident { get; init; } = LeverState.Auto;

    /// <summary>Whether a phase releases its weights at its boundary so the next phase gets the space.</summary>
    public LeverState PhaseUnload { get; init; } = LeverState.Auto;

    /// <summary>Whether denoiser blocks ride a sliding window instead of staying resident.</summary>
    public LeverState WeightStreaming { get; init; } = LeverState.Auto;

    /// <summary>In-flight upload depth for the streamed suffix; null lets the caller pick.</summary>
    public int? PrefetchAhead { get; init; }

    /// <summary>Bytes held back for activations and workspace; null uses the caller's own estimate.</summary>
    public long? HeadroomBytes { get; init; }

    /// <summary>Storage precision for cross-step caches.</summary>
    public CachePrecision Caches { get; init; } = CachePrecision.Auto;

    /// <summary>Whether cross-step state is paged to host as it is produced.</summary>
    public LeverState ActivationOffload { get; init; } = LeverState.Auto;

    /// <summary>Multiplier applied to a model's own chunk/tile size — the lever for activation-bound work, where streaming weights does nothing. 1.0 leaves the model default alone.</summary>
    public float ChunkScale { get; init; } = 1.0f;

    /// <summary>Whether a model's device memory is released once its generation finishes.</summary>
    public LeverState FreeAfterGeneration { get; init; } = LeverState.Auto;

    /// <summary>Whether quantized weights stay compressed on device with a transient dequant per call, instead of caching a dequantized copy. Construction-scoped.</summary>
    public LeverState QuantizedCompute { get; init; } = LeverState.Auto;

    /// <summary>Whether a model too large for one card may spill onto another configured, idle device. Construction-scoped.</summary>
    public LeverState MultiGpuSpill { get; init; } = LeverState.Auto;

    /// <summary>The tier's expansion with no lever overridden — what a backend uses before any request-level override.</summary>
    public static VramPolicy For(VramTier tier) => VramPolicyResolver.Expand(tier);

    /// <summary>One line naming the tier and only the levers that differ from it, for the per-generation log.</summary>
    /// <remarks>Listing every lever every generation would bury the one thing that is usually interesting — which
    /// setting the operator actually pinned — so a policy that is just its tier prints just its tier.</remarks>
    public string Describe()
    {
        VramPolicy preset = VramPolicyResolver.Expand(Tier);
        List<string> pinned = [];
        Add(pinned, nameof(KeepResident), KeepResident, preset.KeepResident);
        Add(pinned, nameof(PhaseUnload), PhaseUnload, preset.PhaseUnload);
        Add(pinned, nameof(WeightStreaming), WeightStreaming, preset.WeightStreaming);
        Add(pinned, nameof(ActivationOffload), ActivationOffload, preset.ActivationOffload);
        Add(pinned, nameof(FreeAfterGeneration), FreeAfterGeneration, preset.FreeAfterGeneration);
        Add(pinned, nameof(QuantizedCompute), QuantizedCompute, preset.QuantizedCompute);
        Add(pinned, nameof(MultiGpuSpill), MultiGpuSpill, preset.MultiGpuSpill);
        if (Caches != preset.Caches) pinned.Add($"{nameof(Caches)}={Caches}");
        if (Math.Abs(ChunkScale - preset.ChunkScale) > 1e-6f) pinned.Add($"{nameof(ChunkScale)}={ChunkScale:0.##}");
        if (PrefetchAhead is int prefetch) pinned.Add($"{nameof(PrefetchAhead)}={prefetch}");
        if (HeadroomBytes is long headroom) pinned.Add($"{nameof(HeadroomBytes)}={ByteFormat.Mb(headroom)}");
        return pinned.Count == 0 ? Tier.ToString() : $"{Tier} ({string.Join(", ", pinned)})";
    }

    private static void Add(List<string> into, string name, LeverState actual, LeverState preset)
    {
        if (actual != preset) into.Add($"{name}={actual}");
    }
}
