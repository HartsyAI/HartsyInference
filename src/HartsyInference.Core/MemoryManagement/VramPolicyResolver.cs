namespace HartsyInference.Core.MemoryManagement;

/// <summary>Expands a <see cref="VramTier"/> into a full <see cref="VramPolicy"/> and applies per-request overrides on top.</summary>
public static class VramPolicyResolver
{
    /// <summary>Chunk multiplier at <see cref="VramTier.Aggressive"/>.</summary>
    private const float AggressiveChunkScale = 0.5f;

    /// <summary>Chunk multiplier at <see cref="VramTier.Maximum"/>.</summary>
    private const float MaximumChunkScale = 0.25f;

    /// <summary>The lever values <paramref name="tier"/> implies, before any override.</summary>
    /// <remarks><see cref="VramTier.Auto"/> leaves every lever on <see cref="LeverState.Auto"/> so each consumer keeps
    /// making its own measured decision — that is what makes Auto byte-for-byte identical to the pre-policy engine.</remarks>
    public static VramPolicy Expand(VramTier tier) => tier switch
    {
        VramTier.Performance => new VramPolicy
        {
            Tier = tier,
            KeepResident = LeverState.On,
            PhaseUnload = LeverState.Off,
            WeightStreaming = LeverState.Off,
            Caches = CachePrecision.Full,
            ActivationOffload = LeverState.Off,
            FreeAfterGeneration = LeverState.Off,
            QuantizedCompute = LeverState.Off,
            MultiGpuSpill = LeverState.Off,
        },
        VramTier.Auto => new VramPolicy { Tier = tier },
        VramTier.Balanced => new VramPolicy
        {
            Tier = tier,
            PhaseUnload = LeverState.On,
            WeightStreaming = LeverState.Auto,
        },
        VramTier.Aggressive => new VramPolicy
        {
            Tier = tier,
            KeepResident = LeverState.Off,
            PhaseUnload = LeverState.On,
            WeightStreaming = LeverState.On,
            Caches = CachePrecision.Half,
            ChunkScale = AggressiveChunkScale,
        },
        VramTier.Maximum => new VramPolicy
        {
            Tier = tier,
            KeepResident = LeverState.Off,
            PhaseUnload = LeverState.On,
            WeightStreaming = LeverState.On,
            Caches = CachePrecision.Half,
            ActivationOffload = LeverState.On,
            ChunkScale = MaximumChunkScale,
            FreeAfterGeneration = LeverState.On,
            QuantizedCompute = LeverState.On,
            MultiGpuSpill = LeverState.On,
        },
        _ => new VramPolicy { Tier = VramTier.Auto },
    };

    /// <summary>Applies <paramref name="overrides"/> to <paramref name="basePolicy"/>; null or empty returns the base unchanged.</summary>
    /// <remarks>An override naming a <see cref="VramOverrides.Tier"/> re-expands from that tier first, so the remaining
    /// members refine the requested preset rather than the backend's.</remarks>
    public static VramPolicy Apply(VramPolicy basePolicy, VramOverrides? overrides)
    {
        ArgumentNullException.ThrowIfNull(basePolicy);
        if (overrides is null || overrides.IsEmpty)
        {
            return basePolicy;
        }
        VramPolicy policy = overrides.Tier is VramTier tier ? Expand(tier) : basePolicy;
        return policy with
        {
            KeepResident = overrides.KeepResident ?? policy.KeepResident,
            PhaseUnload = overrides.PhaseUnload ?? policy.PhaseUnload,
            WeightStreaming = overrides.WeightStreaming ?? policy.WeightStreaming,
            PrefetchAhead = overrides.PrefetchAhead ?? policy.PrefetchAhead,
            HeadroomBytes = overrides.HeadroomBytes ?? policy.HeadroomBytes,
            Caches = overrides.Caches ?? policy.Caches,
            ActivationOffload = overrides.ActivationOffload ?? policy.ActivationOffload,
            ChunkScale = overrides.ChunkScale ?? policy.ChunkScale,
            FreeAfterGeneration = overrides.FreeAfterGeneration ?? policy.FreeAfterGeneration,
        };
    }

    /// <summary>The tier a device with <paramref name="totalVramBytes"/> starts from under <see cref="VramTier.Auto"/>.</summary>
    public static VramPolicy ForDevice(long totalVramBytes) => Expand(GpuVramClass.Seed(totalVramBytes));

    /// <summary>Bridges the legacy three-state mode onto a policy, so callers still on <see cref="LowVramMode"/> resolve identically.</summary>
    public static VramPolicy FromLegacyMode(LowVramMode mode) => mode switch
    {
        LowVramMode.ForceOn => Expand(VramTier.Auto) with { WeightStreaming = LeverState.On },
        LowVramMode.ForceOff => Expand(VramTier.Auto) with { WeightStreaming = LeverState.Off },
        _ => Expand(VramTier.Auto),
    };

    /// <summary>The legacy mode <paramref name="policy"/>'s streaming lever corresponds to, for the many call sites still switching on <see cref="LowVramMode"/>.</summary>
    public static LowVramMode ToLegacyMode(VramPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        return policy.WeightStreaming switch
        {
            LeverState.On => LowVramMode.ForceOn,
            LeverState.Off => LowVramMode.ForceOff,
            _ => LowVramMode.Auto,
        };
    }
}
