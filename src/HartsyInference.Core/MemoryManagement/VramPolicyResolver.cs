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

    /// <summary>The next tier to try after <paramref name="tier"/> ran out of VRAM, or null when there is nothing more aggressive left.</summary>
    /// <remarks>Ordered by how much they give up, not by the enum's numeric order — <see cref="VramTier.Performance"/>
    /// is the least aggressive despite being 0, and <see cref="VramTier.Auto"/> re-enters the ladder at Balanced
    /// because its whole point is that measurement already decided and the measurement was not enough.</remarks>
    public static VramTier? Escalate(VramTier tier) => tier switch
    {
        VramTier.Performance => VramTier.Balanced,
        VramTier.Auto => VramTier.Balanced,
        VramTier.Balanced => VramTier.Aggressive,
        VramTier.Aggressive => VramTier.Maximum,
        _ => null,
    };

    /// <summary>Escalates <paramref name="policy"/> one rung, preserving any explicitly pinned levers.</summary>
    /// <remarks>A caller who pinned a lever meant it, and an automatic retry is not the place to overrule them —
    /// so the pins are re-applied on top of the harder tier rather than discarded. The one exception is a lever
    /// pinned OFF that is exactly what the escalation needs: streaming pinned Off stays Off, because the operator
    /// asking for a loud failure gets a loud failure instead of a silent slow one.</remarks>
    public static VramPolicy? Escalate(VramPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        if (Escalate(policy.Tier) is not VramTier next)
        {
            return null;
        }
        VramPolicy preset = Expand(policy.Tier);
        VramPolicy escalated = Expand(next);
        return escalated with
        {
            KeepResident = Pinned(policy.KeepResident, preset.KeepResident) ?? escalated.KeepResident,
            PhaseUnload = Pinned(policy.PhaseUnload, preset.PhaseUnload) ?? escalated.PhaseUnload,
            WeightStreaming = Pinned(policy.WeightStreaming, preset.WeightStreaming) ?? escalated.WeightStreaming,
            ActivationOffload = Pinned(policy.ActivationOffload, preset.ActivationOffload) ?? escalated.ActivationOffload,
            FreeAfterGeneration = Pinned(policy.FreeAfterGeneration, preset.FreeAfterGeneration) ?? escalated.FreeAfterGeneration,
            QuantizedCompute = Pinned(policy.QuantizedCompute, preset.QuantizedCompute) ?? escalated.QuantizedCompute,
            MultiGpuSpill = Pinned(policy.MultiGpuSpill, preset.MultiGpuSpill) ?? escalated.MultiGpuSpill,
            PrefetchAhead = policy.PrefetchAhead,
            HeadroomBytes = policy.HeadroomBytes,
        };
    }

    /// <summary>The lever value when the caller pinned it away from its tier's preset, else null.</summary>
    private static LeverState? Pinned(LeverState actual, LeverState preset) => actual == preset ? null : actual;

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
