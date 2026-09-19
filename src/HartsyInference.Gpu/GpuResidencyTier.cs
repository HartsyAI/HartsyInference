namespace HartsyInference.Gpu;

/// <summary>Which cache holds a tensor's device copy, if either.</summary>
/// <remarks>One value rather than two independent flags, because the tiers are mutually exclusive by construction:
/// a lookup checks weights first, so a tensor in both would have its activation — the newer bytes — shadowed by a
/// stale weight forever. Keeping "both" unrepresentable is what stops that state from being described as normal.
/// Weight demotion exists to maintain exactly this invariant.</remarks>
public enum GpuResidencyTier
{
    /// <summary>No device copy; the next read uploads one.</summary>
    None,

    /// <summary>Resident as a weight — preloaded or promoted, and kept until explicitly freed.</summary>
    Weight,

    /// <summary>Resident as an activation — an op's output, live until rebound, offloaded or disposed.</summary>
    Activation,
}
