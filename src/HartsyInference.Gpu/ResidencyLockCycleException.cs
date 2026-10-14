namespace HartsyInference.Gpu;

/// <summary>Thrown instead of deadlocking when two threads each hold a residency cache the other needs.</summary>
public sealed class ResidencyLockCycleException()
    : InvalidOperationException("Two threads each held a device residency cache the other one needed, which would "
        + "have deadlocked; this request was failed to let the other proceed. Ops on two backends must not read each "
        + "other's device-resident tensors from host at the same time.");
