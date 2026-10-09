namespace HartsyInference.Core.Backends;

/// <summary>
/// Residency cache for experts whose weights are already host-resident. Admitting an expert records its
/// <see cref="ExpertWeights"/> and pins it through the shared residency path; no bytes are copied, so the upload and evict
/// hooks are no-ops and every transfer fence has already completed. Use it when the CPU executes the experts, so the planner
/// and the lease semantics stay the same as on a device cache.
/// </summary>
public sealed class HostExpertCache : ExpertCacheBase, IResidencyAwareExpertCache
{
    private static readonly object CompletedFence = new();

    /// <summary>Creates a host cache with a residency budget and the banks it resolves experts from.</summary>
    /// <param name="budgetBytes">Upper bound on the resident bytes the cache accounts for.</param>
    /// <param name="banks">Banks to register up front; more can be registered later.</param>
    public HostExpertCache(long budgetBytes, IEnumerable<ExpertBank>? banks = null) : base(budgetBytes, banks)
    {
    }

    /// <summary>Nothing is copied: the expert's tensors are already host memory, so there is no pending upload.</summary>
    protected override object? BeginUpload(ExpertWeights weights) => null;

    /// <summary>No pending upload exists to wait for.</summary>
    protected override void AwaitUpload(object pending)
    {
    }

    /// <summary>Evicting drops the cache entry only; the host tensors belong to the model and stay alive.</summary>
    protected override void Evict(ExpertWeights weights)
    {
    }

    /// <summary>No device work is in flight, so the fence is already complete.</summary>
    protected override object RecordFence() => CompletedFence;

    /// <summary>Always complete.</summary>
    protected override bool IsFenceDone(object fence) => true;

    /// <summary>Always complete.</summary>
    protected override void WaitFence(object fence)
    {
    }

    /// <summary>Nothing to release.</summary>
    protected override void DestroyFence(object fence)
    {
    }

    /// <summary>Nothing is in flight to drain.</summary>
    protected override void Drain()
    {
    }
}
