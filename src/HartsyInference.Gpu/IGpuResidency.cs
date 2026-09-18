using HartsyInference.Core.Tensors;

namespace HartsyInference.Gpu;

/// <summary>What a backend's device-residency cache can be asked to do without naming the type of its buffers.
///
/// <para><see cref="GpuResidencyCache{TBuffer}"/> is generic over the buffer handle — a <c>ulong</c> device pointer,
/// a Vulkan buffer object, an <c>id&lt;MTLBuffer&gt;</c> — because the cache's logic does not care what a buffer is,
/// only which tensor owns one. A backend base class does not care either, so it holds this instead and stays free of
/// the generic parameter.</para></summary>
public interface IGpuResidency : IDisposable
{
    /// <summary>Identifies this cache as the owner of the GPU bindings it plants on a tensor.
    ///
    /// <para>One host tensor can be resident on several devices at once — a weight promoted on two GPUs for a
    /// CFG-parallel step — and each device's cache must be able to release its own binding without disturbing the
    /// others. Every instance therefore gets a distinct key.</para></summary>
    nint BindingKey { get; }

    /// <summary>Device-to-host syncs performed since <see cref="ResetD2hSyncCount"/>. A step that is fully
    /// GPU-resident reports zero; anything above that is a host round-trip somebody did not intend.</summary>
    long D2hSyncCount { get; }

    /// <summary>Resets <see cref="D2hSyncCount"/>, so a caller can measure one step rather than a whole run.</summary>
    void ResetD2hSyncCount();

    /// <summary>Uploads weights ahead of first use, so no op pays a cache-miss transfer mid-generation.</summary>
    void PreloadWeights(IEnumerable<Tensor> weights);

    /// <summary>Releases the device copies of these weights. Pair with <see cref="PreloadWeights"/>: a component
    /// whose weights are freed at a phase boundary must be preloaded again before its next heavy use, or the first
    /// op pays a per-op miss that defeats the bulk upload.</summary>
    void FreeWeights(IEnumerable<Tensor> weights);

    /// <summary>Releases every cached device allocation this cache owns.</summary>
    void FreeAllCached();

    /// <summary>Marks a tensor's activation as surviving <see cref="OffloadActivations"/>, for state that must stay
    /// on the device across steps.</summary>
    void PinActivation(Tensor tensor);

    /// <summary>Removes a <see cref="PinActivation"/> mark.</summary>
    void UnpinActivation(Tensor tensor);

    /// <summary>Materializes cached activations to host, largest first, until <paramref name="targetBytes"/> has been
    /// released; returns the bytes actually freed. Skips pinned tensors. Safe points only — never mid-op, and never
    /// while a step graph is live, since a captured graph bakes activation addresses and a reload returns a new one.</summary>
    long OffloadActivations(long targetBytes);

    /// <summary>Runs the device-side cleanup queued by tensors that were finalized rather than disposed.
    ///
    /// <para>A finalizer runs on the finalizer thread, which cannot touch a device context, so the work is queued and
    /// drained here — on the owning thread, at a safe point. A backend that never drains its own bucket leaks every
    /// tensor its users forgot to dispose.</para></summary>
    void DrainFinalizerCleanup();

    /// <summary>Frees the buffers displaced by a rebind that no caller claimed.
    ///
    /// <para>Call at the start of an op, alongside <see cref="DrainFinalizerCleanup"/>: by then every previous op's
    /// cleanup has run, so anything still parked provably has no owner. A backend that never calls this leaks every
    /// buffer an op displaced, for the whole generation.</para></summary>
    void SweepOrphans();

    /// <summary>A one-line summary of cache occupancy and hit rate, for a log line or a diagnostic dump.</summary>
    string DiagnosticsSummary();
}
