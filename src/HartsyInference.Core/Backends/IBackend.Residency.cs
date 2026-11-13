using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Device residency of activations and weights: pinning, offload, preload, streaming cache and freeing.
public partial interface IBackend
{
    /// <summary>True when the backend has F16-activation kernels (Linear/Gelu/norms/cast); DiT F16 fast paths must gate on this.</summary>
    bool SupportsF16Activations => false;

    /// <summary>True when this backend can hold a weight of <paramref name="dtype"/> <b>packed</b> and dequantize it inside a GEMM, rather than needing it materialized wide before the model loads.</summary>
    /// <remarks><para>This is what decides whether a quantized checkpoint costs its on-disk size or four times it. A
    /// backend answering false for a dtype does not fail — the loader dequantizes on the host instead, which is slower
    /// and far larger but correct. Answering true for a dtype with no kernel is the failure that matters: the weight
    /// reaches a GEMM that cannot read it and the model dies mid-generation rather than at load.</para>
    /// <para>The default is the honest one for a backend with no packed-weight kernels at all.</para></remarks>
    bool SupportsResidentQuant(DType dtype) => false;

    /// <summary>The same question for a specific weight, for formats the dtype alone cannot name: an MXFP8 weight is plain F8E4M3 with its block scales on <see cref="Tensor.QuantInfo"/>. The default defers to the dtype answer.</summary>
    bool SupportsResidentQuant(Tensor weight) => SupportsResidentQuant(weight.DType);

    /// <summary>Marks a tensor's activation as surviving <see cref="FreeActivations()"/>, for cross-step state living only on-device.</summary>
    void PinActivation(Tensor tensor) { }

    /// <summary>Removes a <see cref="PinActivation"/> mark. No-op on host backends.</summary>
    void UnpinActivation(Tensor tensor) { }

    /// <summary>Materializes a cached activation to host and releases its device copy; the tensor reloads on its next
    /// use, so this is the named spelling of the <c>_ = t.DataPointer</c> host-materialize idiom (it also clears the
    /// <see cref="PinActivation"/> mark, since after a real D2H the host copy is authoritative). No-op on host
    /// backends.</summary>
    void OffloadActivation(Tensor tensor) { }

    /// <summary>Offloads cached activations largest-first until <paramref name="targetBytes"/> of device memory is
    /// released, returning the bytes actually freed. Safe points only, on the owning thread — never mid-op, and never
    /// while a step graph is live (a captured graph bakes activation pointers and a reload returns a new one). See
    /// docs/Research/MEMORY_SCHEDULING_SERVING.md §9. Returns 0 on host backends.</summary>
    long OffloadActivations(long targetBytes) => 0;

    /// <summary>Frees specific weight tensors (no-op on CPU) — call between phases to reclaim VRAM (e.g. UNet weights before VAE decode).</summary>
    void FreeWeights(IEnumerable<Tensor> weights) { }

    /// <summary>Frees cached activation buffers while keeping preloaded weights, so multi-step pipelines can reclaim memory between steps.</summary>
    void FreeActivations() { }

    /// <summary>As <see cref="FreeActivations()"/> but with pool-trim control; pass false on hot per-step cleanups to skip a driver re-map.</summary>
    void FreeActivations(bool trimPool) => FreeActivations();

    /// <summary>Returns pool-reserved-but-free device memory to the driver WITHOUT clearing the activation cache — safe mid-computation.</summary>
    void TrimMemoryPool() { }

    /// <summary>Synchronizes pending attention work and discards only backend-library attention execution plans
    /// and their persistent workspaces. Model weights, activation buffers, convolution plans, retry diagnostics,
    /// and cumulative dispatch counters are preserved. Use at a model-phase boundary when a large text encoder
    /// needs the memory held by fused-attention plans without paying for a full device-cache eviction.</summary>
    void ReleaseAttentionExecutionCache() { }

    /// <summary>Frees EVERY cached device allocation for a clean VRAM slate; do NOT call while any model is still in use.</summary>
    void FreeAllDeviceMemory() { }

    /// <summary>Makes every op that starts from now on fail with <see cref="ObjectDisposedException"/>, while the op in
    /// flight finishes. A teardown that could not wait for a job calls this first, so the job stops cleanly at its next
    /// op instead of running against freed device memory. A no-op on a backend with nothing to refuse.</summary>
    void RefuseNewOps() { }

    /// <summary>Free device memory in bytes (0 if not a device backend). For diagnostics / adaptive tiling.</summary>
    /// <remarks>The free half of <see cref="GetVramInfo"/>, so a backend that answers one answers both. It used to
    /// be an independent <c>0</c>, which is not "unknown" to the code that reads it — it is "nothing fits", and it
    /// kept every non-CUDA backend's callers on their smallest path forever.</remarks>
    long FreeMemoryBytes() => GetVramInfo().FreeBytes;

    /// <summary>Device memory as the driver, the allocator pool and the residency cache see it, for diagnostics. The default
    /// knows only <see cref="GetVramInfo"/> and reports -1 for the rest.</summary>
    DeviceMemoryUsage GetDeviceMemoryUsage()
    {
        (long free, long total) = GetVramInfo();
        return new DeviceMemoryUsage(free, total, -1, -1, -1, -1);
    }

    /// <summary>Pre-uploads weights into the weight cache so ops avoid re-uploading; pair with <see cref="FreeWeights"/>.</summary>
    void PreloadWeights(IEnumerable<Tensor> weights) { }

    /// <summary>True when the weight is already on the device in the weight cache, so a preload would not upload it.</summary>
    bool IsWeightResident(Tensor weight) => false;

    /// <summary>Pre-uploads weights as groups a backend may place in ONE device allocation each, such as a MoE layer's
    /// routed experts of one projection; pair with <see cref="FreeWeights"/>, member by member. The members stay ordinary
    /// weights: every op finds them by tensor as before. The default preloads each member on its own.</summary>
    /// <remarks>Thousands of small weights allocated one by one waste device memory, because the driver rounds every
    /// allocation up: 18% to 62% for Qwen3-30B-A3B's experts, enough that an 18.5 GB checkpoint no longer fits 24 GB.
    /// A group's allocation is returned only when its LAST member is freed, so freeing part of a group returns nothing; use
    /// groups for weights that live and die together, and keep weights evicted one at a time (an expert cache) out of them.</remarks>
    void PreloadWeightGroups(IEnumerable<IReadOnlyList<Tensor>> groups)
    {
        foreach (IReadOnlyList<Tensor> group in groups)
        {
            PreloadWeights(group);
        }
    }

    /// <summary>Streaming cache overlapping weight uploads with compute; <c>null</c> means fall back to <see cref="PreloadWeights"/>.</summary>
    IStreamingWeightCache? StreamingCache => null;
}
