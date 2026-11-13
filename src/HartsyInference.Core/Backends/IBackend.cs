using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Backend interface all model code programs against (CPU SIMD, CUDA PTX/cuBLAS); operations are eager, returning when complete.</summary>
public partial interface IBackend : IDisposable
{
    /// <summary>The device this backend targets.</summary>
    DeviceKind Device { get; }

    /// <summary>Stable identity of the physical device this backend actually bound to, for callers that must tell two
    /// backends sharing one GPU apart from two backends on two GPUs.</summary>
    /// <remarks>Deliberately not the selector that was requested. A host composing <c>"vulkan:0"</c> out of its own
    /// settings names the device it ASKED for, and the two part company whenever selection is left to the engine: a
    /// bare <c>vulkan</c> or <c>auto</c> ranks the devices rather than taking raw index 0. Keying a sharing map on the
    /// request instead of on this reads one shared GPU as two, and the VRAM it was meant to protect is gone before
    /// anything notices.</remarks>
    string DeviceKey => Device.ToString().ToLowerInvariant();

    /// <summary>Count of lazy device-to-host syncs since <see cref="ResetD2hSyncCount"/>; ~0 GPU-resident, 0 with no device sync.</summary>
    long GetD2hSyncCount() => 0;

    /// <summary>Resets the D2H sync counter. No-op on backends without a device sync.</summary>
    void ResetD2hSyncCount() { }

    /// <summary>Device memory (free, total) in bytes; (0,0) when not applicable (CPU backend).</summary>
    (long FreeBytes, long TotalBytes) GetVramInfo() => (0, 0);

    /// <summary>When true, F32 GEMMs use full 32-bit compute instead of TF32 (needed by e.g. Zonos); no-op without TF32.</summary>
    bool HighPrecisionGemm { get => false; set { } }

    /// <summary>Low-VRAM lever: when true (the default) a weight's fp8/quant→F16 cast is kept resident instead of
    /// being recomputed per GEMM. Caching is faster but holds BOTH the packed weight and its cast on the device;
    /// turning it off trades recompute for roughly a third of the weight footprint, which is what makes a large fp8
    /// DiT fit a single 24 GB card. No-op on a backend that does not cache casts.</summary>
    /// <remarks>On the interface because callers set it on EVERY placement backend — a text encoder or VAE on its own
    /// GPU needs the same lever or the saving silently does not apply there. Reaching that through a type test means
    /// each new backend is invisible to the policy until someone adds another arm to the chain.</remarks>
    bool CacheWeightCasts { get => true; set { } }

    /// <summary>Whether the widened (F16 or F32) copy of a GGUF-quantized weight is cached like any other weight cast.
    /// The text path turns it off on a backend whose decode GEMV reads the packed blocks (Vulkan): there the copy only
    /// serves prefill, and keeping it resident would cost two to four times the packed weight. A diffusion model that
    /// runs every step at a large M keeps the default and dequantizes once. No-op on a backend that does not cache casts.</summary>
    bool CacheQuantizedWeightCasts { get => true; set { } }

    /// <summary>Whether this backend's GEMM consumes fp8 operands natively instead of widening them first. Recipes
    /// read it to decide whether an fp8 checkpoint can keep its weights packed.</summary>
    /// <remarks>A property rather than a <see cref="BackendCapabilities"/> field because the backends that have such
    /// a path gate it on a settable flag, and a capability snapshot taken at construction would disagree with the
    /// flag the GEMM actually reads.</remarks>
    bool NativeFp8Gemm => false;

    /// <summary>When true, <see cref="Conv2D"/> pads the width axis with wrapped edge pixels instead of zeros, so
    /// the model sees a horizontally-continuous canvas — the standard "seamless tiling" trick for textures/
    /// patterns. Independent of <see cref="SeamlessTilingY"/> (SwarmUI core's "X-Only"/"Y-Only"/"true" modes).
    /// No-op on backends without a padding intercept (this is a request-scoped opt-in, not a capability).</summary>
    bool SeamlessTilingX { get => false; set { } }

    /// <summary>Same as <see cref="SeamlessTilingX"/> for the height axis.</summary>
    bool SeamlessTilingY { get => false; set { } }

    /// <summary>Which operations and dtypes this backend supports, so callers can route work to it or fall back.</summary>
    BackendCapabilities Capabilities { get; }

    /// <summary>Matrix multiply: output = a @ b</summary>
    void MatMul(Tensor output, Tensor a, Tensor b);

    /// <summary>Batched matrix multiply: output[i] = a[i] @ b[i]</summary>
    void BatchedMatMul(Tensor output, Tensor a, Tensor b);

    /// <summary>Linear layer: output = input × weight^T + bias. [M,K]×[N,K]→[M,N], bias [N] optional; leading batch dims also work.</summary>
    void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias);

    /// <summary>GEMM against a contiguous ROW RANGE of <paramref name="weight"/> — rows <c>[weightRowOffset,
    /// weightRowOffset + weightRowCount)</c> of a <c>[outDim, inDim]</c> weight, with <paramref name="bias"/> sliced
    /// to match. Exists so a fused projection can be evaluated in parts WITHOUT materializing a sub-tensor: GPU
    /// caching is keyed on tensor identity, so a real slice would upload a second copy of an already-resident weight
    /// (see <c>MiniMaxH3Transformer</c>'s chunked attention). The default fallback copies the rows out; backends that
    /// can offset a device pointer override it.</summary>
    unsafe void LinearWeightRows(Tensor output, Tensor input, Tensor weight, Tensor? bias, int weightRowOffset, int weightRowCount)
    {
        // Rows are the outermost axis, so the window is a byte offset — including for a block-quantized weight, whose
        // blocks never span rows. This used to refuse those outright, which is what made MiniMax-H3's chunked
        // projections unreachable from a GGUF build: the chunking is how the model runs at all, and the weight it
        // chunks is the one the quantization applies to.
        using Tensor rows = weight.SliceRows(weightRowOffset, weightRowCount);
        rows.QuantInfo = weight.QuantInfo?.SliceRows(weightRowOffset, weightRowCount, "the weight passed to LinearWeightRows");
        // The adjunct's down matrix spans the whole input dimension the window keeps; only its up matrix is narrowed.
        rows.LowRankAdjunct = weight.LowRankAdjunct?.SliceRows(weightRowOffset, weightRowCount);
        // Bias is 1-D, one value per output channel, so its window is the same row range read as elements.
        using Tensor? biasRows = bias?.SliceRows(weightRowOffset, weightRowCount);
        Linear(output, input, rows, biasRows);
    }

    /// <summary>Element-wise add scalar: <c>out[i] = in[i] + c</c>.</summary>
    unsafe void AddScalar(Tensor output, Tensor input, float scalar)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("AddScalar default fallback only supports F32.");
        long n = input.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < n; i++) pOut[i] = pIn[i] + scalar;
    }

    /// <summary>Several projections of the SAME <paramref name="input"/>, so a backend whose Linear preprocesses the
    /// activation can do that once for the group instead of once per weight. Results are identical to calling
    /// <see cref="Linear"/> per op — this only moves shared work, it does not change arithmetic. Default: exactly
    /// that loop.</summary>
    /// <remarks>Written for the int8 chain, where every Linear rotates and row-quantizes its activation before the
    /// GEMM: an attention that projects gate/q/k/v from one tensor pays that pass four times for one answer.</remarks>
    unsafe void LinearMulti(Tensor input, ReadOnlySpan<LinearOp> ops)
    {
        foreach (LinearOp op in ops) Linear(op.Output, input, op.Weight, op.Bias);
    }

    /// <summary>Element-wise addition: output = a + b</summary>
    void Add(Tensor output, Tensor a, Tensor b);

    /// <summary>Element-wise multiplication: output = a * b</summary>
    void Mul(Tensor output, Tensor a, Tensor b);

    /// <summary>Scalar multiplication: output = input * scalar</summary>
    void Scale(Tensor output, Tensor input, float scalar);

    /// <summary>Element-wise clamp: output = clamp(input, min, max)</summary>
    void Clamp(Tensor output, Tensor input, float min, float max);

    /// <summary>Batched 2D transpose: [B, D1, D2] → [B, D2, D1].</summary>
    void Transpose2D(Tensor output, Tensor input, int d1, int d2);

    /// <summary>4D permute swapping dims 1 and 2: [B, S, H, D] → [B, H, S, D].</summary>
    void Permute0213(Tensor output, Tensor input, int s, int h, int d);

    /// <summary>Broadcast add: hidden [B, C, ...spatial] += bias [B, C] in-place.</summary>
    void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial);

    /// <summary>Concatenate tensors along the specified dimension.</summary>
    void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim);

    /// <summary>Splits a nonempty contiguous F32, F16, or BF16 tensor into nonempty chunks along <paramref name="dim"/>. Outputs must have the same dtype and rank as the input, match every non-split dimension, exactly partition the split dimension, and share no storage with the input or one another. Payload bits are copied unchanged.</summary>
    void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim);

    /// <summary>Copy tensor data to a different device.</summary>
    void CopyTo(Tensor destination, Tensor source);

    /// <summary>Copies <paramref name="source"/> (produced/resident on <paramref name="sourceBackend"/>) into
    /// <paramref name="destination"/> on THIS backend — the cross-backend boundary handoff for multi-GPU
    /// placement (LLM stage boundaries, CFG-branch results). The default stages through the host: forcing
    /// <c>source.DataPointer</c> fires the source backend's lazy D2H sync, then <see cref="CopyTo"/> uploads
    /// here. Device-capable backends override with a direct device-to-device path when available.</summary>
    unsafe void CopyFromPeer(Tensor destination, Tensor source, IBackend sourceBackend)
    {
        _ = source.DataPointer;
        CopyTo(destination, source);
    }

    /// <summary>Fill a tensor with a constant value.</summary>
    void Fill(Tensor tensor, float value);

    /// <summary>Waits for all pending GPU work (no-op on CPU) — call at phase boundaries so deferred frees land before large allocations.</summary>
    void Sync() { }

    /// <summary>Dequantizes a packed single-expert NVFP4 bank <c>[1, out, in/2]</c>, its block scales and its global scale into the
    /// BF16 <c>[out, in]</c> <paramref name="destination"/> on the device. Returns false and writes nothing when the backend has no
    /// such kernel, so the caller runs its host dequant. The device result is registered against <paramref name="destination"/> for
    /// the next <see cref="Linear"/> and released when <paramref name="destination"/> is disposed. Its host bytes are not written on this path.</summary>
    bool TryDequantNvfp4ToBf16(Tensor packed, Tensor blockScale, Tensor globalScale, Tensor destination) => false;

    /// <summary>Marks the end of the work issued so far and returns a handle <see cref="WaitFence"/> can block on; 0 on a
    /// backend whose ops have finished by the time they return (the default), where there is nothing to wait for.</summary>
    /// <remarks>Lets a caller bound how far the host runs ahead of the device without draining it: waiting on a fence
    /// recorded a few steps back keeps the steps since then queued. Every non-zero fence goes back through
    /// <see cref="ReleaseFence"/> exactly once, waited on or not.</remarks>
    nint RecordFence() => 0;

    /// <summary>Blocks the calling thread until the work issued before <paramref name="fence"/> was recorded has
    /// finished; returns at once for 0.</summary>
    void WaitFence(nint fence) { }

    /// <summary>Gives back a fence from <see cref="RecordFence"/>; 0 is ignored.</summary>
    void ReleaseFence(nint fence) { }

    /// <summary>Clears the per-op profile accumulator so a later <see cref="DumpOpProfile"/> covers only work after this point.</summary>
    /// <remarks>Lets a pipeline window the profiler onto its steady-state loop. Without it the table folds in
    /// one-time setup — text encode, weight residency warm-up — whose cost varies enough run to run that
    /// differencing two runs to cancel it does not work (measured on MiniMax-H3: a 3-step run spent MORE total
    /// time in <c>Linear</c> than a 6-step run, purely from differing H2D-miss counts). No-op when profiling is off.</remarks>
    void ResetOpProfile() { }

    /// <summary>Writes the per-op profile accumulated so far, tagging the output with <paramref name="label"/>. No-op when profiling is off.</summary>
    void DumpOpProfile(string label) { }
}
