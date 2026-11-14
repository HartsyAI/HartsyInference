using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Decode;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Autoregressive decode on device: step-graph capture, device-side positions, token ids, history and the fused decode-step epilogues.
public partial interface IBackend
{
    // A pipeline whose denoise step issues an IDENTICAL op sequence every step can capture it once and replay
    // with a single graph launch, collapsing thousands of per-op host calls. Contract: between Begin and
    // EndAndLaunch only async device work is issued (no DataPointer reads, no sync allocs); per-step-varying
    // inputs are refreshed into FIXED device buffers via CopyInto before each Launch. Defaults: unsupported.

    /// <summary>True when the backend can capture/replay a step graph (CUDA and Vulkan).</summary>
    bool StepGraphSupported => false;

    /// <summary>True when a captured step graph is ready to <see cref="StepGraphLaunch"/>.</summary>
    bool StepGraphReady => false;

    /// <summary>Begins capturing subsequent backend ops into the step graph.</summary>
    void StepGraphBegin() { }

    /// <summary>Ends capture, instantiates the graph, and launches it once (capture records without executing).</summary>
    void StepGraphEndAndLaunch() { }

    /// <summary>Replays the captured step graph (one launch call).</summary>
    void StepGraphLaunch() { }

    /// <summary>Drops the captured graph (shape/prompt change → recapture) and disables an in-flight capture.</summary>
    void StepGraphReset() { }

    /// <summary>Identity of the model occupying the shared step-graph slot; mismatch before <see cref="StepGraphLaunch"/> forces recapture.</summary>
    object? StepGraphOwner { get => null; set { } }

    /// <summary>Copies src's data into dst's EXISTING storage (preserving its buffer address) to refresh a captured graph's fixed inputs.</summary>
    unsafe void CopyInto(Tensor dst, Tensor src)
    {
        long bytes = dst.DType.ComputeByteCount(dst.ElementCount);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, bytes, bytes);
    }

    // A captured decode graph must NOT re-bake the per-step KV position into its kernel params, or every replay
    // would append/attend at the capture-time position. Instead the self-attention KV-append and flash kernels
    // read the position from a small device buffer {kvLen, qOffset, status}, refreshed (outside capture) each step. This
    // lets ONE captured graph replay at every step. Defaults: unsupported (eager backends use the host ints).

    /// <summary>True when the backend supports device-position decode-graph replay for every graph consumer (text LLM, SSM, CSM, MusicGen). CUDA only; Vulkan reports its text-LLM graph through <see cref="LlmGraphDecodeSupported"/> instead.</summary>
    bool GraphDecodeSupported => false;

    /// <summary>True when the generic text-LLM decode step (<c>GenericTransformer.ForwardGraphDecodeStep</c>) can be captured through <see cref="CaptureGraph"/> and replayed. Separate from <see cref="GraphDecodeSupported"/> so a backend can enable it for text LLMs without also routing the SSM, CSM and MusicGen paths (which gate on that flag) into a capture they were never validated against.</summary>
    bool LlmGraphDecodeSupported => GraphDecodeSupported;

    /// <summary>True when text-LLM graph decode should be used without being asked for (<c>numerics.graphDecode</c> / a request's <c>GraphDecode</c>): the backend measured it faster than its eager loop for every model it supports. Applies to requests of at least <see cref="LlmGraphDecodeDefaultMinTokens"/> new tokens; a failed capture then falls back to eager instead of failing the request. CUDA keeps the explicit opt-in.</summary>
    bool LlmGraphDecodeDefault => false;

    /// <summary>Fewest new tokens for which <see cref="LlmGraphDecodeDefault"/> turns graph decode on. Every backend gets 512 unless it measured a lower one; Vulkan overrides it with its measured 128. CUDA keeps the explicit opt-in (its <see cref="LlmGraphDecodeDefault"/> is false), so this value only matters on backends that default graph decode on. A shorter request decodes eagerly unless it asks for graph decode, because the first capture in a process costs time a short reply does not win back.</summary>
    int LlmGraphDecodeDefaultMinTokens => 512;

    /// <summary>True when the text-LLM graph's device-position attention takes a sliding window and an attention logit soft-cap (the Gemma-2/3 shapes). CUDA does. A backend that returns false refuses those models from the graph: they decode eagerly, and an explicit graph request for them fails with the reason (<c>GenericTransformer.GraphDecodeRefusal</c>).</summary>
    bool LlmGraphDecodeWindowSoftcap => true;

    /// <summary>Allocates a persistent device position buffer {kvLen, qOffset, status} (0 = unsupported); free with <see cref="FreeDevicePos"/>.</summary>
    ulong AllocDevicePos() => 0;

    /// <summary>Refreshes the device position buffer and clears its status (outside any capture region) for the next step/launch.</summary>
    void WriteDevicePos(ulong handle, int kvLen, int qOffset) { }

    /// <summary>Synchronizes and throws when a device-position kernel reported an invalid position. Call at the step/replay readback point.</summary>
    void ThrowOnDevicePosError(ulong handle) { }

    /// <summary>Frees a buffer from <see cref="AllocDevicePos"/>.</summary>
    void FreeDevicePos(ulong handle) { }

    /// <summary>KV-cache append whose write slot comes from the device position buffer when non-zero; else = <see cref="KvCacheAppend"/>.</summary>
    void KvCacheAppendDev(Tensor buffer, Tensor newKv, int offset, ulong devicePos) => KvCacheAppend(buffer, newKv, offset);

    /// <summary>Pre-allocates a fixed KV-cache buffer on the device (uninitialized, no H2D/memset) so the first append writes in place.</summary>
    void ResidentAllocateKv(Tensor buffer) { }

    /// <summary>Self-attention whose kvLen/qOffset come from the device position buffer when non-zero; else = <see cref="FlashAttention"/>.</summary>
    void FlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, ulong devicePos,
        float softcap = 0f, int slidingWindow = 0)
        => FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink: null, slidingWindow, alibiSlopes: null);

    /// <summary>Copies a RESIDENT tensor's data to dst WITHOUT freeing/detaching its buffer, unlike a normal <c>DataPointer</c> read.</summary>
    unsafe void ReadResidentInto(Tensor src, float[] dst)
    {
        new Span<float>((void*)src.DataPointer, dst.Length).CopyTo(dst);
    }

    // The remaining per-token host work a captured graph can't tolerate: RoPE's cos/sin table build
    // (Math.Cos/Sin + H2D every step) and the embedding lookup (host gather from the token id). Both convert
    // to a one-time device-resident precompute (RoPE table) / preload (embed table) plus a fixed-grid kernel
    // indexed by device state — the SAME devicePos buffer above for RoPE's position, and a persistent 1-int
    // "current token id" buffer for embed (see AllocDeviceTokenId) that greedy decode chains entirely
    // on-device via ArgMaxInto (this step's sampled token becomes next step's embed input, no D2H between).

    /// <summary>Allocates a persistent 1-int device buffer for the "token id to embed next" (0 = unsupported).</summary>
    ulong AllocDeviceTokenId() => 0;

    /// <summary>Writes a token id into an <see cref="AllocDeviceTokenId"/> buffer outside capture; replay uses <see cref="ArgMaxInto"/>.</summary>
    void WriteDeviceTokenId(ulong handle, int tokenId) { }

    /// <summary>Frees a buffer from <see cref="AllocDeviceTokenId"/>.</summary>
    void FreeDeviceTokenId(ulong handle) { }

    /// <summary>Reads back the token id from <see cref="AllocDeviceTokenId"/> (the one D2H sync decode needs); throws if unsupported.</summary>
    int ReadDeviceTokenId(ulong handle) => throw new NotSupportedException("ReadDeviceTokenId requires GraphDecodeSupported.");

    /// <summary>Gathers one row (id in the persistent tokenId buffer) from a GPU-resident embedding table into output <c>[1,1,hidden]</c>.</summary>
    /// <remarks>The default throws, like every decode step that writes an output.</remarks>
    void EmbedGatherDecodeStep(Tensor output, Tensor embedTable, ulong tokenId) =>
        throw new NotSupportedException("EmbedGatherDecodeStep requires GraphDecodeSupported.");

    /// <summary>Argmax over the last dim, writing into a persistent buffer (e.g. <see cref="AllocDeviceTokenId"/>) to chain decode.</summary>
    void ArgMaxInto(ulong outputTokenId, Tensor input) =>
        throw new NotSupportedException("ArgMaxInto requires GraphDecodeSupported.");

    // Greedy graph decode previously skipped the sampler chain entirely (raw argmax on unpenalized logits),
    // silently ignoring a nonzero repetition penalty — repetition penalty is the ONLY sampler stage that can
    // change greedy's picked token (temperature is monotonic, top-k/top-p/min-p never remove the argmax
    // survivor), so it's also the only one graph decode needs to replicate. History (the model's own generated
    // tokens, not the prompt) lives in a fixed-capacity device buffer appended to once per replay.

    /// <summary>Allocates a persistent int[capacity] buffer for repetition-penalty history; free with <see cref="FreeDeviceHistory"/>.</summary>
    ulong AllocDeviceHistory(int capacity) => 0;

    /// <summary>Frees a buffer from <see cref="AllocDeviceHistory"/>.</summary>
    void FreeDeviceHistory(ulong handle) { }

    /// <summary>Allocates a persistent 1-int device counter (0 = unsupported); free with <see cref="FreeDeviceCounter"/>.</summary>
    ulong AllocDeviceCounter() => 0;

    /// <summary>Frees a buffer from <see cref="AllocDeviceCounter"/>.</summary>
    void FreeDeviceCounter(ulong handle) { }

    /// <summary>Writes a device counter's value (outside capture — e.g. resetting to 0 at session start).</summary>
    void WriteDeviceCounter(ulong handle, int value) { }

    /// <summary>Appends tokenId's value into history at the historyCount index, then increments the counter (fixed 1-thread launch).</summary>
    void AppendTokenHistoryStep(ulong history, ulong historyCount, ulong tokenId) =>
        throw new NotSupportedException("AppendTokenHistoryStep requires GraphDecodeSupported.");

    /// <summary>Applies HF-convention repetition penalty (divide positive/multiply negative logits) to every history token.</summary>
    void ApplyRepetitionPenaltyStep(Tensor logits, ulong history, ulong historyCount, float penalty) =>
        throw new NotSupportedException("ApplyRepetitionPenaltyStep requires GraphDecodeSupported.");

    /// <summary>A decode step's QKV epilogue: q, k and v cut from the projection output, q and k RMS-normalized as
    /// <see cref="QkvDecodeCall.Norm"/> says and rotated by the table row at the device position, q written head-major
    /// to its output and k, v scattered into slot <c>position</c> of their caches. One launch where a backend fuses it;
    /// the default composes it from <see cref="CopyStrided"/>, <see cref="RmsNorm"/>, <see cref="Rotary"/> and
    /// <see cref="KvCacheAppendDev"/>. Contract in <c>docs/BACKEND_OPS.md</c>.</summary>
    void QkvDecodeEpilogue(in QkvDecodeCall call) => ComposeQkvDecodeEpilogue(this, call);

    /// <summary>The unfused form of <see cref="QkvDecodeEpilogue"/>, the same ops in the same order on every backend. Static so a
    /// backend whose fused kernel does not serve a call can still run it.</summary>
    static void ComposeQkvDecodeEpilogue(IBackend b, in QkvDecodeCall call)
    {
        QkvDecodeContract.Validate(call);
        int hq = call.Heads, hkv = call.KvHeads, d = call.HeadDim;
        bool interleaved = call.Pairing == RotaryPairing.Interleaved;
        Tensor k = new(new TensorShape(1, hkv, 1, d), DType.F32);
        Tensor? vOwn = IsWhole(call.V, hkv, d) ? null : new(new TensorShape(1, hkv, 1, d), DType.F32);
        try
        {
            NormalizeInto(b, call.QOut, call.Q, hq, call.Norm, call.QNormWeight, call.Eps);
            NormalizeInto(b, k, call.K, hkv, call.Norm, call.KNormWeight, call.Eps);
            if (vOwn is not null) Cut(b, vOwn, call.V, call.KvWidth);
            b.Rotary(RotaryCall.DecodeStep(call.QOut, call.Cos, call.Sin, call.RotaryDim, interleaved, call.DevicePosition));
            b.Rotary(RotaryCall.DecodeStep(k, call.Cos, call.Sin, call.RotaryDim, interleaved, call.DevicePosition));
            b.KvCacheAppendDev(call.KCache, k, 0, call.DevicePosition);
            b.KvCacheAppendDev(call.VCache, vOwn ?? call.V.Tensor, 0, call.DevicePosition);
        }
        finally
        {
            k.Dispose();
            vOwn?.Dispose();
        }

        // The cache append reads the token count from the tensor's shape, so V is reused only in its [1, heads, 1, d] form.
        static bool IsWhole(in QkvSection section, int heads, int d) => section.Offset == 0 && section.Tensor.Shape.Equals(new TensorShape(1, heads, 1, d));

        // The section's elements, packed into dst.
        static void Cut(IBackend b, Tensor dst, in QkvSection section, long width)
            => b.CopyStrided(dst, section.Tensor, StridedCopySpec.Gather(StridedView.Of([section.Tensor.ElementCount]).Slice(0, section.Offset, width)));

        // The section lands in dst [1, heads, 1, d]: as projected, normalized per head, or normalized as one row and re-cut per head.
        static void NormalizeInto(IBackend b, Tensor dst, in QkvSection section, int heads, QkvNormKind norm, Tensor? weight, float eps)
        {
            long width = (long)heads * dst.Shape[3];
            if (norm == QkvNormKind.None)
            {
                Cut(b, dst, section, width);
                return;
            }
            Tensor raw = new(norm == QkvNormKind.PerHead ? dst.Shape : new TensorShape(1, 1, width), DType.F32);
            Tensor? row = norm == QkvNormKind.FullWidth ? new(new TensorShape(1, 1, width), DType.F32) : null;
            try
            {
                Cut(b, raw, section, width);
                b.RmsNorm(row ?? dst, raw, weight!, eps);
                if (row is not null) Cut(b, dst, row, width);
            }
            finally
            {
                raw.Dispose();
                row?.Dispose();
            }
        }
    }

    /// <summary>True when the graph decode step should fold each layer's final residual add into the next layer's input norm
    /// (<see cref="AddRmsNormEmitQ8"/> / <see cref="MoeCombineAddRmsNormEmitQ8"/>) because this backend's fused kernels are
    /// bit-identical to its separate ones. False keeps the separate launches.</summary>
    bool FoldsDecodeResidual => false;

    /// <summary>True when this backend can store a transformer KV cache as F16 and run attention and graph decode over it, which is what lets
    /// <c>vram.kvF16</c> default to on for text models. CUDA only today.</summary>
    bool SupportsF16KvCache => false;

    /// <summary>Gemma-4 PLE graph-decode gather: dequantizes one row of the quantized per-layer embedding table.</summary>
    /// <param name="output"><c>[1, 1, ple·layers]</c>.</param>
    /// <remarks>Scaled by sqrt(ple); only the CUDA backend implements this (graph decode is backend-gated), the default throws.</remarks>
    void PleGatherDecodeStep(Tensor output, Tensor quantTable, float scale, ulong deviceTokenId)
        => throw new NotSupportedException("PleGatherDecodeStep requires a graph-decode-capable backend.");

    /// <summary>True when <see cref="EmbedGatherQuantDecodeStep"/> can gather a quantized embed table of <paramref name="tableBytes"/> bytes, so graph decode keeps the QUANTIZED table on device instead of an F32 copy. Other backends (and tables too big for one device storage-buffer range) keep the F32 <see cref="EmbedGatherDecodeStep"/> path.</summary>
    bool CanGatherQuantEmbed(long tableBytes) => false;

    /// <summary>Graph-decode embed gather that dequantizes one row of a quantized (Q8_0, Q5_0, Q4_K or Q6_K) table on device, then multiplies by <paramref name="scale"/> (1.0 = no scale).</summary>
    /// <remarks>Only backends reporting <see cref="CanGatherQuantEmbed"/> true for a table implement this; the default throws.</remarks>
    void EmbedGatherQuantDecodeStep(Tensor output, Tensor quantTable, float scale, ulong deviceTokenId)
        => throw new NotSupportedException("EmbedGatherQuantDecodeStep requires a backend whose CanGatherQuantEmbed accepts the table.");

    /// <summary>True when graph decode can take its step embedding from a host-written row (see <see cref="AllocDecodeEmbedRow"/>).
    /// An UNTIED quantized embed table then needs no device copy at all: the host row is the eager lookup's row, written before each replay.</summary>
    bool StagesDecodeEmbedRows => false;

    /// <summary>Allocates a host-visible row of <paramref name="hidden"/> F32 values that graph decode copies its embedding from; returns its handle. Outside any capture.</summary>
    ulong AllocDecodeEmbedRow(int hidden)
        => throw new NotSupportedException("AllocDecodeEmbedRow requires a backend whose StagesDecodeEmbedRows is true.");

    /// <summary>Writes <paramref name="count"/> F32 values at <paramref name="row"/> into a row from <see cref="AllocDecodeEmbedRow"/>. Call between graph replays, after the previous replay has completed.</summary>
    void WriteDecodeEmbedRow(ulong handle, nint row, int count)
        => throw new NotSupportedException("WriteDecodeEmbedRow requires a backend whose StagesDecodeEmbedRows is true.");

    /// <summary>Graph capture only: records the copy of row <paramref name="handle"/> into <paramref name="output"/> ([1,1,hidden] F32) as the step's first op,
    /// so every replay embeds the row written before it.</summary>
    void EmbedStagedRowDecodeStep(Tensor output, ulong handle)
        => throw new NotSupportedException("EmbedStagedRowDecodeStep requires a backend whose StagesDecodeEmbedRows is true.");

    /// <summary>Frees a row from <see cref="AllocDecodeEmbedRow"/>.</summary>
    void FreeDecodeEmbedRow(ulong handle) { }

    /// <summary>True when <see cref="RelativeL1Distance"/> runs without pulling operands to the host; feature caching must gate on this.</summary>
    bool SupportsDeviceStepCacheGate => false;

    /// <summary>Relative-L1 distance <c>Σ|a−b| / Σ|b|</c>; zero when both tensors are zero, positive infinity when only the reference is zero, and NaN when an operand or an accumulated sum is non-finite.</summary>
    unsafe float RelativeL1Distance(Tensor a, Tensor b)
    {
        if (!a.Shape.Equals(b.Shape))
            throw new ArgumentException($"RelativeL1Distance shape mismatch: a={a.Shape}, b={b.Shape}.");
        if (a.DType != b.DType)
            throw new ArgumentException($"RelativeL1Distance dtype mismatch: a={a.DType}, b={b.DType}.");
        long n = a.ElementCount;
        double diff = 0.0, denom = 0.0;
        if (a.DType == DType.F32)
        {
            float* ap = (float*)a.DataPointer;
            float* bp = (float*)b.DataPointer;
            for (long i = 0; i < n; i++)
            {
                diff += Math.Abs(ap[i] - bp[i]);
                denom += Math.Abs(bp[i]);
            }
        }
        else if (a.DType == DType.F16)
        {
            Half* ap = (Half*)a.DataPointer;
            Half* bp = (Half*)b.DataPointer;
            for (long i = 0; i < n; i++)
            {
                float av = (float)ap[i];
                float bv = (float)bp[i];
                diff += Math.Abs(av - bv);
                denom += Math.Abs(bv);
            }
        }
        else
        {
            throw new NotSupportedException($"RelativeL1Distance supports F32/F16; got {a.DType}.");
        }
        if (!double.IsFinite(diff) || !double.IsFinite(denom))
            return float.NaN;
        if (denom > 0.0)
            return (float)(diff / denom);

        return diff > 0.0 ? float.PositiveInfinity : 0.0f;
    }
}
