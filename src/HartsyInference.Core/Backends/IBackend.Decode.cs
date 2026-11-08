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
    // read the position from a small device buffer {kvLen, qOffset}, refreshed (outside capture) each step. This
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

    /// <summary>Allocates a persistent 2-int device buffer {kvLen, qOffset} (0 = unsupported); free with <see cref="FreeDevicePos"/>.</summary>
    ulong AllocDevicePos() => 0;

    /// <summary>Refreshes the device position buffer (outside any capture region) for the next step/launch.</summary>
    void WriteDevicePos(ulong handle, int kvLen, int qOffset) { }

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

    /// <summary>RoPE on a single decode-step Q/K tensor <c>[1,numHeads,1,headDim]</c>, reading its row from the devicePos qOffset slot.</summary>
    /// <remarks>The default throws: a backend without the kernel must be refused, not silently leave Q/K unrotated.</remarks>
    void RopeApplyDecodeStep(Tensor x, Tensor cosTable, Tensor sinTable, int rotaryDim, bool interleaved, ulong devicePos) =>
        throw new NotSupportedException("RopeApplyDecodeStep requires GraphDecodeSupported.");

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

    /// <summary>Fused graph-decode QKV epilogue (t=1, no QK-norm): ropes q into qOut, ropes+appends k into kCache, copies v into vCache.</summary>
    void QkvRopeScatterDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qkv,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        int nq = hq * headDim, nkv = hkv * headDim;
        Tensor k = new(new TensorShape(1, hkv, 1, headDim), DType.F32);
        Tensor v = new(new TensorShape(1, hkv, 1, headDim), DType.F32);
        try
        {
            SliceLastDim(qOut, qkv, 0);
            SliceLastDim(k, qkv, nq);
            SliceLastDim(v, qkv, nq + nkv);
            RopeApplyDecodeStep(qOut, cosTable, sinTable, rotaryDim, interleaved, devicePos);
            RopeApplyDecodeStep(k, cosTable, sinTable, rotaryDim, interleaved, devicePos);
            KvCacheAppendDev(kCache, k, 0, devicePos);
            KvCacheAppendDev(vCache, v, 0, devicePos);
        }
        finally
        {
            k.Dispose();
            v.Dispose();
        }
    }

    /// <summary>Variant of <see cref="QkvRopeScatterDecodeStep"/> for layers with q/k/v in SEPARATE tensors (mixed-dtype QKV).</summary>
    void RopeScatterKvDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor q, Tensor k, Tensor v,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        RopeApplyDecodeStep(q, cosTable, sinTable, rotaryDim, interleaved, devicePos);
        RopeApplyDecodeStep(k, cosTable, sinTable, rotaryDim, interleaved, devicePos);
        KvCacheAppendDev(kCache, k, 0, devicePos);
        KvCacheAppendDev(vCache, v, 0, devicePos);
        CopyTo(qOut, q);
    }

    /// <summary>Variant of <see cref="QkvRopeScatterDecodeStep"/> for layers where q/k share a buffer but v is its own tensor (GGUF).</summary>
    void QkRopeScatterVDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qk, Tensor v,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        int nq = hq * headDim;
        Tensor k = new(new TensorShape(1, hkv, 1, headDim), DType.F32);
        try
        {
            SliceLastDim(qOut, qk, 0);
            SliceLastDim(k, qk, nq);
            RopeApplyDecodeStep(qOut, cosTable, sinTable, rotaryDim, interleaved, devicePos);
            RopeApplyDecodeStep(k, cosTable, sinTable, rotaryDim, interleaved, devicePos);
            KvCacheAppendDev(kCache, k, 0, devicePos);
            KvCacheAppendDev(vCache, v, 0, devicePos);
        }
        finally
        {
            k.Dispose();
        }
    }

    /// <summary>QK-norm variant of <see cref="QkvRopeScatterDecodeStep"/>: per-head RMS-norms q/k from the projection, ropes, scatters.</summary>
    void QkvNormRopeScatterDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qkv,
        Tensor qNorm, Tensor kNorm, float eps,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        int nq = hq * headDim, nkv = hkv * headDim;
        Tensor q = new(new TensorShape(1, hq, 1, headDim), DType.F32);
        Tensor k = new(new TensorShape(1, hkv, 1, headDim), DType.F32);
        Tensor v = new(new TensorShape(1, hkv, 1, headDim), DType.F32);
        Tensor qN = new(new TensorShape(1, hq, 1, headDim), DType.F32);
        Tensor kN = new(new TensorShape(1, hkv, 1, headDim), DType.F32);
        try
        {
            SliceLastDim(q, qkv, 0);
            SliceLastDim(k, qkv, nq);
            SliceLastDim(v, qkv, nq + nkv);
            RmsNorm(qN, q, qNorm, eps);
            RmsNorm(kN, k, kNorm, eps);
            RopeScatterKvDecodeStep(qOut, kCache, vCache, qN, kN, v, cosTable, sinTable,
                hq, hkv, headDim, rotaryDim, interleaved, devicePos);
        }
        finally
        {
            q.Dispose(); k.Dispose(); v.Dispose(); qN.Dispose(); kN.Dispose();
        }
    }

    /// <summary>Full-width QK-norm variant of <see cref="QkvNormRopeScatterDecodeStep"/> (OLMoE): ONE RMSNorm over all q heads with the
    /// <c>[hq·headDim]</c> weight <paramref name="qNorm"/> and one over all k heads, then RoPE and the KV scatter.</summary>
    /// <param name="qk">The projection output: <c>[q | k | v]</c> when <paramref name="v"/> is null, else <c>[q | k]</c>.</param>
    void QkNormFullRopeScatterDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qk, Tensor? v,
        Tensor qNorm, Tensor kNorm, float eps,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
        => ComposeQkNormFullRopeScatter(this, qOut, kCache, vCache, qk, v, qNorm, kNorm, eps, cosTable, sinTable,
            hq, hkv, headDim, rotaryDim, interleaved, devicePos);

    /// <summary>The unfused composition behind <see cref="QkNormFullRopeScatterDecodeStep"/>: slices, two full-width RMSNorms, the rope-scatter.
    /// Static so a backend that overrides the op can still fall back to it.</summary>
    static void ComposeQkNormFullRopeScatter(IBackend b, Tensor qOut, Tensor kCache, Tensor vCache, Tensor qk, Tensor? v,
        Tensor qNorm, Tensor kNorm, float eps,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        int nq = hq * headDim, nkv = hkv * headDim;
        Tensor q = new(new TensorShape(1, 1, nq), DType.F32);
        Tensor k = new(new TensorShape(1, 1, nkv), DType.F32);
        Tensor? vOwn = v is null ? new(new TensorShape(1, hkv, 1, headDim), DType.F32) : null;
        Tensor qN = new(new TensorShape(1, 1, nq), DType.F32);
        Tensor kN = new(new TensorShape(1, 1, nkv), DType.F32);
        try
        {
            b.SliceLastDim(q, qk, 0);
            b.SliceLastDim(k, qk, nq);
            if (vOwn is not null) b.SliceLastDim(vOwn, qk, nq + nkv);
            b.RmsNorm(qN, q, qNorm, eps);
            b.RmsNorm(kN, k, kNorm, eps);
            b.RopeScatterKvDecodeStep(qOut, kCache, vCache, qN, kN, v ?? vOwn!, cosTable, sinTable,
                hq, hkv, headDim, rotaryDim, interleaved, devicePos);
        }
        finally
        {
            q.Dispose(); k.Dispose(); vOwn?.Dispose(); qN.Dispose(); kN.Dispose();
        }
    }

    /// <summary>True when the graph decode step should fold each layer's final residual add into the next layer's input norm
    /// (<see cref="AddRmsNormEmitQ8"/> / <see cref="MoeCombineAddRmsNormEmitQ8"/>) because this backend's fused kernels are
    /// bit-identical to its separate ones. False keeps the separate launches.</summary>
    bool FoldsDecodeResidual => false;

    /// <summary>True when this backend can store a transformer KV cache as F16 and run attention and graph decode over it, which is what lets
    /// <c>vram.kvF16</c> default to on for text models. CUDA only today.</summary>
    bool SupportsF16KvCache => false;

    /// <summary>Partial-fusion <see cref="QkvNormRopeScatterDecodeStep"/> variant: q/k concatenated, v separate (mixed-dtype v).</summary>
    void QkNormRopeScatterVDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qk, Tensor v,
        Tensor qNorm, Tensor kNorm, float eps,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        int nq = hq * headDim;
        Tensor q = new(new TensorShape(1, hq, 1, headDim), DType.F32);
        Tensor k = new(new TensorShape(1, hkv, 1, headDim), DType.F32);
        Tensor qN = new(new TensorShape(1, hq, 1, headDim), DType.F32);
        Tensor kN = new(new TensorShape(1, hkv, 1, headDim), DType.F32);
        try
        {
            SliceLastDim(q, qk, 0);
            SliceLastDim(k, qk, nq);
            RmsNorm(qN, q, qNorm, eps);
            RmsNorm(kN, k, kNorm, eps);
            RopeScatterKvDecodeStep(qOut, kCache, vCache, qN, kN, v, cosTable, sinTable,
                hq, hkv, headDim, rotaryDim, interleaved, devicePos);
        }
        finally
        {
            q.Dispose(); k.Dispose(); qN.Dispose(); kN.Dispose();
        }
    }

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
