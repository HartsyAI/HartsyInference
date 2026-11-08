using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Attention: the AttentionCall contract and its planner, the legacy SDPA and Flash entries, video sparse attention,
// neighborhood and deformable attention, KV append, and the single-latent sparse-attention, indexer and
// hyper-connection primitives. Backends without a kernel inherit the host-reference or throwing defaults.
public partial interface IBackend
{
    /// <summary>Runs one attention call (docs/ATTENTION_CONTRACT.md). A backend with kernel rows overrides this with its
    /// planner; the default computes the contract on the host and never calls the legacy entry points, so it cannot recurse.</summary>
    void Attention(in AttentionCall call)
    {
        AttentionPlanLog.HostReference(GetType().Name);
        AttentionHostReference.Run(in call);
    }

    /// <summary>The kernel <see cref="Attention"/> would run for <paramref name="call"/> and why; launches nothing.</summary>
    AttentionPlan PlanAttention(in AttentionCall call) => new(null, null, "host reference: this backend declares no attention rows", null);

    /// <summary>Whether the backend's token-major INT8 attention v2 row would take an unmasked, equal-head, aligned call of this
    /// shape now (CUDA's M.4); an attention layer keeps such a call token-major instead of permuting it for SageAttention.</summary>
    bool Int8AttentionV2TakesTokenMajor(int batch, int heads, int queryTokens, int keys, int headDim, DType dtype) => false;

    /// <summary>Attention where one latent row is both key and value: per token and head, softmax over the scaled
    /// dot products of the indexed rows plus the head's fp32 sink, then the weighted sum of those rows.
    /// See <see cref="SparseLatentAttentionReference.Apply"/> for the exact index and sink rules.</summary>
    /// <param name="output">F32 <c>[tokens,heads,dim]</c>.</param>
    /// <param name="query">F32 <c>[tokens,heads,dim]</c>.</param>
    /// <param name="window">Sliding-window ring with exactly <paramref name="windowSlots"/> rows.</param>
    /// <param name="main">Compressed cache addressed after the ring; may be empty.</param>
    /// <param name="indices">I32 <c>[tokens,k]</c>; -1 is skipped.</param>
    /// <param name="sink">F32 <c>[heads]</c>, joins the softmax denominator only.</param>
    void SparseLatentAttention(Tensor output, Tensor query, in LatentSource window, in LatentSource main,
        Tensor indices, int windowSlots, Tensor sink, float scale) =>
        throw NotSupportedPrimitive(nameof(SparseLatentAttention));

    /// <summary>Indexer scores <c>scale * sum_h relu(q[t,h] . key[n]) * headWeights[t,h]</c>, masked to -infinity at
    /// <c>n >= compressLens[t]</c> and where <paramref name="candidates"/> is zero.</summary>
    /// <param name="scores">F32 <c>[tokens,keys.Rows]</c>.</param>
    /// <param name="query">F32 <c>[tokens,heads,dim]</c>, already rotated and quantize-dequantized.</param>
    /// <param name="headWeights">F32 <c>[tokens,heads]</c>.</param>
    /// <param name="compressLens">I32 <c>[tokens]</c> visible key count per query.</param>
    /// <param name="candidates">Optional U8 <c>[tokens,keys.Rows]</c> allow-mask.</param>
    void IndexerScores(Tensor scores, Tensor query, in LatentSource keys, Tensor headWeights, Tensor compressLens,
        Tensor? candidates, float scale) =>
        throw NotSupportedPrimitive(nameof(IndexerScores));

    /// <summary>Splits mixes <c>[tokens,(2+hc)*hc]</c> into <c>pre[tokens,hc]</c>, <c>post[tokens,hc]</c> and a Sinkhorn-normalized
    /// <c>comb[tokens,hc,hc]</c>; see <see cref="HcReference.SplitSinkhorn"/>.</summary>
    /// <param name="scale">F32 <c>[3]</c> for pre, post and comb.</param>
    /// <param name="bias">F32 <c>[(2+hc)*hc]</c>.</param>
    void HcSplitSinkhorn(Tensor pre, Tensor post, Tensor comb, Tensor mixes, Tensor scale, Tensor bias, int hc,
        int iters, float eps) =>
        throw NotSupportedPrimitive(nameof(HcSplitSinkhorn));

    /// <summary>Collapses the streams of <c>x[tokens,hc,dim]</c> into <c>output[tokens,dim]</c> weighted by <c>pre[tokens,hc]</c>.</summary>
    void HcPreMix(Tensor output, Tensor x, Tensor pre) =>
        throw NotSupportedPrimitive(nameof(HcPreMix));

    /// <summary>Expands <c>x[tokens,dim]</c> to <c>output[tokens,hc,dim]</c> and mixes in <c>residual[tokens,hc,dim]</c>
    /// through <c>post[tokens,hc]</c> and <c>comb[tokens,hc,hc]</c>; <paramref name="output"/> must not alias the residual.</summary>
    void HcPostMix(Tensor output, Tensor x, Tensor residual, Tensor post, Tensor comb) =>
        throw NotSupportedPrimitive(nameof(HcPostMix));

    /// <summary>Replaces F32 values with their quantize-dequantize round trip in <paramref name="encoding"/>,
    /// per group along the last dimension.</summary>
    void ActQuantDequantInPlace(Tensor x, LatentEncoding encoding) =>
        throw NotSupportedPrimitive(nameof(ActQuantDequantInPlace));

    /// <summary>Fills I32 <paramref name="indices"/> with the sliding-window ring slots per query, -1 for empty;
    /// see <see cref="WindowIndicesReference.Apply"/> for the shape and ordering.</summary>
    void BuildWindowIndices(Tensor indices, int windowSize, int seqLen, int startPos) =>
        throw NotSupportedPrimitive(nameof(BuildWindowIndices));

    /// <summary>In-place interleaved rotary on the <c>[dimOffset, dimOffset+rotaryDim)</c> slice of each vector of
    /// <c>x [B,L,D]</c> or <c>[B,L,H,D]</c>, with <c>cos</c>/<c>sin [B,L,rotaryDim/2]</c>; negate sin to invert.</summary>
    void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim, int dimOffset) =>
        throw NotSupportedPrimitive(nameof(ApplyRopeInterleaved));

    /// <summary>Start of the attended window along one axis, using NATTEN's rule: the window is always exactly
    /// <paramref name="kernel"/> wide and slides inward at the borders rather than being truncated or zero-padded,
    /// so every query attends the same number of keys. Callers must pre-clamp <paramref name="kernel"/> to
    /// <paramref name="length"/>.</summary>
    static int Na3dWindowStart(int index, int length, int kernel)
    {
        int start = index - kernel / 2;
        int last = length - kernel;
        if (start < 0) start = 0;
        if (start > last) start = last;
        return start;
    }

    /// <summary>3D neighborhood attention over <c>[batch, T, H, W, heads, headDim]</c> — each query attends a local
    /// <c>kernelT×kernelH×kernelW</c> window instead of the whole grid. Used by the LTX-2.5 diffusion video decoder.</summary>
    /// <param name="scale">Applied to the scores; pass 1.0 when the caller has already folded it into Q (the decoder
    /// folds it into the query RMS-norm weight, which commutes with the rotation).</param>
    /// <remarks>This managed implementation is the numerical reference, not a performance path — it is a plain
    /// six-deep loop and allocates one score buffer per call. <paramref name="output"/> must not alias any input:
    /// it is zeroed and accumulated into while neighbouring queries still read the originals.</remarks>
    unsafe void Na3d(Tensor output, Tensor q, Tensor k, Tensor v, int kernelT, int kernelH, int kernelW, float scale)
        => Na3dReference(output, q, k, v, kernelT, kernelH, kernelW, scale);

    /// <summary>True when <paramref name="output"/> is either the rank-6 q grid or its contiguous
    /// <c>[batch·T·H·W, heads·headDim]</c> flattening — the same bytes in the same order.</summary>
    /// <remarks>Accepting the flat form lets a caller whose next op is a row-wise GEMM allocate the output already
    /// flat, instead of reshaping the rank-6 result afterwards. <see cref="Tensor.Reshape"/> is not free: it reads
    /// <c>DataPointer</c>, which pulls a device-resident tensor back to the host and drops the device copy, so the
    /// following op re-uploads it — measured at 11.5 s of a 74.9 s LTX-2.5 decode.</remarks>
    static bool Na3dOutputShapeMatches(TensorShape output, TensorShape q)
        => output.Equals(q) || (output.Rank == 2 && output[0] == q[0] * q[1] * q[2] * q[3] && output[1] == q[4] * q[5]);

    /// <summary>The managed <see cref="Na3d"/> body, callable directly so a device backend can fall back to it and a
    /// test can diff against it without going through interface dispatch.</summary>
    static unsafe void Na3dReference(Tensor output, Tensor q, Tensor k, Tensor v, int kernelT, int kernelH, int kernelW, float scale)
    {
        if (output.DType != DType.F32 || q.DType != DType.F32 || k.DType != DType.F32 || v.DType != DType.F32)
            throw new NotSupportedException($"Na3d default fallback only supports F32 — got output={output.DType}, q={q.DType}, k={k.DType}, v={v.DType}.");
        if (q.Shape.Rank != 6)
            throw new ArgumentException($"Na3d expects [batch, T, H, W, heads, headDim]; got {q.Shape}.", nameof(q));
        if (!q.Shape.Equals(k.Shape) || !q.Shape.Equals(v.Shape) || !Na3dOutputShapeMatches(output.Shape, q.Shape))
            throw new ArgumentException($"Na3d requires identical shapes; got q={q.Shape}, k={k.Shape}, v={v.Shape}, output={output.Shape}.");
        if (kernelT <= 0 || kernelH <= 0 || kernelW <= 0)
            throw new ArgumentException($"Na3d kernel must be positive; got ({kernelT}, {kernelH}, {kernelW}).");
        // The output is zeroed and accumulated into while other queries still read the inputs, so an aliased buffer
        // would corrupt keys/values mid-pass. Inputs may alias each other freely.
        if (output.HasOverlappingHostStorageWithoutSync(q) || output.HasOverlappingHostStorageWithoutSync(k)
            || output.HasOverlappingHostStorageWithoutSync(v))
        {
            throw new ArgumentException("Na3d output must not alias q, k or v.", nameof(output));
        }

        int batch = (int)q.Shape[0], dimT = (int)q.Shape[1], dimH = (int)q.Shape[2], dimW = (int)q.Shape[3];
        int heads = (int)q.Shape[4], headDim = (int)q.Shape[5];
        // An axis shorter than its kernel collapses to the whole axis, matching the reference.
        int kt = Math.Min(kernelT, dimT), kh = Math.Min(kernelH, dimH), kw = Math.Min(kernelW, dimW);

        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer;
        float* vp = (float*)v.DataPointer, op = (float*)output.DataPointer;
        long headStride = headDim, wStride = (long)heads * headDim;
        long hStride = dimW * wStride, tStride = (long)dimH * hStride, bStride = (long)dimT * tStride;
        float[] scores = new float[kt * kh * kw];

        for (int b = 0; b < batch; b++)
        for (int it = 0; it < dimT; it++)
        {
            int t0 = Na3dWindowStart(it, dimT, kt);
            for (int ih = 0; ih < dimH; ih++)
            {
                int h0 = Na3dWindowStart(ih, dimH, kh);
                for (int iw = 0; iw < dimW; iw++)
                {
                    int w0 = Na3dWindowStart(iw, dimW, kw);
                    long qBase = b * bStride + it * tStride + ih * hStride + iw * wStride;
                    for (int head = 0; head < heads; head++)
                    {
                        long qOff = qBase + head * headStride;
                        float max = float.NegativeInfinity;
                        int n = 0;
                        for (int wt = 0; wt < kt; wt++)
                        for (int wh = 0; wh < kh; wh++)
                        for (int ww = 0; ww < kw; ww++)
                        {
                            long kOff = b * bStride + (t0 + wt) * tStride + (h0 + wh) * hStride
                                + (w0 + ww) * wStride + head * headStride;
                            float dot = 0f;
                            for (int d = 0; d < headDim; d++) dot += qp[qOff + d] * kp[kOff + d];
                            dot *= scale;
                            scores[n++] = dot;
                            if (dot > max) max = dot;
                        }

                        float sum = 0f;
                        for (int i = 0; i < n; i++)
                        {
                            float e = MathF.Exp(scores[i] - max);
                            scores[i] = e;
                            sum += e;
                        }
                        float inv = sum > 0f ? 1f / sum : 0f;

                        for (int d = 0; d < headDim; d++) op[qOff + d] = 0f;
                        n = 0;
                        for (int wt = 0; wt < kt; wt++)
                        for (int wh = 0; wh < kh; wh++)
                        for (int ww = 0; ww < kw; ww++)
                        {
                            float weight = scores[n++] * inv;
                            long vOff = b * bStride + (t0 + wt) * tStride + (h0 + wh) * hStride
                                + (w0 + ww) * wStride + head * headStride;
                            for (int d = 0; d < headDim; d++) op[qOff + d] += weight * vp[vOff + d];
                        }
                    }
                }
            }
        }
    }

    /// <summary>Whether this backend can execute the versioned MiniMax-H3 video sparse-attention contract without
    /// falling back to dense attention. Planning must check this before model construction.</summary>
    bool SupportsVideoSparseAttention => false;

    /// <summary>Creates persistent generation-scoped VSA state. The default refuses so unsupported backends cannot
    /// silently reinterpret a sparse checkpoint as dense attention.</summary>
    IVideoSparseAttentionSession CreateVideoSparseAttentionSession(VideoSparseAttentionPlan plan)
        => throw new NotSupportedException($"{GetType().Name} does not support video sparse attention.");

    /// <summary>Scaled dot-product attention: output = softmax(Q @ K^T / sqrt(d)) @ V</summary>
    /// <param name="allowF16">When true, a backend MAY run the attention in F16 for speed (halves score-matrix
    /// traffic + F16 tensor cores). Callers pass true ONLY when pre-softmax scores are bounded — e.g. Q/K are
    /// RMS/L2-normalized (Wan) — since unbounded scores (some fp8 archs) overflow F16 and produce black output.</param>
    void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false);

    /// <summary>Whether <see cref="ScaledDotProductAttentionTokenMajor"/> is served; false means the caller must
    /// build head-major <c>[B,H,S,D]</c> tensors and use <see cref="ScaledDotProductAttention"/> instead.</summary>
    bool SupportsTokenMajorAttention => false;

    /// <summary>SDPA whose Q/K/V/output are TOKEN-MAJOR <c>[S, heads*headDim]</c> (or that one sequence as
    /// <c>[1, S, heads*headDim]</c> or <c>[1, S, heads, headDim]</c>) — the layout a Linear already emits and <c>to_out</c>
    /// already wants, so the caller pays no permute on either side. Semantics are otherwise identical to
    /// <see cref="ScaledDotProductAttention"/>.</summary>
    void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int headDim, float scale, bool allowF16 = false)
        => throw new NotSupportedException($"{GetType().Name} does not serve token-major attention; check SupportsTokenMajorAttention first.");

    /// <summary>Whether the grouped-query overload of <see cref="ScaledDotProductAttentionTokenMajor(Tensor, Tensor, Tensor, Tensor, Tensor?, int, int, int, float, bool)"/>
    /// is served, reading K/V with fewer heads than Q without repeating them.</summary>
    bool SupportsTokenMajorGqaAttention => false;

    /// <summary>Token-major SDPA with <paramref name="kvHeads"/> K/V heads shared across <paramref name="heads"/> query
    /// heads: Q/output are <c>[S, heads*headDim]</c>, K/V are <c>[Skv, kvHeads*headDim]</c>, and query head h reads
    /// kv head <c>h / (heads/kvHeads)</c>.</summary>
    void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int kvHeads, int headDim, float scale, bool allowF16 = false)
        => throw new NotSupportedException($"{GetType().Name} does not serve grouped-query token-major attention; check SupportsTokenMajorGqaAttention first.");

    /// <summary>True when the backend has the GPU-resident decode toolkit, so autoregressive decoders (e.g. Zonos) can run fully on-device.</summary>
    bool FlashDecodeSupported => false;

    /// <summary>Fused online-softmax attention that never materializes the score matrix; grouped-query aware (head h reads kv h/kvGroup).
    /// The default builds an <see cref="AttentionCall"/> and runs <see cref="Attention"/>, so a backend without an override
    /// serves this entry through its kernel rows or the host reference.</summary>
    /// <param name="causal">When true, query row r (absolute position qOffset+r) attends keys 0..qOffset+r; otherwise all Lk keys.</param>
    /// <param name="softcap">&gt; 0 soft-caps each logit via <c>softcap·tanh(score/softcap)</c> before softmax (Gemma-2); 0 disables it.</param>
    /// <param name="sink">Non-null <c>[Hq]</c> F32 tensor (GPT-OSS sinks): each head's sink logit joins the softmax denominator only.</param>
    /// <param name="slidingWindow">&gt; 0 restricts each query to the most recent N keys (Gemma-2/3, Cohere2, GPT-OSS); else full causal.</param>
    /// <param name="alibiSlopes">Non-null <c>[Hq]</c> F32 tensor: adds ALiBi penalty <c>slope_h·(k_pos−q_pos)</c> (MPT/BLOOM/Falcon/Jais).</param>
    unsafe void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap = 0f, Tensor? sink = null, int slidingWindow = 0, Tensor? alibiSlopes = null)
    {
        AttentionCall.ValidateFlashGroup(query, key, kvGroup);
        Attention(AttentionCall.FromFlashAttention(output, query, key, value, kvLen, causal, qOffset, scale, softcap, sink, slidingWindow, alibiSlopes));
    }

    /// <summary>In-place KV-cache append: writes newKv <c>[1,H,tNew,D]</c> into the fixed buffer, avoiding the Concat-grown cache's O(n²).</summary>
    unsafe void KvCacheAppend(Tensor buffer, Tensor newKv, int offset)
    {
        if (buffer.DType != DType.F32 || newKv.DType != DType.F32)
            throw new NotSupportedException("KvCacheAppend default fallback only supports F32.");
        int h = (int)buffer.Shape[1];
        int maxSeq = (int)buffer.Shape[2];
        int d = (int)buffer.Shape[3];
        int tNew = (int)newKv.Shape[2];
        float* bp = (float*)buffer.DataPointer;
        float* np = (float*)newKv.DataPointer;
        for (int hi = 0; hi < h; hi++)
        {
            long srcBase = (long)hi * tNew * d;
            long dstBase = ((long)hi * maxSeq + offset) * d;
            Buffer.MemoryCopy(np + srcBase, bp + dstBase, (long)tNew * d * 4, (long)tNew * d * 4);
        }
    }

    /// <summary>Multi-scale deformable attention (Deformable DETR): softmaxes logits, samples value maps, accumulates weighted taps.</summary>
    /// <param name="refPoints"><c>[q·refQueryStride + l·refLevelStride + coord]</c> — GDINO uses per-level refs, RT-DETR shares one.</param>
    unsafe void DeformableAttention(Tensor output, Tensor value, Tensor sampOff, Tensor attnLogits,
        Tensor refPoints, ReadOnlySpan<int> spatialShapes, ReadOnlySpan<int> levelStart,
        int heads, int levels, int points, int coords, int refQueryStride, int refLevelStride)
    {
        if (output.DType != DType.F32 || value.DType != DType.F32 || sampOff.DType != DType.F32
            || attnLogits.DType != DType.F32 || refPoints.DType != DType.F32)
            throw new NotSupportedException("DeformableAttention default fallback only supports F32.");
        if (coords != 2 && coords != 4)
            throw new ArgumentException($"DeformableAttention coords must be 2 or 4, got {coords}.");

        int nq = (int)output.Shape[1];
        int d = (int)output.Shape[output.Shape.Rank - 1];
        int hd = d / heads;
        int lp = levels * points;

        float* offP = (float*)sampOff.DataPointer;
        float* atP = (float*)attnLogits.DataPointer;
        float* valP = (float*)value.DataPointer;
        float* refP = (float*)refPoints.DataPointer;
        float* outP = (float*)output.DataPointer;

        for (int q = 0; q < nq; q++)
        {
            long refBase = (long)q * refQueryStride;
            for (int h = 0; h < heads; h++)
            {
                long outOff = ((long)q * heads + h) * hd;
                for (int c = 0; c < hd; c++) outP[outOff + c] = 0f;

                // softmax denominator over lp for this (query, head)
                long aBase = ((long)q * heads + h) * lp;
                float amax = float.NegativeInfinity;
                for (int i = 0; i < lp; i++) { float v = atP[aBase + i]; if (v > amax) amax = v; }
                float asum = 0f;
                for (int i = 0; i < lp; i++) asum += MathF.Exp(atP[aBase + i] - amax);
                float invSum = 1f / asum;

                for (int l = 0; l < levels; l++)
                {
                    int hL = spatialShapes[l * 2 + 0], wL = spatialShapes[l * 2 + 1], start = levelStart[l];
                    long refOff = refBase + (long)l * refLevelStride;
                    float refX = refP[refOff + 0], refY = refP[refOff + 1];
                    float refW = coords == 4 ? refP[refOff + 2] : 0f;
                    float refH = coords == 4 ? refP[refOff + 3] : 0f;
                    for (int p = 0; p < points; p++)
                    {
                        long offIdx = ((((long)q * heads + h) * levels + l) * points + p) * 2;
                        float ox = offP[offIdx + 0], oy = offP[offIdx + 1];
                        float locX, locY;
                        if (coords == 2)
                        {
                            locX = refX + ox / wL;
                            locY = refY + oy / hL;
                        }
                        else
                        {
                            locX = refX + ox / points * refW * 0.5f;
                            locY = refY + oy / points * refH * 0.5f;
                        }
                        float weight = MathF.Exp(atP[aBase + l * points + p] - amax) * invSum;
                        float gx = 2f * locX - 1f, gy = 2f * locY - 1f;
                        float ix = ((gx + 1f) * wL - 1f) * 0.5f;
                        float iy = ((gy + 1f) * hL - 1f) * 0.5f;
                        int x0 = (int)MathF.Floor(ix), y0 = (int)MathF.Floor(iy);
                        int x1 = x0 + 1, y1 = y0 + 1;
                        float wx1 = ix - x0, wx0 = 1f - wx1, wy1 = iy - y0, wy0 = 1f - wy1;
                        bool x0ok = x0 >= 0 && x0 < wL, x1ok = x1 >= 0 && x1 < wL;
                        bool y0ok = y0 >= 0 && y0 < hL, y1ok = y1 >= 0 && y1 < hL;
                        long chanOff = (long)h * hd;
                        for (int c = 0; c < hd; c++)
                        {
                            float acc = 0f;
                            if (y0ok && x0ok) acc += wy0 * wx0 * valP[((long)start + (long)y0 * wL + x0) * d + chanOff + c];
                            if (y0ok && x1ok) acc += wy0 * wx1 * valP[((long)start + (long)y0 * wL + x1) * d + chanOff + c];
                            if (y1ok && x0ok) acc += wy1 * wx0 * valP[((long)start + (long)y1 * wL + x0) * d + chanOff + c];
                            if (y1ok && x1ok) acc += wy1 * wx1 * valP[((long)start + (long)y1 * wL + x1) * d + chanOff + c];
                            outP[outOff + c] += weight * acc;
                        }
                    }
                }
            }
        }
    }
}
