using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Backend interface all model code programs against (CPU SIMD, CUDA PTX/cuBLAS); operations are eager, returning when complete.</summary>
public interface IBackend : IDisposable
{
    /// <summary>The device this backend targets.</summary>
    DeviceKind Device { get; }

    /// <summary>Count of lazy device-to-host syncs since <see cref="ResetD2hSyncCount"/>; ~0 GPU-resident, 0 with no device sync.</summary>
    long GetD2hSyncCount() => 0;

    /// <summary>Resets the D2H sync counter. No-op on backends without a device sync.</summary>
    void ResetD2hSyncCount() { }

    /// <summary>Device memory (free, total) in bytes; (0,0) when not applicable (CPU backend).</summary>
    (long FreeBytes, long TotalBytes) GetVramInfo() => (0, 0);

    /// <summary>When true, F32 GEMMs use full 32-bit compute instead of TF32 (needed by e.g. Zonos); no-op without TF32.</summary>
    bool HighPrecisionGemm { get => false; set { } }

    /// <summary>When true, <see cref="Conv2D"/> pads the width axis with wrapped edge pixels instead of zeros, so
    /// the model sees a horizontally-continuous canvas — the standard "seamless tiling" trick for textures/
    /// patterns. Independent of <see cref="SeamlessTilingY"/> (SwarmUI core's "X-Only"/"Y-Only"/"true" modes).
    /// No-op on backends without a padding intercept (this is a request-scoped opt-in, not a capability).</summary>
    bool SeamlessTilingX { get => false; set { } }

    /// <summary>Same as <see cref="SeamlessTilingX"/> for the height axis.</summary>
    bool SeamlessTilingY { get => false; set { } }

    /// <summary>Capabilities of this backend.</summary>
    BackendCapabilities Capabilities { get; }

    // ── Linear Algebra ──────────────────────────────────────────────────

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
        if (weight.DType.IsQuantized)
            throw new NotSupportedException($"LinearWeightRows cannot row-slice block-quantized weights (got {weight.DType}).");
        using Tensor rows = new Tensor(new TensorShape(weightRowCount, weight.Shape[1]), weight.DType)
        {
            Fp8ScaleFactor = weight.Fp8ScaleFactor,
        };
        SliceRowsGeneric(rows, weight, weightRowOffset);
        if (bias is null)
        {
            Linear(output, input, rows, null);
            return;
        }
        // Bias is 1-D, one value per output channel, so it slices by ELEMENT — SliceRowsGeneric's row stride
        // (its last dim is the whole vector) would offset by rowOffset*count instead.
        using Tensor biasRows = new Tensor(new TensorShape(weightRowCount), bias.DType);
        long biasOffsetBytes = bias.DType.ComputeByteCount(weightRowOffset);
        long biasBytes = bias.DType.ComputeByteCount(weightRowCount);
        Buffer.MemoryCopy((byte*)bias.DataPointer + biasOffsetBytes, biasRows.DataPointer, biasBytes, biasBytes);
        Linear(output, input, rows, biasRows);
    }

    // ── Convolution ─────────────────────────────────────────────────────

    /// <summary>2D convolution: output = conv2d(input, weight, bias, stride, padding)</summary>
    void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW);

    // ── Normalization ───────────────────────────────────────────────────

    /// <summary>Group normalization.</summary>
    void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps);

    /// <summary>Layer normalization.</summary>
    void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps);

    /// <summary>RMS normalization.</summary>
    void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps);

    /// <summary>AdaIN 1D: per-(batch,channel) normalize <c>[B,C,T]</c> over T, scale by <c>(1+gamma)</c>, shift by beta.</summary>
    /// <param name="gamma"><c>[B, C]</c> (or <c>[C]</c>, broadcast across batch), typically a Linear projection of a style/speaker embedding.</param>
    /// <param name="beta"><c>[B, C]</c> (or <c>[C]</c>, broadcast across batch), typically a Linear projection of a style/speaker embedding.</param>
    void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps);

    // ── DiT transformer glue ────────────────────────────────────────────
    // These keep the per-block modulation math GPU-resident in diffusion transformers
    // (Ideogram 4, etc.). Default implementations are scalar-F32 CPU loops that also serve
    // as the numerical reference; CudaBackend overrides them with PTX kernels.

    /// <summary>Broadcast affine over the last dim: <c>out = in*scale + (shift ?? 0)</c>; null shift = Ideogram 4's scale-only adaLN.</summary>
    unsafe void AffineBroadcastLastDim(Tensor output, Tensor input, Tensor scale, Tensor? shift)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32 || scale.DType != DType.F32 || (shift is not null && shift.DType != DType.F32))
            throw new NotSupportedException("AffineBroadcastLastDim default fallback only supports F32.");
        int rank = input.Shape.Rank;
        int dim = (int)input.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)input.Shape[rank - 2] : 1;
        long total = input.ElementCount;
        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        float* pScale = (float*)scale.DataPointer;
        float* pShift = shift is null ? null : (float*)shift.DataPointer;
        for (long i = 0; i < total; i++)
        {
            int d = (int)(i % dim);
            long row = i / dim;
            int b = (int)(row / seqLen);
            long pIdx = (long)b * dim + d;
            float v = pIn[i] * pScale[pIdx];
            if (pShift != null) v += pShift[pIdx];
            pOut[i] = v;
        }
    }

    /// <summary>Gated residual over the last dim: <c>out = residual + gate*value</c>, gate <c>[B,D]</c> broadcast over the seq axis.</summary>
    unsafe void GatedResidualLastDim(Tensor output, Tensor residual, Tensor value, Tensor gate)
    {
        if (output.DType != DType.F32 || residual.DType != DType.F32 || value.DType != DType.F32 || gate.DType != DType.F32)
            throw new NotSupportedException("GatedResidualLastDim default fallback only supports F32.");
        int rank = value.Shape.Rank;
        int dim = (int)value.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)value.Shape[rank - 2] : 1;
        long total = value.ElementCount;
        float* pRes = (float*)residual.DataPointer;
        float* pVal = (float*)value.DataPointer;
        float* pGate = (float*)gate.DataPointer;
        float* pOut = (float*)output.DataPointer;
        for (long i = 0; i < total; i++)
        {
            int d = (int)(i % dim);
            long row = i / dim;
            int b = (int)(row / seqLen);
            long pIdx = (long)b * dim + d;
            pOut[i] = pRes[i] + pGate[pIdx] * pVal[i];
        }
    }

    /// <summary>Per-token adaLN with the modulation gather fused in: <c>out[r,d] = in[r,d]*(1 + scaleTable[rowIndex[r],d]) + shiftTable[rowIndex[r],d]</c>.</summary>
    /// <param name="scaleTable">Small <c>[modRows, D]</c> modulation table — the <c>1 +</c> is applied here, so pass the RAW scale (no AddScalar).</param>
    /// <param name="shiftTable">Small <c>[modRows, D]</c> table, or null for scale-only.</param>
    /// <param name="rowIndex">I32 <c>[rows]</c> — one table row per token; replaces materializing the gathered <c>[rows, D]</c> scale/shift.</param>
    unsafe void AffineBroadcastRowIndexed(Tensor output, Tensor input, Tensor scaleTable, Tensor? shiftTable, Tensor rowIndex)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32 || scaleTable.DType != DType.F32 || (shiftTable is not null && shiftTable.DType != DType.F32))
            throw new NotSupportedException($"AffineBroadcastRowIndexed default fallback only supports F32 — got output={output.DType}, input={input.DType}, scaleTable={scaleTable.DType}, shiftTable={shiftTable?.DType.Name ?? "null"}.");
        if (rowIndex.DType != DType.I32)
            throw new NotSupportedException($"AffineBroadcastRowIndexed requires I32 rowIndex, got {rowIndex.DType}.");
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long total = input.ElementCount;
        if (rowIndex.ElementCount < total / dim)
            throw new ArgumentException($"AffineBroadcastRowIndexed rowIndex has {rowIndex.ElementCount} entries, need {total / dim}.", nameof(rowIndex));
        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        float* pScale = (float*)scaleTable.DataPointer;
        float* pShift = shiftTable is null ? null : (float*)shiftTable.DataPointer;
        int* pIdx = (int*)rowIndex.DataPointer;
        for (long i = 0; i < total; i++)
        {
            int d = (int)(i % dim);
            long tIdx = (long)pIdx[i / dim] * dim + d;
            float v = pIn[i] * (1f + pScale[tIdx]);
            if (pShift != null) v += pShift[tIdx];
            pOut[i] = v;
        }
    }

    /// <summary>Per-token gated residual with the gate gather fused in: <c>out[r,d] = residual[r,d] + gateTable[rowIndex[r],d]*value[r,d]</c>.</summary>
    /// <param name="gateTable">Small <c>[modRows, D]</c> gate table, indexed per token instead of expanded to <c>[rows, D]</c>.</param>
    unsafe void GatedResidualRowIndexed(Tensor output, Tensor residual, Tensor value, Tensor gateTable, Tensor rowIndex)
    {
        if (output.DType != DType.F32 || residual.DType != DType.F32 || value.DType != DType.F32 || gateTable.DType != DType.F32)
            throw new NotSupportedException($"GatedResidualRowIndexed default fallback only supports F32 — got output={output.DType}, residual={residual.DType}, value={value.DType}, gateTable={gateTable.DType}.");
        if (rowIndex.DType != DType.I32)
            throw new NotSupportedException($"GatedResidualRowIndexed requires I32 rowIndex, got {rowIndex.DType}.");
        int dim = (int)value.Shape[value.Shape.Rank - 1];
        long total = value.ElementCount;
        if (rowIndex.ElementCount < total / dim)
            throw new ArgumentException($"GatedResidualRowIndexed rowIndex has {rowIndex.ElementCount} entries, need {total / dim}.", nameof(rowIndex));
        float* pRes = (float*)residual.DataPointer;
        float* pVal = (float*)value.DataPointer;
        float* pGate = (float*)gateTable.DataPointer;
        float* pOut = (float*)output.DataPointer;
        int* pIdx = (int*)rowIndex.DataPointer;
        for (long i = 0; i < total; i++)
        {
            int d = (int)(i % dim);
            long tIdx = (long)pIdx[i / dim] * dim + d;
            pOut[i] = pRes[i] + pGate[tIdx] * pVal[i];
        }
    }

    /// <summary>Fused adaLN modulation for DiT NormModulate: <c>out = (1+scale)·LayerNormNoAffine(in) + shift</c>.</summary>
    unsafe void LayerNormModulate(Tensor output, Tensor input, Tensor scale, Tensor shift, float eps)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32 || scale.DType != DType.F32 || shift.DType != DType.F32)
            throw new NotSupportedException("LayerNormModulate default fallback only supports F32.");
        int rank = input.Shape.Rank;
        int dim = (int)input.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)input.Shape[rank - 2] : 1;
        long rows = input.ElementCount / dim;
        float* pIn = (float*)input.DataPointer, pOut = (float*)output.DataPointer;
        float* pSc = (float*)scale.DataPointer, pSh = (float*)shift.DataPointer;
        for (long row = 0; row < rows; row++)
        {
            float* inRow = pIn + row * dim; float* outRow = pOut + row * dim;
            long b = (row / seqLen);
            float* sc = pSc + b * dim; float* sh = pSh + b * dim;
            double mean = 0; for (int i = 0; i < dim; i++) mean += inRow[i]; mean /= dim;
            double variance = 0; for (int i = 0; i < dim; i++) { double d = inRow[i] - mean; variance += d * d; } variance /= dim;
            float invStd = (float)(1.0 / Math.Sqrt(variance + eps));
            for (int i = 0; i < dim; i++) outRow[i] = ((float)(inRow[i] - mean)) * invStd * (1f + sc[i]) + sh[i];
        }
    }

    /// <summary>Fused QKV split + per-head QK-RMSNorm: qkv <c>[.,3w]</c> → q/k/v each <c>[.,w]</c>; RMS-norms q/k (w=heads·headDim).</summary>
    unsafe void QkvSplitNorm(Tensor q, Tensor k, Tensor v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
    {
        if (q.DType != DType.F32 || qkv.DType != DType.F32) throw new NotSupportedException("QkvSplitNorm default fallback only supports F32.");
        int headDim = (int)qWeight.Shape[qWeight.Shape.Rank - 1];
        int w = (int)qkv.Shape[qkv.Shape.Rank - 1] / 3;   // fused qkv last dim = 3·w
        int heads = w / headDim;
        long tok = qkv.ElementCount / (3L * w);
        float* pq = (float*)q.DataPointer, pk = (float*)k.DataPointer, pv = (float*)v.DataPointer;
        float* pqkv = (float*)qkv.DataPointer, pqw = (float*)qWeight.DataPointer, pkw = (float*)kWeight.DataPointer;
        for (long t = 0; t < tok; t++)
        {
            float* baseP = pqkv + t * 3L * w;
            for (int h = 0; h < heads; h++)
            {
                float* qs = baseP + h * headDim; float* ks = baseP + w + h * headDim; float* vs = baseP + 2 * w + h * headDim;
                long outOff = t * w + (long)h * headDim;
                double qss = 0, kss = 0; for (int d = 0; d < headDim; d++) { qss += (double)qs[d] * qs[d]; kss += (double)ks[d] * ks[d]; }
                float qInv = (float)(1.0 / Math.Sqrt(qss / headDim + eps)), kInv = (float)(1.0 / Math.Sqrt(kss / headDim + eps));
                for (int d = 0; d < headDim; d++) { pq[outOff + d] = qs[d] * qInv * pqw[d]; pk[outOff + d] = ks[d] * kInv * pkw[d]; pv[outOff + d] = vs[d]; }
            }
        }
    }

    /// <summary>Head-major <see cref="QkvSplitNorm"/>: same split + per-head QK-RMSNorm, but q/k/v come out <c>[B, heads, seq, headDim]</c>.</summary>
    /// <remarks>Lets the attention caller skip the q/k/v <see cref="Permute0213"/>; RMSNorm is over headDim, the last dim in either layout.</remarks>
    /// <remarks>Any of <paramref name="q"/>/<paramref name="k"/>/<paramref name="v"/> may be null to emit only a
    /// SUBSET. A 3-wide packed source keeps the canonical q=0,k=1,v=2 segments even then; a narrowed <c>[k|v]</c> or
    /// <c>[q]</c> source carries only what it names, numbered in q,k,v order. That is what lets
    /// <c>MiniMaxH3Transformer</c>'s chunked attention project k+v in one pass and q in the next, so a full-sequence
    /// q never stays resident across the pass boundary.</remarks>
    unsafe void QkvSplitNormHeadMajor(Tensor? q, Tensor? k, Tensor? v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
    {
        Tensor shapeRef = q ?? k ?? v ?? throw new ArgumentException("QkvSplitNormHeadMajor needs at least one of q/k/v.", nameof(q));
        if (shapeRef.DType != DType.F32 || qkv.DType != DType.F32) throw new NotSupportedException("QkvSplitNormHeadMajor default fallback only supports F32.");
        if (shapeRef.Shape.Rank != 4) throw new ArgumentException($"QkvSplitNormHeadMajor needs q/k/v shaped [B,heads,seq,headDim], got {shapeRef.Shape}.", nameof(q));
        foreach (Tensor? other in new[] { q, k, v })
            if (other is not null && !other.Shape.Equals(shapeRef.Shape))
                throw new ArgumentException($"QkvSplitNormHeadMajor needs the emitted q/k/v identically shaped, got q {q?.Shape} k {k?.Shape} v {v?.Shape}.", nameof(k));
        int headDim = (int)qWeight.Shape[qWeight.Shape.Rank - 1];
        int heads = (int)shapeRef.Shape[1];
        int w = heads * headDim;
        int packStride = (int)qkv.Shape[qkv.Shape.Rank - 1] / w;
        int qSlot, kSlot, vSlot;
        if (packStride == 3)
        {
            qSlot = q is null ? -1 : 0; kSlot = k is null ? -1 : 1; vSlot = v is null ? -1 : 2;
        }
        else
        {
            int next = 0;
            qSlot = q is null ? -1 : next++; kSlot = k is null ? -1 : next++; vSlot = v is null ? -1 : next++;
            if (next != packStride)
                throw new ArgumentException($"QkvSplitNormHeadMajor: a {packStride}-wide packed source must carry exactly the requested outputs.", nameof(qkv));
        }
        int seq = (int)shapeRef.Shape[2];
        long tok = qkv.ElementCount / ((long)packStride * w);
        float* pq = q is null ? null : (float*)q.DataPointer;
        float* pk = k is null ? null : (float*)k.DataPointer;
        float* pv = v is null ? null : (float*)v.DataPointer;
        float* pqkv = (float*)qkv.DataPointer, pqw = (float*)qWeight.DataPointer, pkw = (float*)kWeight.DataPointer;
        for (long t = 0; t < tok; t++)
        {
            float* baseP = pqkv + t * (long)packStride * w;
            long b = t / seq, s = t % seq;
            for (int h = 0; h < heads; h++)
            {
                long outOff = ((b * heads + h) * seq + s) * headDim;
                if (pq is not null)
                {
                    float* qs = baseP + (long)qSlot * w + h * headDim;
                    double qss = 0; for (int d = 0; d < headDim; d++) qss += (double)qs[d] * qs[d];
                    float qInv = (float)(1.0 / Math.Sqrt(qss / headDim + eps));
                    for (int d = 0; d < headDim; d++) pq[outOff + d] = qs[d] * qInv * pqw[d];
                }
                if (pk is not null)
                {
                    float* ks = baseP + (long)kSlot * w + h * headDim;
                    double kss = 0; for (int d = 0; d < headDim; d++) kss += (double)ks[d] * ks[d];
                    float kInv = (float)(1.0 / Math.Sqrt(kss / headDim + eps));
                    for (int d = 0; d < headDim; d++) pk[outOff + d] = ks[d] * kInv * pkw[d];
                }
                if (pv is not null)
                {
                    float* vs = baseP + (long)vSlot * w + h * headDim;
                    for (int d = 0; d < headDim; d++) pv[outOff + d] = vs[d];
                }
            }
        }
    }

    /// <summary>FourierEmbedder (freqs 2^i, include_input, no include_pi): writes x, sin, cos per band into a 3·(2·bands+1) row.</summary>
    unsafe void FourierEmbed(Tensor dst, Tensor coords, int count, int bands)
    {
        if (dst.DType != DType.F32 || coords.DType != DType.F32) throw new NotSupportedException("FourierEmbed default fallback only supports F32.");
        int dim = 3 * (2 * bands + 1);
        int sinBase = 3, cosBase = 3 + 3 * bands;
        float* dp = (float*)dst.DataPointer, cp = (float*)coords.DataPointer;
        for (int i = 0; i < count; i++)
        {
            long o = (long)i * dim;
            for (int c = 0; c < 3; c++)
            {
                float x = cp[i * 3 + c];
                dp[o + c] = x;
                for (int band = 0; band < bands; band++)
                {
                    float a = x * (1 << band);
                    dp[o + sinBase + c * bands + band] = MathF.Sin(a);
                    dp[o + cosBase + c * bands + band] = MathF.Cos(a);
                }
            }
        }
    }

    // ── Oasis spatio-temporal attention layout (default host impls = numeric reference;
    //    CudaBackend overrides with PTX kernels to keep the attention island GPU-resident) ──

    /// <summary>Frame-major qkv <c>[token,3·dim]</c> → <c>[b,heads,seq,headDim]</c>; spatial: b=frame; temporal: b=spatial,seq=frames.</summary>
    unsafe void OasisSplitHeads(Tensor output, Tensor qkv, int frames, int sp, int heads, int headDim, int part, bool temporal)
    {
        if (output.DType != DType.F32 || qkv.DType != DType.F32) throw new NotSupportedException("OasisSplitHeads default fallback only supports F32.");
        int dim = heads * headDim;
        int seq = temporal ? frames : sp;
        float* src = (float*)qkv.DataPointer, dst = (float*)output.DataPointer;
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < sp; i++)
            {
                long token = (long)f * sp + i;
                int b = temporal ? i : f, s = temporal ? f : i;
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < headDim; d++)
                        dst[(((long)b * heads + h) * seq + s) * headDim + d] = src[token * 3 * dim + (long)part * dim + (long)h * headDim + d];
            }
    }

    /// <summary>Interleaved (2i,2i+1) partial rotary on <c>[b,heads,seq,headDim]</c>, in place, rotating the first rotDim dims.</summary>
    unsafe void OasisRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int batch, int heads, int seq, int headDim, int rotDim)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32) throw new NotSupportedException("OasisRopeInterleaved default fallback only supports F32.");
        int pairs = rotDim / 2;
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp2 = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int s = 0; s < seq; s++)
                {
                    long xOff = (((long)b * heads + h) * seq + s) * headDim, cOff = (long)s * rotDim;
                    for (int p = 0; p < pairs; p++)
                    {
                        int i0 = 2 * p;
                        float re = xp[xOff + i0], im = xp[xOff + i0 + 1], c = cp[cOff + i0], sn = sp2[cOff + i0];
                        xp[xOff + i0] = re * c - im * sn;
                        xp[xOff + i0 + 1] = re * sn + im * c;
                    }
                }
    }

    /// <summary><c>[b, heads, seq, headDim]</c> → frame-major <c>[token, dim]</c> (inverse of <see cref="OasisSplitHeads"/>).</summary>
    unsafe void OasisMergeHeads(Tensor output, Tensor attn, int frames, int sp, int heads, int headDim, bool temporal)
    {
        if (output.DType != DType.F32 || attn.DType != DType.F32) throw new NotSupportedException("OasisMergeHeads default fallback only supports F32.");
        int dim = heads * headDim;
        int seq = temporal ? frames : sp;
        float* src = (float*)attn.DataPointer, dst = (float*)output.DataPointer;
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < sp; i++)
            {
                long token = (long)f * sp + i;
                int b = temporal ? i : f, s = temporal ? f : i;
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < headDim; d++)
                        dst[token * dim + (long)h * headDim + d] = src[(((long)b * heads + h) * seq + s) * headDim + d];
            }
    }

    /// <summary>Fused Oasis adaLN: <c>out[r,d] = LayerNorm(x[r])·(1+scale[f,d]) + shift[f,d]</c> (f=r/sp), scale/shift from <c>mod[f]</c>.</summary>
    unsafe void OasisAdaLn(Tensor output, Tensor input, Tensor mod, int dim, int sp, int totalRows, int modStride, int shiftOff, int scaleOff, float eps)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32 || mod.DType != DType.F32) throw new NotSupportedException("OasisAdaLn default fallback only supports F32.");
        float* xp = (float*)input.DataPointer, op = (float*)output.DataPointer, mp = (float*)mod.DataPointer;
        for (int r = 0; r < totalRows; r++)
        {
            long off = (long)r * dim;
            int f = r / sp;
            float* scale = mp + (long)f * modStride + scaleOff;
            float* shift = mp + (long)f * modStride + shiftOff;
            double mean = 0;
            for (int d = 0; d < dim; d++) mean += xp[off + d];
            mean /= dim;
            double variance = 0;
            for (int d = 0; d < dim; d++) { double dd = xp[off + d] - mean; variance += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(variance / dim) + eps);
            for (int d = 0; d < dim; d++)
                op[off + d] = (float)((xp[off + d] - mean) * inv) * (1f + scale[d]) + shift[d];
        }
    }

    /// <summary>Oasis per-frame unpatchify: <c>proj[t·sp, c·p²]</c> → <c>out[t,c,H,W]</c>, out-vector laid <c>[py,px,ci]</c>.</summary>
    unsafe void OasisUnpatchify(Tensor output, Tensor proj, int frames, int channels, int gh, int gw, int patch)
    {
        if (output.DType != DType.F32 || proj.DType != DType.F32) throw new NotSupportedException("OasisUnpatchify default fallback only supports F32.");
        int p = patch, h = gh * p, w = gw * p, outVec = channels * p * p;
        float* src = (float*)proj.DataPointer, dst = (float*)output.DataPointer;
        for (int f = 0; f < frames; f++)
            for (int ty = 0; ty < gh; ty++)
                for (int tx = 0; tx < gw; tx++)
                {
                    long token = ((long)f * gh + ty) * gw + tx;
                    for (int ci = 0; ci < channels; ci++)
                        for (int py = 0; py < p; py++)
                            for (int px = 0; px < p; px++)
                                dst[(((long)f * channels + ci) * h + ty * p + py) * w + tx * p + px] = src[token * outVec + (long)(py * p + px) * channels + ci];
                }
    }

    /// <summary>DIAMOND pixel quantize to 256 levels: <c>floor((clamp(v,−1,1)+1)·127.5)/127.5 − 1</c>.</summary>
    unsafe void PixelQuantize(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32) throw new NotSupportedException("PixelQuantize default fallback only supports F32.");
        long n = input.ElementCount;
        float* ip = (float*)input.DataPointer, op = (float*)output.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float v = ip[i];
            if (v > 1f) v = 1f; else if (v < -1f) v = -1f;
            int b = (int)((v + 1f) * 0.5f * 255f);
            op[i] = b / 255f * 2f - 1f;
        }
    }

    /// <summary>TripoSR triplane grid-sample: bilinearly samples the 3 feature planes into a concatenated <c>[3·C]</c> vector per point.</summary>
    unsafe void TriplaneGridSample(Tensor outputF, Tensor planes, Tensor? coords, long chunkStart,
        int count, int channels, int planeH, int planeW, float radius, int gridRes)
    {
        if (outputF.DType != DType.F32 || planes.DType != DType.F32)
            throw new NotSupportedException("TriplaneGridSample default fallback only supports F32.");
        float* pf = (float*)planes.DataPointer, of = (float*)outputF.DataPointer;
        float* cp = coords is null ? null : (float*)coords.DataPointer;
        int c = channels, h = planeH, w = planeW;
        long plane2d = (long)h * w, planeSz = (long)c * h * w;
        static void SamplePlane(float* plane, int c, int h, int w, long plane2d, float ga, float gb, float* dst)
        {
            float fx = ((ga + 1f) * w - 1f) * 0.5f, fy = ((gb + 1f) * h - 1f) * 0.5f;
            int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy), x1 = x0 + 1, y1 = y0 + 1;
            float tx = fx - x0, ty = fy - y0;
            float w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
            bool x0ok = x0 >= 0 && x0 < w, x1ok = x1 >= 0 && x1 < w, y0ok = y0 >= 0 && y0 < h, y1ok = y1 >= 0 && y1 < h;
            for (int ch = 0; ch < c; ch++)
            {
                float* b = plane + ch * plane2d;
                float acc = 0f;
                if (y0ok && x0ok) acc += w00 * b[y0 * w + x0];
                if (y0ok && x1ok) acc += w10 * b[y0 * w + x1];
                if (y1ok && x0ok) acc += w01 * b[y1 * w + x0];
                if (y1ok && x1ok) acc += w11 * b[y1 * w + x1];
                dst[ch] = acc;
            }
        }
        for (int t = 0; t < count; t++)
        {
            float x, y, z;
            if (gridRes > 0)
            {
                long lin = chunkStart + t;
                int iz = (int)(lin % gridRes), iy = (int)((lin / gridRes) % gridRes), ix = (int)(lin / ((long)gridRes * gridRes));
                float inv = gridRes > 1 ? 1f / (gridRes - 1) : 0f, span = 2f * radius;
                x = -radius + ix * inv * span; y = -radius + iy * inv * span; z = -radius + iz * inv * span;
            }
            else { x = cp[t * 3]; y = cp[t * 3 + 1]; z = cp[t * 3 + 2]; }
            float gx = x / radius, gy = y / radius, gz = z / radius;
            long baseF = (long)t * 3 * c;
            SamplePlane(pf, c, h, w, plane2d, gx, gy, of + baseF);
            SamplePlane(pf + planeSz, c, h, w, plane2d, gx, gz, of + baseF + c);
            SamplePlane(pf + 2 * planeSz, c, h, w, plane2d, gy, gz, of + baseF + 2 * c);
        }
    }

    /// <summary>GEGLU with exact (erf) GELU gate: <c>output[rows,inner] = proj[:,:inner] · gelu_erf(proj[:,inner:])</c>, fused split+gate.</summary>
    unsafe void GegluErf(Tensor output, Tensor proj, long rows, int inner)
    {
        if (output.DType != DType.F32 || proj.DType != DType.F32)
            throw new NotSupportedException("GegluErf default fallback only supports F32.");
        float* pp = (float*)proj.DataPointer, gp = (float*)output.DataPointer;
        for (long row = 0; row < rows; row++)
        {
            float* hin = pp + row * 2 * inner, gin = hin + inner, gout = gp + row * inner;
            for (int i = 0; i < inner; i++)
            {
                float g = gin[i];
                gout[i] = hin[i] * (0.5f * g * (1f + Erf(g * 0.70710678118654752440f)));
            }
        }
    }

    /// <summary>Exact (erf) GELU: <c>out = 0.5·x·(1+erf(x/√2))</c>, PyTorch's <c>nn.GELU()</c>; <see cref="Gelu"/> is the tanh approx.</summary>
    unsafe void GeluErf(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException($"GeluErf default fallback only supports F32 — got input={input.DType}, output={output.DType}.");
        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        long count = output.ElementCount;
        for (long i = 0; i < count; i++)
        {
            float x = pIn[i];
            pOut[i] = 0.5f * x * (1f + Erf(x * 0.70710678118654752440f));
        }
    }

    /// <summary>Abramowitz-Stegun 7.1.26 rational polynomial approximation of the Gauss error function (max error ~1.5e-7).</summary>
    private static float Erf(float x)
    {
        int sign = x < 0 ? -1 : 1;
        x = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * x);
        float y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    /// <summary>AdaLN modulation split: chunks <c>proj [B,4D]</c> into 4 <c>[B,D]</c>, <c>1+x</c> for scales, <c>tanh(x)</c> for gates.</summary>
    unsafe void ModulationSplit4(Tensor scaleMsa, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp, Tensor proj)
    {
        if (proj.DType != DType.F32)
            throw new NotSupportedException("ModulationSplit4 default fallback only supports F32.");
        int dim = (int)scaleMsa.Shape[scaleMsa.Shape.Rank - 1];
        int batch = (int)(scaleMsa.ElementCount / dim);
        float* pProj = (float*)proj.DataPointer;
        float* sMsa = (float*)scaleMsa.DataPointer;
        float* gMsa = (float*)gateMsa.DataPointer;
        float* sMlp = (float*)scaleMlp.DataPointer;
        float* gMlp = (float*)gateMlp.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long src = (long)b * 4 * dim;
            long dst = (long)b * dim;
            for (int d = 0; d < dim; d++)
            {
                sMsa[dst + d] = 1.0f + pProj[src + d];
                gMsa[dst + d] = MathF.Tanh(pProj[src + dim + d]);
                sMlp[dst + d] = 1.0f + pProj[src + 2 * dim + d];
                gMlp[dst + d] = MathF.Tanh(pProj[src + 3 * dim + d]);
            }
        }
    }

    /// <summary>MoE top-k gate weights (HiDream <c>MoEGate</c>): softmax, top-k, renormalize to sum 1, write a dense <c>[B,S,E]</c> table.</summary>
    unsafe void MoeTopKGate(Tensor weights, Tensor logits, int topK)
    {
        if (logits.DType != DType.F32 || weights.DType != DType.F32)
            throw new NotSupportedException("MoeTopKGate default fallback only supports F32.");
        int numExperts = (int)logits.Shape[logits.Shape.Rank - 1];
        long tokens = logits.ElementCount / numExperts;
        float* lp = (float*)logits.DataPointer;
        float* wp = (float*)weights.DataPointer;
        Span<float> probs = stackalloc float[numExperts];
        int k = Math.Min(topK, numExperts);
        for (long t = 0; t < tokens; t++)
        {
            long off = t * numExperts;
            float maxLogit = lp[off];
            for (int e = 1; e < numExperts; e++)
                if (lp[off + e] > maxLogit) maxLogit = lp[off + e];
            float sum = 0f;
            for (int e = 0; e < numExperts; e++)
            {
                float ex = MathF.Exp(lp[off + e] - maxLogit);
                probs[e] = ex;
                sum += ex;
            }
            for (int e = 0; e < numExperts; e++)
            {
                probs[e] /= sum;
                wp[off + e] = 0f;
            }
            float denom = 0f;
            for (int kk = 0; kk < k; kk++)
            {
                int best = -1;
                float bestVal = float.NegativeInfinity;
                for (int e = 0; e < numExperts; e++)
                {
                    if (wp[off + e] != 0f) continue;
                    if (probs[e] > bestVal) { bestVal = probs[e]; best = e; }
                }
                wp[off + best] = bestVal;
                denom += bestVal;
            }
            denom += 1e-20f;
            for (int e = 0; e < numExperts; e++)
                if (wp[off + e] != 0f) wp[off + e] /= denom;
        }
    }

    /// <summary>MoE expert combine, in-place: <c>inout += gate[expertIndex] * value</c>, gate from <see cref="MoeTopKGate"/>'s dense table.</summary>
    unsafe void RowGatedAccumulate(Tensor inout, Tensor value, Tensor gate, int numExperts, int expertIndex)
    {
        if (inout.DType != DType.F32 || value.DType != DType.F32 || gate.DType != DType.F32)
            throw new NotSupportedException("RowGatedAccumulate default fallback only supports F32.");
        int dim = (int)inout.Shape[inout.Shape.Rank - 1];
        long rows = inout.ElementCount / dim;
        float* op = (float*)inout.DataPointer;
        float* vp = (float*)value.DataPointer;
        float* gp = (float*)gate.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            float g = gp[r * numExperts + expertIndex];
            if (g == 0f) continue;
            long off = r * dim;
            for (int d = 0; d < dim; d++)
                op[off + d] += g * vp[off + d];
        }
    }

    /// <summary>CFG combine + Euler step, in-place on <paramref name="z"/>:
    /// <c>v = guidance*pos + (1-guidance)*neg; z += v*delta</c>. <paramref name="pos"/> and
    /// <paramref name="neg"/> may be the same tensor; neither may alias the mutated <paramref name="z"/>.</summary>
    unsafe void CfgEulerStep(Tensor z, Tensor pos, Tensor neg, float guidance, float delta)
    {
        if (z.DType != DType.F32 || pos.DType != DType.F32 || neg.DType != DType.F32)
            throw new NotSupportedException("CfgEulerStep default fallback only supports F32.");
        if (!z.Shape.Equals(pos.Shape) || !z.Shape.Equals(neg.Shape))
            throw new ArgumentException(
                $"CfgEulerStep requires identical shapes; got z={z.Shape}, pos={pos.Shape}, neg={neg.Shape}.");
        if (z.ElementCount <= 0)
            throw new ArgumentException("CfgEulerStep requires nonempty tensors.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(pos) || z.HasOverlappingHostStorageWithoutSync(neg))
            throw new ArgumentException("CfgEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        long count = z.ElementCount;
        float* pZ = (float*)z.DataPointer;
        float* pPos = (float*)pos.DataPointer;
        float* pNeg = (float*)neg.DataPointer;
        for (long i = 0; i < count; i++)
        {
            float v = guidance * pPos[i] + (1.0f - guidance) * pNeg[i];
            pZ[i] = pZ[i] + v * delta;
        }
    }

    /// <summary>
    /// Elementwise affine mix: <c>output = xScale·x + yScale·y</c>. Inputs may alias each other, but neither
    /// input storage may overlap <paramref name="output"/>. All tensors are F32, nonempty, and identically shaped.
    /// </summary>
    unsafe void AffineMix(Tensor output, Tensor x, Tensor y, float xScale, float yScale)
    {
        long count = MixContract.ValidateAffineMix(output, x, y, xScale, yScale);
        float* outputPointer = (float*)output.DataPointer;
        float* xPointer = (float*)x.DataPointer;
        float* yPointer = (float*)y.DataPointer;
        for (long i = 0; i < count; i++)
            outputPointer[i] = xScale * xPointer[i] + yScale * yPointer[i];
        GC.KeepAlive(output);
        GC.KeepAlive(x);
        GC.KeepAlive(y);
    }

    /// <summary>
    /// In-place masked affine replacement:
    /// <c>target = mask·target + (1−mask)·(sourceScale·source + noiseScale·noise)</c>. When
    /// <paramref name="noiseScale"/> is zero, <paramref name="noise"/> must be null and the noise term is omitted.
    /// Mask values are consumed exactly as supplied (never clamped). Source, noise, and mask remain unchanged.
    /// </summary>
    unsafe void MaskedAffineMixInPlace(
        Tensor target,
        Tensor source,
        Tensor? noise,
        Tensor mask,
        float sourceScale,
        float noiseScale,
        MaskBroadcastLayout layout)
    {
        MaskedMixGeometry geometry = MixContract.ValidateMaskedAffineMix(
            target, source, noise, mask, sourceScale, noiseScale, layout);
        float* targetPointer = (float*)target.DataPointer;
        float* sourcePointer = (float*)source.DataPointer;
        float* noisePointer = noise is null ? null : (float*)noise.DataPointer;
        float* maskPointer = (float*)mask.DataPointer;

        if (layout == MaskBroadcastLayout.DenseNchwBroadcast)
        {
            long batchPlane = checked(geometry.Channels * geometry.Spatial);
            for (long i = 0; i < geometry.ElementCount; i++)
            {
                long batch = i / batchPlane;
                long maskIndex = batch * geometry.Spatial + i % geometry.Spatial;
                MixAt(i, maskIndex);
            }
        }
        else
        {
            long channels = geometry.FeatureDimension / geometry.PatchArea;
            for (long i = 0; i < geometry.ElementCount; i++)
            {
                long feature = i % geometry.FeatureDimension;
                long token = i / geometry.FeatureDimension;
                long patchIndex = layout == MaskBroadcastLayout.PackedChannelOuter
                    ? feature % geometry.PatchArea
                    : feature / channels;
                MixAt(i, token * geometry.PatchArea + patchIndex);
            }
        }

        GC.KeepAlive(target);
        GC.KeepAlive(source);
        GC.KeepAlive(noise);
        GC.KeepAlive(mask);

        void MixAt(long elementIndex, long maskIndex)
        {
            float replacement = sourceScale * sourcePointer[elementIndex];
            if (noisePointer is not null)
                replacement += noiseScale * noisePointer[elementIndex];
            float maskValue = maskPointer[maskIndex];
            targetPointer[elementIndex] =
                targetPointer[elementIndex] * maskValue + replacement * (1f - maskValue);
        }
    }

    /// <summary>
    /// Global-renormalized CFG plus Euler update, in-place on <paramref name="z"/> while leaving
    /// <paramref name="cond"/> and <paramref name="uncond"/> unchanged:
    /// <c>guided = uncond + guidance·(cond−uncond)</c>,
    /// <c>scale = clamp(‖cond‖₂/(‖guided‖₂+1e-8), renormMin, 1)</c>,
    /// <c>z += delta·scale·guided</c>. The two prediction inputs may alias each other; neither may alias
    /// the mutated <paramref name="z"/>.
    /// </summary>
    unsafe void CfgRenormEulerStep(
        Tensor z, Tensor cond, Tensor uncond, float guidance, float delta, float renormMin)
    {
        if (z.DType != DType.F32 || cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new NotSupportedException("CfgRenormEulerStep default fallback only supports F32.");
        if (!z.Shape.Equals(cond.Shape) || !z.Shape.Equals(uncond.Shape))
            throw new ArgumentException(
                $"CfgRenormEulerStep requires identical shapes; got z={z.Shape}, cond={cond.Shape}, uncond={uncond.Shape}.");
        if (z.ElementCount <= 0)
            throw new ArgumentException("CfgRenormEulerStep requires nonempty tensors.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(cond) || z.HasOverlappingHostStorageWithoutSync(uncond))
            throw new ArgumentException(
                "CfgRenormEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        if (!float.IsFinite(renormMin) || renormMin < 0f || renormMin > 1f)
            throw new ArgumentOutOfRangeException(nameof(renormMin), "Renorm minimum must be finite and in [0,1].");

        long count = z.ElementCount;
        if (delta == 0f) return;

        float* pZ = (float*)z.DataPointer;
        float* pCond = (float*)cond.DataPointer;
        float* pUncond = (float*)uncond.DataPointer;
        double normCondSq = 0.0;
        double normGuidedSq = 0.0;
        for (long i = 0; i < count; i++)
        {
            float c = pCond[i];
            float guided = pUncond[i] + guidance * (c - pUncond[i]);
            normCondSq += (double)c * c;
            normGuidedSq += (double)guided * guided;
        }

        double ratio = Math.Sqrt(normCondSq) / (Math.Sqrt(normGuidedSq) + 1e-8);
        float scale = (float)Math.Clamp(ratio, renormMin, 1.0);
        for (long i = 0; i < count; i++)
        {
            float guided = pUncond[i] + guidance * (pCond[i] - pUncond[i]);
            guided *= scale;
            pZ[i] += guided * delta;
        }
    }

    /// <summary>
    /// Last-dimension-normalized CFG plus Euler update, in-place on <paramref name="z"/> while leaving
    /// <paramref name="cond"/> and <paramref name="uncond"/> unchanged. For every row formed by flattening
    /// all dimensions except the last:
    /// <c>guided = uncond + guidance·(cond−uncond)</c>,
    /// <c>guided *= ‖cond‖₂/(‖guided‖₂+eps)</c>, and <c>z += delta·guided</c>.
    /// The two prediction inputs may alias each other; neither may alias the mutated <paramref name="z"/>.
    /// </summary>
    unsafe void CfgNormalizedEulerStep(
        Tensor z, Tensor cond, Tensor uncond, float guidance, float delta, float eps = 1e-12f)
    {
        if (z.DType != DType.F32 || cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new NotSupportedException("CfgNormalizedEulerStep default fallback only supports F32.");
        if (!z.Shape.Equals(cond.Shape) || !z.Shape.Equals(uncond.Shape))
            throw new ArgumentException(
                $"CfgNormalizedEulerStep requires identical shapes; got z={z.Shape}, cond={cond.Shape}, uncond={uncond.Shape}.");
        if (z.Shape.Rank < 1 || z.ElementCount <= 0 || z.Shape[z.Shape.Rank - 1] <= 0)
            throw new ArgumentException("CfgNormalizedEulerStep requires a nonempty tensor with a positive last dimension.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(cond) || z.HasOverlappingHostStorageWithoutSync(uncond))
            throw new ArgumentException(
                "CfgNormalizedEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        if (!float.IsFinite(eps) || eps < 0f)
            throw new ArgumentOutOfRangeException(nameof(eps), "Normalization epsilon must be finite and nonnegative.");
        if (delta == 0f) return;

        long lastDim = z.Shape[z.Shape.Rank - 1];
        long rows = z.ElementCount / lastDim;
        float* pZ = (float*)z.DataPointer;
        float* pCond = (float*)cond.DataPointer;
        float* pUncond = (float*)uncond.DataPointer;
        for (long row = 0; row < rows; row++)
        {
            long rowOffset = row * lastDim;
            double condSq = 0.0;
            double guidedSq = 0.0;
            for (long d = 0; d < lastDim; d++)
            {
                long i = rowOffset + d;
                float c = pCond[i];
                float guided = pUncond[i] + guidance * (c - pUncond[i]);
                condSq += (double)c * c;
                guidedSq += (double)guided * guided;
            }

            // A zero denominator (eps=0 with an all-zero guided row) contributes nothing either way — use
            // ratio=0 rather than letting 0/0 produce NaN and poison z through 0·NaN.
            double denom = Math.Sqrt(guidedSq) + eps;
            float ratio = denom > 0.0 ? (float)(Math.Sqrt(condSq) / denom) : 0f;
            for (long d = 0; d < lastDim; d++)
            {
                long i = rowOffset + d;
                float guided = pUncond[i] + guidance * (pCond[i] - pUncond[i]);
                pZ[i] += delta * (guided * ratio);
            }
        }
    }

    /// <summary>In-place rotary on q/k <c>[B,L,numHeads,headDim]</c>: <c>out = x*cos + rotate_half(x)*sin</c> (Ideogram 4's ApplyRotary).</summary>
    unsafe void ApplyRope(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        if (q.DType != DType.F32 || k.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRope default fallback only supports F32.");
        int batch = (int)q.Shape[0];
        int seqLen = (int)q.Shape[1];
        int numHeads = (int)q.Shape[2];
        int headDim = (int)q.Shape[3];
        int half = headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    RotateHalfInPlace(qPtr + vecOff, cosPtr + freqBase, sinPtr + freqBase, half);
                    RotateHalfInPlace(kPtr + vecOff, cosPtr + freqBase, sinPtr + freqBase, half);
                }
            }
        }

        static void RotateHalfInPlace(float* vec, float* cos, float* sin, int half)
        {
            for (int i = 0; i < half; i++)
            {
                float lower = vec[i];
                float upper = vec[i + half];
                vec[i] = lower * cos[i] - upper * sin[i];
                vec[i + half] = upper * cos[i + half] + lower * sin[i + half];
            }
        }
    }

    /// <summary>In-place <b>interleaved (GPT-J)</b> rotary on <c>x [B,L,numHeads,headDim]</c>; NOT same as <see cref="ApplyRopeSingle"/>.</summary>
    /// <param name="rotaryDim">0 (or &gt;=headDim) rotates every pair; otherwise only <c>[0, rotaryDim)</c> pairs rotate (HF partial-rotary).</param>
    unsafe void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeInterleaved default fallback only supports F32.");
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        int half = headDim / 2;
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    float* vec = xPtr + vecOff;
                    float* c = cosPtr + freqBase;
                    float* si = sinPtr + freqBase;
                    for (int i = 0; i < half; i++)
                    {
                        if (2 * i >= rdim) break;
                        float xe = vec[2 * i];
                        float xo = vec[2 * i + 1];
                        vec[2 * i] = xe * c[i] - xo * si[i];
                        vec[2 * i + 1] = xo * c[i] + xe * si[i];
                    }
                }
            }
    }

    /// <summary>Wan-Video interleaved in-place RoPE (shared path): <c>x [S,heads·headDim]</c>, cos/sin shared across heads.</summary>
    unsafe void WanRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        int pairs = headDim / 2;
        for (int s = 0; s < seqLen; s++)
            for (int h = 0; h < heads; h++)
            {
                long xoff = ((long)s * heads + h) * headDim;
                long coff = (long)s * headDim;
                for (int i = 0; i < pairs; i++)
                {
                    int i0 = 2 * i;
                    float re = xp[xoff + i0], im = xp[xoff + i0 + 1];
                    float c = cp[coff + i0], sn = sp[coff + i0];
                    xp[xoff + i0] = re * c - im * sn;
                    xp[xoff + i0 + 1] = re * sn + im * c;
                }
            }
    }

    /// <summary>Matrix-Game 3.0 ActionModule: gathers a QKV slot into the batched temporal layout <c>[sp,heads,tt,headDim]</c>.</summary>
    unsafe void Mg3SplitQkvTemporal(Tensor outT, Tensor qkv, int tt, int sp, int heads, int headDim, int part, int stride)
    {
        int streamDim = heads * headDim, rowDim = stride * streamDim;
        float* src = (float*)qkv.DataPointer, dst = (float*)outT.DataPointer;
        for (int f = 0; f < tt; f++)
            for (int s = 0; s < sp; s++)
            {
                long token = (long)f * sp + s;
                for (int h = 0; h < heads; h++)
                    for (int d = 0; d < headDim; d++)
                        dst[(((long)s * heads + h) * tt + f) * headDim + d] = src[token * rowDim + part * streamDim + h * headDim + d];
            }
    }

    /// <summary>Matrix-Game 3.0 ActionModule: inverse of <see cref="Mg3SplitQkvTemporal"/>, back to token rows <c>[tt·sp, streamDim]</c>.</summary>
    unsafe void Mg3MergeTemporal(Tensor outT, Tensor attn, int tt, int sp, int heads, int headDim)
    {
        int streamDim = heads * headDim;
        float* src = (float*)attn.DataPointer, dst = (float*)outT.DataPointer;
        for (int s = 0; s < sp; s++)
            for (int h = 0; h < heads; h++)
                for (int f = 0; f < tt; f++)
                    for (int d = 0; d < headDim; d++)
                        dst[((long)f * sp + s) * streamDim + h * headDim + d] = src[(((long)s * heads + h) * tt + f) * headDim + d];
    }

    /// <summary>Matrix-Game 3.0 ActionModule: interleaved rope on <c>[sp,heads,tt,headDim]</c>, in place; grid row per broadcastSpatial.</summary>
    unsafe void Mg3RopeBatched(Tensor x, Tensor cos, Tensor sin, int sp, int heads, int tt, int headDim, int gh, int gw, bool broadcastSpatial)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, spn = (float*)sin.DataPointer;
        int pairs = headDim / 2;
        for (int s = 0; s < sp; s++)
            for (int h = 0; h < heads; h++)
                for (int f = 0; f < tt; f++)
                {
                    long gridRow = broadcastSpatial ? f : (long)f * gh * gw + s;
                    long cOff = gridRow * headDim, xOff = (((long)s * heads + h) * tt + f) * headDim;
                    for (int i = 0; i < pairs; i++)
                    {
                        int i0 = 2 * i;
                        float re = xp[xOff + i0], im = xp[xOff + i0 + 1];
                        float c = cp[cOff + i0], sn = spn[cOff + i0];
                        xp[xOff + i0] = re * c - im * sn;
                        xp[xOff + i0 + 1] = re * sn + im * c;
                    }
                }
    }

    /// <summary>Matrix-Game 3.0 ActionModule keyboard stream: expands K/V into <c>[sp,heads,tt,headDim]</c>, broadcast across sp.</summary>
    unsafe void Mg3KvExpand(Tensor kOut, Tensor vOut, Tensor kv, int sp, int heads, int tt, int headDim)
    {
        int streamDim = heads * headDim;
        float* kvp = (float*)kv.DataPointer, kp = (float*)kOut.DataPointer, vp = (float*)vOut.DataPointer;
        for (int s = 0; s < sp; s++)
            for (int h = 0; h < heads; h++)
                for (int f = 0; f < tt; f++)
                    for (int d = 0; d < headDim; d++)
                    {
                        long dst = (((long)s * heads + h) * tt + f) * headDim + d;
                        long kvBase = (long)f * 2 * streamDim + h * headDim + d;
                        kp[dst] = kvp[kvBase]; vp[dst] = kvp[kvBase + streamDim];
                    }
    }

    /// <summary>Matrix-Game 3.0 ActionModule mouse stream: builds the MLP input = hidden features concat per-frame mouse window.</summary>
    unsafe void Mg3MouseMlpConcat(Tensor outT, Tensor hidden, Tensor mouseWin, int tt, int sp, int imgDim, int winFloats)
    {
        int rowW = imgDim + winFloats;
        float* hp = (float*)hidden.DataPointer, wp = (float*)mouseWin.DataPointer, op = (float*)outT.DataPointer;
        for (int f = 0; f < tt; f++)
            for (int s = 0; s < sp; s++)
            {
                long token = (long)f * sp + s;
                for (int c = 0; c < imgDim; c++) op[token * rowW + c] = hp[token * imgDim + c];
                for (int wf = 0; wf < winFloats; wf++) op[token * rowW + imgDim + wf] = wp[(long)f * winFloats + wf];
            }
    }

    /// <summary>Per-head Wan interleaved RoPE (MG3 sigma_theta): like <see cref="WanRopeInterleaved"/>, cos/sin <c>[heads,S,headDim]</c>.</summary>
    unsafe void WanRopeInterleavedPerHead(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        int pairs = headDim / 2;
        for (int s = 0; s < seqLen; s++)
            for (int h = 0; h < heads; h++)
            {
                long xoff = ((long)s * heads + h) * headDim;
                long coff = ((long)h * seqLen + s) * headDim;
                for (int i = 0; i < pairs; i++)
                {
                    int i0 = 2 * i;
                    float re = xp[xoff + i0], im = xp[xoff + i0 + 1];
                    float c = cp[coff + i0], sn = sp[coff + i0];
                    xp[xoff + i0] = re * c - im * sn;
                    xp[xoff + i0 + 1] = re * sn + im * c;
                }
            }
    }

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
    {
        if (output.DType != DType.F32 || q.DType != DType.F32 || k.DType != DType.F32 || v.DType != DType.F32)
            throw new NotSupportedException($"Na3d default fallback only supports F32 — got output={output.DType}, q={q.DType}, k={k.DType}, v={v.DType}.");
        if (q.Shape.Rank != 6)
            throw new ArgumentException($"Na3d expects [batch, T, H, W, heads, headDim]; got {q.Shape}.", nameof(q));
        if (!q.Shape.Equals(k.Shape) || !q.Shape.Equals(v.Shape) || !q.Shape.Equals(output.Shape))
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

    /// <summary>LTX-2 "split" rotary (rotate-half, per-head cos), in-place on <c>x [seqLen,dim]</c>; matches <c>LtxVideo2Rope</c>.</summary>
    unsafe void Ltx2SplitRope(Tensor x, Tensor cos, Tensor sin, int seqLen, int numHeads, int headDim)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        int r = headDim / 2;
        int dim = numHeads * headDim;
        int cosWidth = dim / 2;
        for (int s = 0; s < seqLen; s++)
        {
            long xoff = (long)s * dim;
            long coff = (long)s * cosWidth;
            for (int h = 0; h < numHeads; h++)
            {
                long headBase = xoff + (long)h * headDim;
                long cosBase = coff + (long)h * r;
                for (int i = 0; i < r; i++)
                {
                    float a = xp[headBase + i];
                    float b = xp[headBase + i + r];
                    float c = cp[cosBase + i], sn = sp[cosBase + i];
                    xp[headBase + i] = a * c - b * sn;
                    xp[headBase + i + r] = b * c + a * sn;
                }
            }
        }
    }

    /// <summary>Extracts temporal frame <paramref name="ti"/> of a 5D <c>[B,C,Tsrc,H,W]</c> source into the 4D <c>[B,C,H,W]</c> output.</summary>
    unsafe void ExtractVaeFrame(Tensor output, Tensor src, int ti)
    {
        int b = (int)output.Shape[0], c = (int)output.Shape[1];
        long frameHW = output.ElementCount / ((long)b * c);
        int tsrc = (int)src.Shape[2];
        float* op = (float*)output.DataPointer, sp = (float*)src.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                Buffer.MemoryCopy(sp + ((((long)bi * c + ci) * tsrc + ti) * frameHW), op + (((long)bi * c + ci) * frameHW), frameHW * 4, frameHW * 4);
    }

    /// <summary>Writes 4D frame acc (plus optional per-channel bias) into temporal slot "to" of the 5D output, in place.</summary>
    unsafe void WriteVaeFrame(Tensor output, Tensor acc, Tensor? bias, int to)
    {
        int b = (int)acc.Shape[0], c = (int)acc.Shape[1];
        long frameHW = acc.ElementCount / ((long)b * c);
        int tout = (int)output.Shape[2];
        float* op = (float*)output.DataPointer, ap = (float*)acc.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                float bv = bp is null ? 0f : bp[ci];
                long srcOff = ((long)bi * c + ci) * frameHW, dstOff = (((long)bi * c + ci) * tout + to) * frameHW;
                for (long i = 0; i < frameHW; i++) op[dstOff + i] = ap[srcOff + i] + bv;
            }
    }

    /// <summary>Builds the padded input for BATCHED CausalConv3d: transpose + temporal pad + cache prepend + spatial
    /// pad. <paramref name="reflectSpatial"/> mirrors the H/W borders (<c>F.pad(mode="reflect")</c>, the LTX-2 VAE's
    /// default) instead of edge-clamping them.</summary>
    unsafe void BuildPaddedFrames(Tensor padded, Tensor input, Tensor? cache, int zeroPad,
        bool replicateFirst = false, int padH = 0, int padW = 0, bool reflectSpatial = false)
    {
        int paddedT = (int)padded.Shape[0], cIn = (int)padded.Shape[1];
        int h = (int)input.Shape[3], w = (int)input.Shape[4];
        int hp = h + 2 * padH, wp = w + 2 * padW;
        long hw = (long)h * w;
        int tin = (int)input.Shape[2];
        int cacheLen = cache is null ? 0 : (int)cache.Shape[2];
        float* pp = (float*)padded.DataPointer, ip = (float*)input.DataPointer;
        float* cp = cache is null ? null : (float*)cache.DataPointer;
        for (int t = 0; t < paddedT; t++)
            for (int ci = 0; ci < cIn; ci++)
            {
                long dst = ((long)t * cIn + ci) * hp * wp;
                if (t < zeroPad && !replicateFirst)
                {
                    for (long i = 0; i < (long)hp * wp; i++) pp[dst + i] = 0f;
                    continue;
                }
                int az = t < zeroPad ? 0 : t - zeroPad;   // replicate-first: pad frames clamp to the first content frame
                float* src;
                if (az < cacheLen) src = cp! + ((long)ci * cacheLen + az) * hw;
                else
                {
                    int ti = az - cacheLen;
                    if (ti >= tin) ti = tin - 1;          // trailing clamp (non-causal replicate)
                    src = ip + ((long)ci * tin + ti) * hw;
                }
                for (int y = 0; y < hp; y++)
                {
                    int sy = PadIndex(y - padH, h, reflectSpatial);
                    for (int x = 0; x < wp; x++)
                    {
                        int sx = PadIndex(x - padW, w, reflectSpatial);
                        pp[dst + (long)y * wp + x] = src[(long)sy * w + sx];
                    }
                }
            }
    }

    /// <summary>Maps an out-of-range spatial index to its source index — mirrored (<c>-1 → 1</c>, <c>len → len-2</c>,
    /// no edge repeat) when <paramref name="reflect"/>, else edge-clamped. The trailing clamp only bites when the pad
    /// exceeds <c>len-1</c>, where reflection has no defined answer.</summary>
    private static int PadIndex(int i, int len, bool reflect)
    {
        if (reflect)
        {
            if (i < 0) i = -i;
            else if (i >= len) i = 2 * (len - 1) - i;
        }
        return Math.Clamp(i, 0, len - 1);
    }

    /// <summary>Fills <paramref name="output"/> <c>[1,cOut,tout,H,W]</c> with per-channel bias (or 0 when null).</summary>
    unsafe void FillBias(Tensor output, Tensor? bias)
    {
        int cOut = (int)output.Shape[1], tout = (int)output.Shape[2];
        long hw = output.ElementCount / ((long)cOut * tout);
        float* op = (float*)output.DataPointer; float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int co = 0; co < cOut; co++)
        {
            float bv = bp is null ? 0f : bp[co];
            for (int to = 0; to < tout; to++) { long o = ((long)co * tout + to) * hw; for (long i = 0; i < hw; i++) op[o + i] = bv; }
        }
    }

    /// <summary>Temporal gather-sum for batched CausalConv3d, in place: <c>output[co][to][hw] += convDt[to·strideT+dt][co][hw]</c>.</summary>
    unsafe void AccumulateTap(Tensor output, Tensor convDt, int dt, int strideT)
    {
        int cOut = (int)output.Shape[1], tout = (int)output.Shape[2];
        long hw = output.ElementCount / ((long)cOut * tout);
        float* op = (float*)output.DataPointer, cp = (float*)convDt.DataPointer;
        for (int co = 0; co < cOut; co++)
            for (int to = 0; to < tout; to++)
            {
                int srcT = to * strideT + dt;
                long o = ((long)co * tout + to) * hw, s = ((long)srcT * cOut + co) * hw;
                for (long i = 0; i < hw; i++) op[o + i] += cp[s + i];
            }
    }

    /// <summary>SeedVR2 MAGViT channel→space shuffle: <c>[1,cIn,f,h,w] → [1,c,fFinal,h·sr,w·sr]</c> with
    /// <c>c = cIn/(sr²·tr)</c>, source channel <c>((xi·sr+yi)·tr+zi)·c+ci</c>, and (temporal only) OUTPUT
    /// FRAME INDEX 1 dropped (T→2T−1). Shapes carry every dim; F32/BF16. Default is the host reference.</summary>
    unsafe void SeedVr2PixelShuffle(Tensor output, Tensor input, int spatialRatio, int temporalRatio)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("SeedVr2PixelShuffle default fallback only supports F32.");
        Sync();
        int cIn = (int)input.Shape[1], f = (int)input.Shape[2], h = (int)input.Shape[3], w = (int)input.Shape[4];
        int c = (int)output.Shape[1], fFinal = (int)output.Shape[2], hOut = (int)output.Shape[3], wOut = (int)output.Shape[4];
        bool dropDup = temporalRatio > 1;
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        long total = (long)c * fFinal * hOut * wOut;
        for (long idx = 0; idx < total; idx++)
        {
            int ox = (int)(idx % wOut);
            long r = idx / wOut;
            int oy = (int)(r % hOut); r /= hOut;
            int of = (int)(r % fFinal);
            int ci = (int)(r / fFinal);
            int outF = dropDup && of >= 1 ? of + 1 : of;
            int fi = outF / temporalRatio, zi = outF % temporalRatio;
            int y = oy / spatialRatio, xi = oy % spatialRatio;
            int x = ox / spatialRatio, yi = ox % spatialRatio;
            int srcC = ((xi * spatialRatio + yi) * temporalRatio + zi) * c + ci;
            dst[idx] = src[(((long)srcC * f + fi) * h + y) * w + x];
        }
    }

    /// <summary>SeedVR2 asymmetric zero pad (right/bottom only, diffusers <c>Downsample2D(padding=0)</c>):
    /// <c>[B,C,T,h,w] → [B,C,T,h+1,w+1]</c>. F32/BF16. Default is the host reference.</summary>
    unsafe void SeedVr2PadBottomRight(Tensor output, Tensor input)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("SeedVr2PadBottomRight default fallback only supports F32.");
        Sync();
        int h = (int)input.Shape[3], w = (int)input.Shape[4];
        long planes = input.ElementCount / ((long)h * w);
        int hp = h + 1, wp = w + 1;
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        long total = planes * hp * wp;
        for (long idx = 0; idx < total; idx++)
        {
            int x = (int)(idx % wp);
            long r = idx / wp;
            int y = (int)(r % hp);
            long p = r / hp;
            dst[idx] = y < h && x < w ? src[(p * h + y) * w + x] : 0f;
        }
    }

    /// <summary>Wan2.2 VAE channel RMS norm: <c>scale = sqrt(C)/max(L2_over_C, eps)</c>, <c>out = x·scale·gamma</c>.</summary>
    unsafe void WanRmsNormChannel(Tensor output, Tensor input, Tensor? gamma, float eps)
    {
        int b = (int)input.Shape[0], c = (int)input.Shape[1];
        long spatial = input.ElementCount / ((long)b * c);
        float sqrtC = MathF.Sqrt(c);
        float* xp = (float*)input.DataPointer, op = (float*)output.DataPointer;
        float* g = gamma is null ? null : (float*)gamma.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            long baseB = (long)bi * c * spatial;
            for (long s = 0; s < spatial; s++)
            {
                double sumSq = 0;
                for (int ci = 0; ci < c; ci++) { float v = xp[baseB + (long)ci * spatial + s]; sumSq += (double)v * v; }
                float denom = MathF.Max((float)Math.Sqrt(sumSq), eps);
                float scale = sqrtC / denom;
                for (int ci = 0; ci < c; ci++)
                {
                    long off = baseB + (long)ci * spatial + s;
                    op[off] = xp[off] * scale * (g is null ? 1f : g[ci]);
                }
            }
        }
    }

    // ── Step-graph capture (CUDA-graph replay of a fixed per-step op sequence) ─────────────────────────────
    // A pipeline whose denoise step issues an IDENTICAL op sequence every step can capture it once and replay
    // with a single graph launch, collapsing thousands of per-op host calls. Contract: between Begin and
    // EndAndLaunch only async device work is issued (no DataPointer reads, no sync allocs); per-step-varying
    // inputs are refreshed into FIXED device buffers via CopyInto before each Launch. Defaults: unsupported.

    /// <summary>True when the backend can capture/replay a step graph (CUDA only).</summary>
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

    // ── Device-side decode position (autoregressive step-graph replay) ──────────────────────────────────────
    // A captured decode graph must NOT re-bake the per-step KV position into its kernel params, or every replay
    // would append/attend at the capture-time position. Instead the self-attention KV-append and flash kernels
    // read the position from a small device buffer {kvLen, qOffset}, refreshed (outside capture) each step. This
    // lets ONE captured graph replay at every step. Defaults: unsupported (eager backends use the host ints).

    /// <summary>True when the backend supports device-position decode-graph replay (CUDA only).</summary>
    bool GraphDecodeSupported => false;

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

    // ── LLM decode-graph: RoPE / embed / argmax reading & writing device state ──────────────────────────────
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

    /// <summary>Precomputes the full RoPE cos/sin table <c>[maxPos,headDim]</c> device-resident, so a decode step only indexes a row.</summary>
    /// <param name="splitHalfPartial">Split-half vs. interleaved rotary pairing when rotaryDim &lt; headDim; ignored for full rotary.</param>
    (Tensor cos, Tensor sin) BuildRopeTableDevice(int maxPos, int headDim, int rotaryDim, float theta, HartsyInference.Core.Rope.RopeScaling scaling, bool splitHalfPartial = false)
        => throw new NotSupportedException("BuildRopeTableDevice requires GraphDecodeSupported.");

    /// <summary>RoPE on a single decode-step Q/K tensor <c>[1,numHeads,1,headDim]</c>, reading its row from the devicePos qOffset slot.</summary>
    void RopeApplyDecodeStep(Tensor x, Tensor cosTable, Tensor sinTable, int rotaryDim, bool interleaved, ulong devicePos) { }

    /// <summary>Gathers one row (id in the persistent tokenId buffer) from a GPU-resident embedding table into output <c>[1,1,hidden]</c>.</summary>
    void EmbedGatherDecodeStep(Tensor output, Tensor embedTable, ulong tokenId) { }

    /// <summary>Argmax over the last dim, writing into a persistent buffer (e.g. <see cref="AllocDeviceTokenId"/>) to chain decode.</summary>
    void ArgMaxInto(ulong outputTokenId, Tensor input) { }

    // ── Graph-capture decode: repetition penalty ────────────────────────────────────────────────────────────
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
    void AppendTokenHistoryStep(ulong history, ulong historyCount, ulong tokenId) { }

    /// <summary>Applies HF-convention repetition penalty (divide positive/multiply negative logits) to every history token.</summary>
    void ApplyRepetitionPenaltyStep(Tensor logits, ulong history, ulong historyCount, float penalty) { }

    // ── Graph capture/replay (backend-agnostic handle) ──────────────────────────────────────────────────────
    // Callers outside HartsyInference.Cuda (e.g. TextGenerationPipeline, which must stay backend-agnostic —
    // CPU builds link against IBackend only) capture/replay through this opaque-handle indirection instead of
    // referencing CudaGraph directly. Default: unsupported: CaptureGraph returns null, meaning "not captured,
    // call recordWork() yourself every time" — callers must treat a null handle as the eager fallback, not an error.

    /// <summary>Captures the device work recordWork issues into a graph; returns a handle for <see cref="LaunchGraph"/>, or null.</summary>
    object? CaptureGraph(Action recordWork) => null;

    /// <summary>Replays a graph captured by <see cref="CaptureGraph"/> with a single launch.</summary>
    void LaunchGraph(object graphHandle) => throw new NotSupportedException("LaunchGraph requires a handle from a backend whose CaptureGraph returned non-null.");

    /// <summary>Disposes a graph handle from <see cref="CaptureGraph"/>.</summary>
    void DisposeGraph(object graphHandle) { }

    /// <summary>Image-output conversion: CHW F32 in [-1,1] → interleaved HWC u8 in [0,255] (round-half-up, clamped).</summary>
    unsafe void ChwF32ToHwcU8(Tensor output, Tensor input)
    {
        int height = (int)input.Shape[2], width = (int)input.Shape[3];
        long hw = (long)height * width;
        float* ip = (float*)input.DataPointer;
        byte* op = (byte*)output.DataPointer;
        for (long i = 0; i < hw; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                float v = (ip[c * hw + i] + 1.0f) * 0.5f;
                v = Math.Clamp(v, 0.0f, 1.0f);
                op[i * 3 + c] = (byte)(v * 255.0f + 0.5f);
            }
        }
    }

    /// <summary>
    /// DiT token-grid patchify: <c>[B,C,H,W]</c> → <c>[B,(H/p)·(W/p),p²·C]</c>.
    /// This byte-preserving shuffle supports F32, F16, and BF16.
    /// </summary>
    /// <param name="innerChannelFastest">Token order: true=(ph,pw,c) (Z-Image/Lumina2), false=channel-outer (Krea2/diffusers).</param>
    unsafe void PatchifyTokens(Tensor output, Tensor input, int patch, bool innerChannelFastest)
    {
        PatchTokenGeometry geometry = PatchTokenContract.ValidatePatchify(output, input, patch);
        PatchTokenHostShuffle.Patchify(output, input, geometry, patch, innerChannelFastest);
    }

    /// <summary>DiT token-grid unpatchify: <c>[B,hP·wP,C·p²]</c> → <c>[B,C,hP·p,wP·p]</c> (F32/F16/BF16).</summary>
    /// <param name="innerChannelFastest">Token order: true=(ph,pw,c) (Z-Image/Lumina2), false=channel-outer (Krea2/diffusers).</param>
    unsafe void UnpatchifyTokens(Tensor output, Tensor tokens, int channels, int hPacked, int wPacked, int patch,
        bool innerChannelFastest)
    {
        PatchTokenGeometry geometry = PatchTokenContract.ValidateUnpatchify(
            output, tokens, channels, hPacked, wPacked, patch);
        PatchTokenHostShuffle.Unpatchify(output, tokens, geometry, patch, innerChannelFastest);
    }

    /// <summary>Wan2.2 VAE unpatchify: <c>[b, c·p², t, h, w] → [b, c, t, h·p, w·p]</c>, unpack <c>oc = ci·p² + r·p + q</c>.</summary>
    unsafe void UnpatchifyVae(Tensor output, Tensor input, int patchSize)
    {
        int b = (int)input.Shape[0], packedC = (int)input.Shape[1], t = (int)input.Shape[2], h = (int)input.Shape[3], w = (int)input.Shape[4];
        int p = patchSize, c = packedC / (p * p), outH = h * p, outW = w * p;
        float* src = (float*)input.DataPointer, dst = (float*)output.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                    for (int hh = 0; hh < h; hh++)
                        for (int ww = 0; ww < w; ww++)
                            for (int q = 0; q < p; q++)
                                for (int r = 0; r < p; r++)
                                {
                                    int oc = ci * p * p + r * p + q;
                                    long srcOff = ((((long)bi * packedC + oc) * t + ti) * h + hh) * w + ww;
                                    long dstOff = ((((long)bi * c + ci) * t + ti) * outH + (hh * p + q)) * outW + (ww * p + r);
                                    dst[dstOff] = src[srcOff];
                                }
    }

    /// <summary>Wan2.2 VAE attention qkv split: <c>src [bt, 3c, h, w] → q,k,v each [bt, 1, hw, c]</c>. Default = host loop.</summary>
    unsafe void SplitVaeQkv(Tensor q, Tensor k, Tensor v, Tensor qkv, int bt, int c, int hw)
    {
        float* src = (float*)qkv.DataPointer, qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        long frame = hw;
        for (int i = 0; i < bt; i++)
            for (int ci = 0; ci < c; ci++)
                for (int token = 0; token < hw; token++)
                {
                    long srcBase = ((long)i * 3 * c) * frame + token;
                    long dstOff = ((long)i * hw + token) * c + ci;
                    qp[dstOff] = src[srcBase + (long)ci * frame];
                    kp[dstOff] = src[srcBase + (long)(c + ci) * frame];
                    vp[dstOff] = src[srcBase + (long)(2 * c + ci) * frame];
                }
    }

    /// <summary>Wan2.2 VAE attention output un-transpose: <c>attn [bt, 1, hw, c] → out [bt, c, h, w]</c>. Default = host loop.</summary>
    unsafe void VaeTokensToFrame(Tensor output, Tensor attn, int bt, int c, int hw)
    {
        float* a = (float*)attn.DataPointer, o = (float*)output.DataPointer;
        long frame = hw;
        for (int i = 0; i < bt; i++)
            for (int ci = 0; ci < c; ci++)
                for (int token = 0; token < hw; token++)
                    o[((long)i * c + ci) * frame + token] = a[((long)i * hw + token) * c + ci];
    }

    /// <summary>In-place rotary on <c>x [B,L,numHeads,headDim]</c> — same math as <see cref="ApplyRope"/>, for GQA head-count mismatch.</summary>
    unsafe void ApplyRopeSingle(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeSingle default fallback only supports F32.");
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        // Partial rotary (Phi-4 / StableLM): rotate only the first rotaryDim dims of each head (NEOX pairing
        // (i, i+half), half = rotaryDim/2), leaving the rest unchanged. 0 / full = the whole head. cos/sin keep
        // headDim stride (only their first rotaryDim entries are read).
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        int half = rdim / 2;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    float* vec = xPtr + vecOff;
                    float* c = cosPtr + freqBase;
                    float* si = sinPtr + freqBase;
                    for (int i = 0; i < half; i++)
                    {
                        float lower = vec[i];
                        float upper = vec[i + half];
                        vec[i] = lower * c[i] - upper * si[i];
                        vec[i + half] = upper * c[i + half] + lower * si[i + half];
                    }
                }
            }
    }

    /// <summary>Head-major <see cref="ApplyRopeSingle"/>: in-place rotary on <c>x [B,heads,L,headDim]</c>, cos/sin still <c>[B,L,headDim]</c>.</summary>
    /// <remarks>Lets q/k from <see cref="QkvSplitNormHeadMajor"/> be roped in place and fed straight to attention.</remarks>
    unsafe void ApplyRopeSingleHeadMajor(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeSingleHeadMajor default fallback only supports F32.");
        if (x.Shape.Rank != 4)
            throw new ArgumentException($"ApplyRopeSingleHeadMajor needs x shaped [B,heads,seq,headDim], got {x.Shape}.", nameof(x));
        int batch = (int)x.Shape[0];
        int numHeads = (int)x.Shape[1];
        int seqLen = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        // A token-major x is the same element count with heads/seq swapped; only the cos/sin row count catches it.
        long tableElements = (long)batch * seqLen * headDim;
        if (cos.ElementCount != tableElements || sin.ElementCount != tableElements)
            throw new ArgumentException(
                $"ApplyRopeSingleHeadMajor expects cos/sin [B,seq,headDim] for x {x.Shape}; got cos={cos.Shape}, sin={sin.Shape}.",
                cos.ElementCount != tableElements ? nameof(cos) : nameof(sin));
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        int half = rdim / 2;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                for (int s = 0; s < seqLen; s++)
                {
                    float* vec = xPtr + (((long)b * numHeads + h) * seqLen + s) * headDim;
                    long freqBase = ((long)b * seqLen + s) * headDim;
                    float* c = cosPtr + freqBase;
                    float* si = sinPtr + freqBase;
                    for (int i = 0; i < half; i++)
                    {
                        float lower = vec[i];
                        float upper = vec[i + half];
                        vec[i] = lower * c[i] - upper * si[i];
                        vec[i + half] = upper * c[i + half] + lower * si[i + half];
                    }
                }
    }

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

    /// <summary>Qwen3.5 SSM decode step (seq=1), part 1: causal depthwise conv step + SiLU + q/k/v split + per-head L2 norms.</summary>
    /// <remarks><paramref name="history"/> is the device-persistent conv ring, shifted in place. CUDA-only (graph-capable backends).</remarks>
    void SsmConvSplitStep(Tensor q, Tensor k, Tensor v, Tensor qkvMixed, Tensor history, Tensor convW,
        int kernel, int kHeads, int skDim, int vHeads, int svDim, float eps, float qScale)
        => throw new NotSupportedException("SsmConvSplitStep requires a CUDA backend.");

    /// <summary>Qwen3.5 SSM decode step (seq=1), part 2: delta-rule recurrence + gated RMSNorm.</summary>
    /// <remarks><paramref name="state"/> is the device-persistent [hv, sv, sk] recurrent state. CUDA-only.</remarks>
    void SsmDeltaStep(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor z,
        Tensor alphaRaw, Tensor betaRaw, Tensor dtBias, Tensor ssmA, Tensor normW,
        int hv, int sv, int sk, int repeat, float eps)
        => throw new NotSupportedException("SsmDeltaStep requires a CUDA backend.");

    /// <summary>Fused residual-add + RMSNorm: <c>residOut = a + b; normOut = rmsnorm(residOut) · weight</c>, one pass instead of two.</summary>
    void AddRmsNorm(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor weight, float eps)
    {
        Add(residOut, a, b);
        RmsNorm(normOut, residOut, weight, eps);
    }

    /// <summary>RMSNorm that may also emit a Q8_1 sidecar, so the dp4a GEMV skips its own quantize; semantically = <see cref="RmsNorm"/>.</summary>
    void RmsNormEmitQ8(Tensor output, Tensor input, Tensor weight, float eps)
        => RmsNorm(output, input, weight, eps);

    /// <summary>Residual-add + RMSNorm that may emit a Q8_1 sidecar (<see cref="RmsNormEmitQ8"/>); semantically = <see cref="AddRmsNorm"/>.</summary>
    void AddRmsNormEmitQ8(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor weight, float eps)
        => AddRmsNorm(residOut, normOut, a, b, weight, eps);

    /// <summary>Sandwich-norm fusion: <c>residOut = a + rmsnorm(b)·w1; normOut = rmsnorm(residOut)·w2</c>, one launch.</summary>
    /// <remarks>Replaces the post-attention norm, residual add, and pre-FFN norm; may emit a Q8_1 sidecar for
    /// <paramref name="normOut"/> (see <see cref="RmsNormEmitQ8"/>). Backends without a fused kernel fall back to the exact composition.</remarks>
    void NormAddRmsNormEmitQ8(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor w1, Tensor w2, float eps)
    {
        Tensor n1 = new(b.Shape, DType.F32);
        try
        {
            RmsNorm(n1, b, w1, eps);
            AddRmsNormEmitQ8(residOut, normOut, a, n1, w2, eps);
        }
        finally { n1.Dispose(); }
    }

    /// <summary>Sandwich-norm fusion: <c>output = a + rmsnorm(b)·w</c>, one launch replacing the post-FFN norm and residual add.</summary>
    /// <remarks>Backends without a fused kernel fall back to the exact composition.</remarks>
    void RmsNormAdd(Tensor output, Tensor a, Tensor b, Tensor weight, float eps)
    {
        Tensor n = new(b.Shape, DType.F32);
        try
        {
            RmsNorm(n, b, weight, eps);
            Add(output, a, n);
        }
        finally { n.Dispose(); }
    }

    /// <summary>Gated-FFN epilogue that may emit a Q8_1 sidecar (<see cref="RmsNormEmitQ8"/>); semantically = <see cref="GluActivate"/>.</summary>
    void GluActivateEmitQ8(Tensor output, Tensor gateUp, int ff, bool gelu)
        => GluActivate(output, gateUp, ff, gelu);

    /// <summary>Fused gated-FFN epilogue: <c>output[r,i] = act(gateUp[r,i])·gateUp[r,ff+i]</c> over the concatenated [gate|up] projection.</summary>
    /// <param name="gelu">false = SiLU (SwiGLU), true = GELU-tanh (GeGLU).</param>
    void GluActivate(Tensor output, Tensor gateUp, int ff, bool gelu)
    {
        long rows = gateUp.ElementCount / (2L * ff);
        TensorShape half = new(1, (int)rows, ff);
        Tensor gate = new(half, DType.F32);
        Tensor up = new(half, DType.F32);
        try
        {
            SliceLastDim(gate, gateUp, 0);
            SliceLastDim(up, gateUp, ff);
            if (gelu) Gelu(gate, gate);
            else Silu(gate, gate);
            Mul(output, gate, up);
        }
        finally
        {
            gate.Dispose();
            up.Dispose();
        }
    }

    /// <summary>Copies a contiguous last-dim slice: <c>out[row,d] = in[row,offset+d]</c>; splits a fused tensor (e.g. QKV) into a chunk.</summary>
    unsafe void SliceLastDim(Tensor output, Tensor input, int offset)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("SliceLastDim default fallback only supports F32.");
        // outDim is the slice width over the row, derived from row count (not the output's last dim,
        // which may be split into [.., heads, headDim]). rows = input rows = output rows.
        int inDim = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / inDim;
        int outDim = (int)(output.ElementCount / rows);
        long total = output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < total; i++)
        {
            int d = (int)(i % outDim);
            long row = i / outDim;
            pOut[i] = pIn[row * inDim + offset + d];
        }
    }

    /// <summary>Extracts a TIME sub-range from a KV tensor: <c>output[1,h,t,d]=input[1,h,start+t,d]</c>; slices dim 2, not the last dim.</summary>
    unsafe void SliceTimeRange(Tensor output, Tensor input, int start, int len)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("SliceTimeRange default fallback only supports F32.");
        int h = (int)input.Shape[1];
        int tIn = (int)input.Shape[2];
        int d = (int)input.Shape[3];
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int hi = 0; hi < h; hi++)
        {
            long srcBase = ((long)hi * tIn + start) * d;
            long dstBase = (long)hi * len * d;
            Buffer.MemoryCopy(pIn + srcBase, pOut + dstBase, (long)len * d * 4, (long)len * d * 4);
        }
    }

    /// <summary>Per-row scalar multiply: <c>out[row, c] = in[row, c] * rowMask[row]</c> — token masking for DiT embeddings.</summary>
    unsafe void MaskRows(Tensor output, Tensor input, Tensor rowMask)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32 || rowMask.DType != DType.F32)
            throw new NotSupportedException("MaskRows default fallback only supports F32.");
        int channels = (int)input.Shape[input.Shape.Rank - 1];
        long total = input.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        float* pMask = (float*)rowMask.DataPointer;
        for (long i = 0; i < total; i++)
            pOut[i] = pIn[i] * pMask[i / channels];
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

    /// <summary>Non-affine LayerNorm over the last dim: per row, zero mean and unit variance (biased var), no learned scale/bias.</summary>
    unsafe void LayerNormNoAffine(Tensor output, Tensor input, float eps)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("LayerNormNoAffine default fallback only supports F32.");
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / dim;
        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            long off = r * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += pIn[off + d];
            mean /= dim;
            float variance = 0f;
            for (int d = 0; d < dim; d++) { float diff = pIn[off + d] - mean; variance += diff * diff; }
            variance /= dim;
            float invStd = 1.0f / MathF.Sqrt(variance + eps);
            for (int d = 0; d < dim; d++) pOut[off + d] = (pIn[off + d] - mean) * invStd;
        }
    }

    /// <summary>In-place index-add of embedding rows: <c>h[row,d] += table[indices[row],d]</c>, keeping the embedding GPU-resident.</summary>
    unsafe void IndexAddRows(Tensor h, Tensor table, Tensor indices)
    {
        if (h.DType != DType.F32 || table.DType != DType.F32)
            throw new NotSupportedException("IndexAddRows default fallback only supports F32.");
        if (indices.DType != DType.I32)
            throw new NotSupportedException("IndexAddRows requires I32 indices.");
        int dim = (int)h.Shape[h.Shape.Rank - 1];
        long rows = h.ElementCount / dim;
        float* pH = (float*)h.DataPointer;
        float* pTable = (float*)table.DataPointer;
        int* pIdx = (int*)indices.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            long off = r * dim;
            long tOff = (long)pIdx[r] * dim;
            for (int d = 0; d < dim; d++) pH[off + d] += pTable[tOff + d];
        }
    }

    /// <summary>Per-row argmax over the last dim: <c>indices[r] = argmax_c input[r,c]</c> (first max wins) — keeps sampling GPU-resident.</summary>
    unsafe void ArgMaxLastDim(Tensor indices, Tensor input)
    {
        if (input.DType != DType.F32)
            throw new NotSupportedException("ArgMaxLastDim default fallback only supports F32.");
        if (indices.DType != DType.I32)
            throw new NotSupportedException("ArgMaxLastDim requires I32 indices.");
        int c = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / c;
        if (indices.ElementCount < rows)
            throw new ArgumentException($"ArgMaxLastDim indices length {indices.ElementCount} < rows {rows}.");
        float* pIn = (float*)input.DataPointer;
        int* pIdx = (int*)indices.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            float* row = pIn + r * c;
            int best = 0; float bv = row[0];
            for (int i = 1; i < c; i++) if (row[i] > bv) { bv = row[i]; best = i; }
            pIdx[r] = best;
        }
    }

    /// <summary>Scatter rows after a zeroed head block: <c>output = [zeros(headRows), input]</c> — builds the conditional latent.</summary>
    unsafe void ScatterRowsAfter(Tensor output, Tensor input, int headRows)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("ScatterRowsAfter default fallback only supports F32.");
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long total = output.ElementCount;
        long headElems = (long)headRows * dim;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < total; i++)
            pOut[i] = i < headElems ? 0f : pIn[i - headElems];
    }

    /// <summary>Writes a <c>[1, heads, c, hd]</c> chunk into sequence rows <c>[seqOffset, seqOffset + c)</c> of a
    /// <c>[1, heads, seq, hd]</c> head-major tensor, in place and accumulating across calls. Same bytes to the same
    /// offsets as concatenating every chunk along dim 2, but without holding the whole chunk list alive alongside the
    /// result — which is what makes long-sequence chunked attention fit (see <c>MiniMaxH3Transformer</c>).</summary>
    unsafe void ScatterSeqHeadMajor(Tensor output, Tensor input, int seqOffset)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("ScatterSeqHeadMajor default fallback only supports F32.");
        int heads = (int)output.Shape[1], seq = (int)output.Shape[2], hd = (int)output.Shape[3];
        int c = (int)input.Shape[2];
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int h = 0; h < heads; h++)
        {
            long dst = ((long)h * seq + seqOffset) * hd;
            long src = (long)h * c * hd;
            for (long i = 0; i < (long)c * hd; i++) pOut[dst + i] = pIn[src + i];
        }
    }

    /// <summary>Contiguous row-block slice: copies <c>output.ElementCount</c> elements starting at row rowOffset of input into output.</summary>
    unsafe void SliceRows(Tensor output, Tensor input, int rowOffset)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("SliceRows default fallback only supports F32.");
        int dim = (int)output.Shape[output.Shape.Rank - 1];
        long elemOffset = (long)rowOffset * dim;
        long total = output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < total; i++) pOut[i] = pIn[elemOffset + i];
    }

    /// <summary>Contiguous row-block slice, dtype-agnostic (a pure byte-range copy — used where <see cref="SliceRows"/>'s
    /// F32/F16/BF16 guard is too narrow, e.g. slicing an fp8 activation for chunked attention/MLP). Copies
    /// <see cref="Tensor.Fp8ScaleFactor"/> onto <paramref name="output"/> so a sliced fp8 chunk still carries its
    /// parent's per-tensor dequant scale into a downstream fp8 GEMM.</summary>
    unsafe void SliceRowsGeneric(Tensor output, Tensor input, int rowOffset)
    {
        if (output.DType != input.DType)
            throw new ArgumentException($"SliceRowsGeneric requires matching dtypes, got output {output.DType} vs input {input.DType}.");
        if (output.DType.IsQuantized)
            throw new NotSupportedException("SliceRowsGeneric does not support block-quantized dtypes.");
        int dim = (int)output.Shape[output.Shape.Rank - 1];
        long rowBytes = output.DType.ComputeByteCount(dim);
        long byteOffset = (long)rowOffset * rowBytes;
        long totalBytes = output.DType.ComputeByteCount(output.ElementCount);
        Buffer.MemoryCopy((byte*)input.DataPointer + byteOffset, output.DataPointer, totalBytes, totalBytes);
        output.Fp8ScaleFactor = input.Fp8ScaleFactor;
    }

    /// <summary>Writes <paramref name="input"/>'s rows into <paramref name="output"/> starting at row
    /// <paramref name="rowOffset"/>, in place and accumulating across calls — the byte-offset inverse of
    /// <see cref="SliceRowsGeneric"/>. Same bytes to the same offsets as concatenating every chunk along dim 0, but
    /// without holding the whole chunk list alive alongside the result, which is what makes long-sequence chunked
    /// attention/MLP fit (see <c>MiniMaxH3Transformer</c>; the head-major sibling is
    /// <see cref="ScatterSeqHeadMajor"/>).</summary>
    unsafe void ScatterRowsGeneric(Tensor output, Tensor input, int rowOffset)
    {
        if (output.DType != input.DType)
            throw new ArgumentException($"ScatterRowsGeneric requires matching dtypes, got output {output.DType} vs input {input.DType}.");
        if (output.DType.IsQuantized)
            throw new NotSupportedException("ScatterRowsGeneric does not support block-quantized dtypes.");
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long byteOffset = (long)rowOffset * input.DType.ComputeByteCount(dim);
        long totalBytes = input.DType.ComputeByteCount(input.ElementCount);
        Buffer.MemoryCopy(input.DataPointer, (byte*)output.DataPointer + byteOffset, totalBytes, totalBytes);
    }

    /// <summary>Fused quantized matmul: <c>output = input @ quantWeight^T (+ bias)</c>, dequantized in-kernel (Q8_0/Q4_K/Q5_K/Q6_K).</summary>
    void QuantizedMatMul(Tensor output, Tensor input, Tensor quantWeight, Tensor? bias)
        => throw new NotSupportedException(
            "QuantizedMatMul is implemented on the CUDA backend only; dequantize the weight to F16/F32 for CPU/Vulkan.");

    // ── Attention ───────────────────────────────────────────────────────

    /// <summary>Scaled dot-product attention: output = softmax(Q @ K^T / sqrt(d)) @ V</summary>
    /// <param name="allowF16">When true, a backend MAY run the attention in F16 for speed (halves score-matrix
    /// traffic + F16 tensor cores). Callers pass true ONLY when pre-softmax scores are bounded — e.g. Q/K are
    /// RMS/L2-normalized (Wan) — since unbounded scores (some fp8 archs) overflow F16 and produce black output.</param>
    void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false);

    /// <summary>Whether <see cref="ScaledDotProductAttentionTokenMajor"/> is served; false means the caller must
    /// build head-major <c>[B,H,S,D]</c> tensors and use <see cref="ScaledDotProductAttention"/> instead.</summary>
    bool SupportsTokenMajorAttention => false;

    /// <summary>SDPA whose Q/K/V/output are TOKEN-MAJOR rank-2 <c>[S, heads*headDim]</c> — the layout a Linear
    /// already emits and <c>to_out</c> already wants, so the caller pays no permute on either side. Semantics are
    /// otherwise identical to <see cref="ScaledDotProductAttention"/>.</summary>
    void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        int heads, int headDim, float scale, bool allowF16 = false)
        => throw new NotSupportedException($"{GetType().Name} does not serve token-major attention; check SupportsTokenMajorAttention first.");

    /// <summary>LTX-2 attention QK path in one pass: full-row RMS norm over <c>heads*headDim</c>, then optional
    /// SPLIT RoPE, then a head-major emit — <c>input [seq, heads*headDim]</c> to <c>output [1, heads, seq, headDim]</c>.
    /// Pass <paramref name="cos"/>/<paramref name="sin"/> null to skip the rotation (text cross-attention).
    /// INTERLEAVED rope is not served here; that caller must keep the unfused sequence.
    /// Default: composes the three ops it replaces, which is exactly what the unfused path did.</summary>
    unsafe void Ltx2QkNormRopeHeadMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin,
        int seq, int heads, int headDim, float eps)
    {
        Tensor normed = new Tensor(input.Shape, input.DType);
        try
        {
            RmsNorm(normed, input, normWeight, eps);
            if (cos is not null && sin is not null) Ltx2SplitRope(normed, cos, sin, seq, heads, headDim);
            Permute0213(output, normed, seq, heads, headDim);
        }
        finally { normed.Dispose(); }
    }

    /// <summary>Token-major twin of <see cref="Ltx2QkNormRopeHeadMajor"/> — <c>output</c> keeps the input's
    /// <c>[seq, heads*headDim]</c> shape. Default: composes the two ops it replaces.</summary>
    unsafe void Ltx2QkNormRopeTokenMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin,
        int seq, int heads, int headDim, float eps)
    {
        RmsNorm(output, input, normWeight, eps);
        if (cos is not null && sin is not null) Ltx2SplitRope(output, cos, sin, seq, heads, headDim);
    }

    /// <summary>Linear immediately followed by GELU. Backends may fold the activation into the GEMM epilogue;
    /// the default runs the two ops in sequence, which is what every caller did before.</summary>
    unsafe void LinearGelu(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        Linear(output, input, weight, bias);
        Gelu(output, output);
    }

    /// <summary>LTX-2 block norm+modulate in one pass: <c>out[r,d] = rms(in[r,:])[d] * (1+scale[d]) + shift[d]</c>,
    /// where the RMS carries no affine weight. Replaces RmsNorm-then-AddScalar-then-AffineBroadcastLastDim, which
    /// walked the activation twice. Default: composes exactly those ops.</summary>
    unsafe void Ltx2RmsModulate(Tensor output, Tensor input, Tensor onesWeight, Tensor scale, Tensor? shift, float eps)
    {
        int dim = (int)scale.Shape[scale.Shape.Rank - 1];
        Tensor normed = new Tensor(input.Shape, input.DType);
        Tensor scale1 = new Tensor(new TensorShape(dim), scale.DType);
        try
        {
            RmsNorm(normed, input, onesWeight, eps);
            AddScalar(scale1, scale, 1f);
            AffineBroadcastLastDim(output, normed, scale1, shift);
        }
        finally { normed.Dispose(); scale1.Dispose(); }
    }

    /// <summary>LTX-2 per-head output gate, IN PLACE: <c>x[s, h*headDim+d] *= 2*sigmoid(logits[s,h])</c>.
    /// Replaces expanding the per-(token,head) scalar to a full <c>[seq, heads*headDim]</c> tensor through a
    /// constant 0/1 GEMM and then multiplying.</summary>
    unsafe void Ltx2HeadGate(Tensor x, Tensor logits, int seq, int heads, int headDim)
    {
        if (x.DType != DType.F32 || logits.DType != DType.F32)
            throw new NotSupportedException($"Ltx2HeadGate default fallback only supports F32, got x={x.DType}, logits={logits.DType}.");
        float* p = (float*)x.DataPointer;
        float* g = (float*)logits.DataPointer;
        int w = heads * headDim;
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
            {
                float lg = g[(long)s * heads + h];
                float sig = lg >= 0f ? 1f / (1f + MathF.Exp(-lg)) : MathF.Exp(lg) / (1f + MathF.Exp(lg));
                long baseIdx = (long)s * w + (long)h * headDim;
                for (int d = 0; d < headDim; d++) p[baseIdx + d] *= 2f * sig;
            }
    }

    /// <summary>True when the backend has F16-activation kernels (Linear/Gelu/norms/cast); DiT F16 fast paths must gate on this.</summary>
    bool SupportsF16Activations => false;

    /// <summary>True when <see cref="RelativeL1Distance"/> runs without pulling operands to the host; feature caching must gate on this.</summary>
    bool SupportsDeviceStepCacheGate => false;

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

    /// <summary>
    /// Relative-L1 distance <c>Σ|a−b| / Σ|b|</c>; zero when both tensors are zero, positive infinity
    /// when only the reference is zero, and NaN when an operand or an accumulated sum is non-finite.
    /// </summary>
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

    /// <summary>True when the backend has the GPU-resident decode toolkit, so autoregressive decoders (e.g. Zonos) can run fully on-device.</summary>
    bool FlashDecodeSupported => false;

    /// <summary>Fused online-softmax attention that never materializes the score matrix; grouped-query aware (head h reads kv h/kvGroup).</summary>
    /// <param name="causal">When true, query row r (absolute position qOffset+r) attends keys 0..qOffset+r; otherwise all Lk keys.</param>
    /// <param name="softcap">&gt; 0 soft-caps each logit via <c>softcap·tanh(score/softcap)</c> before softmax (Gemma-2); 0 disables it.</param>
    /// <param name="sink">Non-null <c>[Hq]</c> F32 tensor (GPT-OSS sinks): each head's sink logit joins the softmax denominator only.</param>
    /// <param name="slidingWindow">&gt; 0 restricts each query to the most recent N keys (Gemma-2/3, Cohere2, GPT-OSS); else full causal.</param>
    /// <param name="alibiSlopes">Non-null <c>[Hq]</c> F32 tensor: adds ALiBi penalty <c>slope_h·(k_pos−q_pos)</c> (MPT/BLOOM/Falcon/Jais).</param>
    unsafe void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap = 0f, Tensor? sink = null, int slidingWindow = 0, Tensor? alibiSlopes = null)
        => AttentionReference.FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink, slidingWindow, alibiSlopes);

    /// <summary>Gathers rows: <c>output[m] = input[rowIndices[m]]</c> — used for MoE expert dispatch (collects routed tokens).</summary>
    unsafe void GatherRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        if (ReferenceEquals(output, input))
            throw new ArgumentException("GatherRows does not support an in-place output.", nameof(output));
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("GatherRows default fallback only supports F32.");
        if (input.Shape.Rank < 1)
            throw new ArgumentException("GatherRows input must have at least one dimension.", nameof(input));
        long width = input.Shape[input.Shape.Rank - 1];
        if (width <= 0 || width > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(input), $"GatherRows row width must be in [1,{int.MaxValue}]; got {width}.");
        long inputRows = input.ElementCount / width;
        long requiredOutputElements = checked((long)rowIndices.Length * width);
        if (output.ElementCount != requiredOutputElements)
            throw new ArgumentException(
                $"GatherRows output has {output.ElementCount} elements; {rowIndices.Length} rows of width {width} require {requiredOutputElements}.",
                nameof(output));
        for (int i = 0; i < rowIndices.Length; i++)
        {
            if (rowIndices[i] < 0 || (long)rowIndices[i] >= inputRows)
                throw new ArgumentOutOfRangeException(
                    nameof(rowIndices), rowIndices[i], $"Row index at position {i} is outside [0,{inputRows}).");
        }
        int k = (int)width;
        int m = rowIndices.Length;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int row = 0; row < m; row++)
        {
            long src = (long)rowIndices[row] * k;
            long dst = (long)row * k;
            Buffer.MemoryCopy(pIn + src, pOut + dst, (long)k * 4, (long)k * 4);
        }
    }

    /// <summary>Attempts a device-native row gather only when <paramref name="input"/> is already resident.
    /// Returns <c>false</c> without uploading or modifying <paramref name="output"/> when that fast path is not
    /// available. This prevents a small embedding lookup from pulling a multi-gigabyte vocabulary table onto a
    /// memory-constrained device merely to collect a handful of rows.</summary>
    bool TryGatherRowsResident(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices) => false;

    /// <summary>Scatter-add with per-row scale: <c>output[rowIndices[m]] += scales[m]*input[m]</c> — combines an MoE expert's output.</summary>
    unsafe void ScatterAddWeightedRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices, ReadOnlySpan<float> scales)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("ScatterAddWeightedRows default fallback only supports F32.");
        int k = (int)input.Shape[input.Shape.Rank - 1];
        int m = rowIndices.Length;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int row = 0; row < m; row++)
        {
            float s = scales[row];
            long dst = (long)rowIndices[row] * k;
            long src = (long)row * k;
            for (int j = 0; j < k; j++) pOut[dst + j] += s * pIn[src + j];
        }
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

    // ── Activations ─────────────────────────────────────────────────────

    /// <summary>GELU activation using the standard tanh approximation; use <see cref="GeluErf"/> for the exact erf form.</summary>
    void Gelu(Tensor output, Tensor input);

    /// <summary>SiLU activation (x * sigmoid(x)).</summary>
    void Silu(Tensor output, Tensor input);

    /// <summary>Mish activation (x * tanh(softplus(x))) — the Matcha/CosyVoice CFM decoder's block activation.</summary>
    unsafe void Mish(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("Mish default fallback only supports F32.");
        long count = output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < count; i++)
        {
            float x = pIn[i];
            float sp = x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
            pOut[i] = x * MathF.Tanh(sp);
        }
    }

    /// <summary>Sigmoid activation (1 / (1 + exp(-x))). Used by LSTM gating.</summary>
    void Sigmoid(Tensor output, Tensor input);

    /// <summary>Hyperbolic tangent activation. Used by LSTM cell update/output and several vocoders (Mish, snake-bias).</summary>
    void Tanh(Tensor output, Tensor input);

    /// <summary>ELU: <c>x if x &gt;= 0 else alpha * (exp(x) - 1)</c>. Used by SEANet residual blocks in EnCodec/DAC (alpha typically 1.0).</summary>
    void Elu(Tensor output, Tensor input, float alpha);

    /// <summary>Leaky ReLU: <c>x if x &gt;= 0 else slope * x</c>. Kokoro/StyleTTS 2 and HiFi-GAN MRF blocks use slope=0.2.</summary>
    void LeakyRelu(Tensor output, Tensor input, float slope);

    /// <summary>Snake: <c>x + (sin(alpha*x))^2 / divisor</c>; divisor=alpha (Oobleck VAE) or beta+1e-8 when supplied (BigVGAN snake-beta).</summary>
    void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta);

    /// <summary>Parametric ReLU: <c>x if x&gt;=0 else alpha*x</c>, alpha per-channel (<c>ElementCount==C</c>) or a single shared value.</summary>
    unsafe void Prelu(Tensor output, Tensor input, Tensor alpha)
    {
        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], timeDim = (int)input.Shape[2];
        float* op = (float*)output.DataPointer, ip = (float*)input.DataPointer, ap = (float*)alpha.DataPointer;
        bool perCh = alpha.ElementCount > 1;
        for (int bi = 0; bi < batch; bi++)
            for (int ci = 0; ci < channels; ci++)
            {
                float a = perCh ? ap[ci] : ap[0];
                long off = ((long)bi * channels + ci) * timeDim;
                for (int ti = 0; ti < timeDim; ti++) { float v = ip[off + ti]; op[off + ti] = v >= 0f ? v : a * v; }
            }
    }

    /// <summary>Frame-repeat over time on <c>[B,C,inT]</c>: each time-step duplicated numSamples times (nearest 1D upsample).</summary>
    unsafe void RepeatTime(Tensor output, Tensor input, int numSamples)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("RepeatTime default fallback only supports F32.");
        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], inT = (int)input.Shape[2];
        int outT = inT * numSamples;
        float* ip = (float*)input.DataPointer, op = (float*)output.DataPointer;
        for (int bc = 0; bc < batch * channels; bc++)
        {
            long inBase = (long)bc * inT, outBase = (long)bc * outT;
            for (int ti = 0; ti < inT; ti++)
            {
                float v = ip[inBase + ti];
                for (int s = 0; s < numSamples; s++) op[outBase + (long)ti * numSamples + s] = v;
            }
        }
    }

    // ── Element-wise ────────────────────────────────────────────────────

    /// <summary>Element-wise addition: output = a + b</summary>
    void Add(Tensor output, Tensor a, Tensor b);

    /// <summary>Element-wise multiplication: output = a * b</summary>
    void Mul(Tensor output, Tensor a, Tensor b);

    /// <summary>Scalar multiplication: output = input * scalar</summary>
    void Scale(Tensor output, Tensor input, float scalar);

    /// <summary>Element-wise clamp: output = clamp(input, min, max)</summary>
    void Clamp(Tensor output, Tensor input, float min, float max);

    // ── Transpose / Permute ─────────────────────────────────────────────

    /// <summary>Batched 2D transpose: [B, D1, D2] → [B, D2, D1].</summary>
    void Transpose2D(Tensor output, Tensor input, int d1, int d2);

    /// <summary>4D permute swapping dims 1 and 2: [B, S, H, D] → [B, H, S, D].</summary>
    void Permute0213(Tensor output, Tensor input, int s, int h, int d);

    /// <summary>GQA K/V head repeat: expands <c>[B,Hkv,L,D]</c> to <c>[B,Hkv*groupSize,L,D]</c>, matching HF <c>repeat_kv</c>.</summary>
    unsafe void RepeatKvHeads(Tensor output, Tensor input, int kvHeads, int groupSize)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("RepeatKvHeads default fallback only supports F32.");
        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[2];
        int headDim = (int)input.Shape[3];
        long perHead = (long)seqLen * headDim;
        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < kvHeads; h++)
            {
                long srcOff = ((long)b * kvHeads + h) * perHead;
                for (int g = 0; g < groupSize; g++)
                {
                    int qHead = h * groupSize + g;
                    long dstOff = ((long)b * (kvHeads * groupSize) + qHead) * perHead;
                    Buffer.MemoryCopy(ip + srcOff, op + dstOff, perHead * 4, perHead * 4);
                }
            }
    }

    /// <summary>GEGLU activation: splits input in half along last dim, applies GELU gate. Output has half the elements of input.</summary>
    void GeGlu(Tensor output, Tensor input);

    /// <summary>Broadcast add: hidden [B, C, ...spatial] += bias [B, C] in-place.</summary>
    void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial);

    // ── Shape Operations ────────────────────────────────────────────────

    /// <summary>Concatenate tensors along the specified dimension.</summary>
    void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim);

    /// <summary>
    /// Splits a nonempty contiguous F32, F16, or BF16 tensor into nonempty chunks along
    /// <paramref name="dim"/>. Outputs must have the same dtype and rank as the input, match every
    /// non-split dimension, exactly partition the split dimension, and share no storage with the input
    /// or one another. Payload bits are copied unchanged.
    /// </summary>
    void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim);

    // ── Convolution ─────────────────────────────────────────────────────

    /// <summary>1D convolution with asymmetric padding, channels-first; separate padLeft/padRight cover causal and symmetric padding.</summary>
    /// <param name="weight">PyTorch convention <c>[C_out, C_in / groups, K]</c>.</param>
    /// <param name="groups">Equal to channels for depthwise mode.</param>
    void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups);

    /// <summary>1D transposed convolution, channels-first; output length = <c>(T_in-1)*stride + dilation*(K-1) + 1 - pads</c>.</summary>
    /// <param name="weight">PyTorch convention <c>[C_in, C_out / groups, K]</c>.</param>
    /// <param name="padRight">VibeVoice/EnCodec causal decoders pass padLeft=0, padRight=K-stride to remove all trailing pad.</param>
    /// <param name="groups">Equal to channels for depthwise mode (e.g. BigVGAN anti-aliased upsampling).</param>
    void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups);

    // ── Sampling ────────────────────────────────────────────────────────

    /// <summary>Nearest-neighbor 2D upsample by the given scale factor.</summary>
    void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);

    /// <summary>Bilinear 2D upsample by the given scale factor.</summary>
    void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW);

    /// <summary>Bilinear 2D resize of an NCHW tensor to output's size, matching PyTorch <c>F.interpolate(mode="bilinear")</c> mapping.</summary>
    /// <param name="alignCorners">True: <c>src=dst·(in−1)/(out−1)</c>; false: <c>src=(dst+0.5)·in/out−0.5</c> (DPT fusion: true).</param>
    unsafe void InterpolateBilinear2D(Tensor output, Tensor input, bool alignCorners)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException($"InterpolateBilinear2D default fallback only supports F32 — got input={input.DType}, output={output.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"InterpolateBilinear2D requires 4D tensors; got input {input.Shape}, output {output.Shape}.");
        if (input.Shape[0] != output.Shape[0] || input.Shape[1] != output.Shape[1])
            throw new ArgumentException($"InterpolateBilinear2D batch/channel dims must match; got input {input.Shape}, output {output.Shape}.");

        int n = (int)input.Shape[0], c = (int)input.Shape[1];
        int inH = (int)input.Shape[2], inW = (int)input.Shape[3];
        int outH = (int)output.Shape[2], outW = (int)output.Shape[3];
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;

        for (int oy = 0; oy < outH; oy++)
        {
            float sy = alignCorners
                ? (outH == 1 ? 0f : oy * (float)(inH - 1) / (outH - 1))
                : (oy + 0.5f) * inH / outH - 0.5f;
            int y0 = (int)MathF.Floor(sy);
            float fy = sy - y0;
            int y0c = Math.Clamp(y0, 0, inH - 1), y1c = Math.Clamp(y0 + 1, 0, inH - 1);
            for (int ox = 0; ox < outW; ox++)
            {
                float sx = alignCorners
                    ? (outW == 1 ? 0f : ox * (float)(inW - 1) / (outW - 1))
                    : (ox + 0.5f) * inW / outW - 0.5f;
                int x0 = (int)MathF.Floor(sx);
                float fx = sx - x0;
                int x0c = Math.Clamp(x0, 0, inW - 1), x1c = Math.Clamp(x0 + 1, 0, inW - 1);
                for (long plane = 0; plane < (long)n * c; plane++)
                {
                    float* sp = src + plane * inH * inW;
                    float v0 = sp[y0c * inW + x0c] * (1f - fx) + sp[y0c * inW + x1c] * fx;
                    float v1 = sp[y1c * inW + x0c] * (1f - fx) + sp[y1c * inW + x1c] * fx;
                    dst[plane * outH * outW + (long)oy * outW + ox] = v0 * (1f - fy) + v1 * fy;
                }
            }
        }
    }

    /// <summary>2D transposed convolution (YOLO seg's Proto mask upsample); weight is <c>[C_in,C_out,kH,kW]</c> — input channels first.</summary>
    unsafe void ConvTranspose2d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException($"ConvTranspose2d default fallback only supports F32 — got input={input.DType}, output={output.DType}, weight={weight.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4 || weight.Shape.Rank != 4)
            throw new ArgumentException($"ConvTranspose2d requires 4D tensors; got input {input.Shape}, output {output.Shape}, weight {weight.Shape}.");

        int n = (int)input.Shape[0];
        int cIn = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int cOut = (int)output.Shape[1];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];
        if (weight.Shape[0] != cIn || weight.Shape[1] != cOut)
            throw new ArgumentException($"ConvTranspose2d weight shape [{weight.Shape[0]}, {weight.Shape[1]}, ...] must equal [C_in={cIn}, C_out={cOut}, ...].");

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;
        float* wBase = (float*)weight.DataPointer;
        float* bBase = bias is null ? null : (float*)bias.DataPointer;

        // Initialize output to bias (so the scatter-add can accumulate on top).
        for (int b = 0; b < n; b++)
        {
            for (int co = 0; co < cOut; co++)
            {
                float biasVal = bBase is null ? 0f : bBase[co];
                float* plane = dstBase + ((long)b * cOut + co) * oH * oW;
                for (long i = 0; i < (long)oH * oW; i++) plane[i] = biasVal;
            }
        }

        // Scatter-add: each input pixel (ci, yi, xi) contributes weight[ci, co, ky, kx] * value
        // to every output position (yi*sH+ky-pH, xi*sW+kx-pW) for each (co, ky, kx).
        for (int b = 0; b < n; b++)
        {
            for (int ci = 0; ci < cIn; ci++)
            {
                float* srcPlane = srcBase + ((long)b * cIn + ci) * iH * iW;
                for (int co = 0; co < cOut; co++)
                {
                    float* dstPlane = dstBase + ((long)b * cOut + co) * oH * oW;
                    long wOff = ((long)ci * cOut + co) * kH * kW;
                    for (int yi = 0; yi < iH; yi++)
                    {
                        for (int xi = 0; xi < iW; xi++)
                        {
                            float v = srcPlane[yi * iW + xi];
                            if (v == 0f) continue;
                            for (int ky = 0; ky < kH; ky++)
                            {
                                int yo = yi * strideH + ky - padH;
                                if (yo < 0 || yo >= oH) continue;
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int xo = xi * strideW + kx - padW;
                                    if (xo < 0 || xo >= oW) continue;
                                    dstPlane[yo * oW + xo] += wBase[wOff + ky * kW + kx] * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>Dense 3D convolution (gather form) over <c>[N,C,D,H,W]</c>; used by TRELLIS's sparse-structure VAE decoder.</summary>
    unsafe void Conv3d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException($"Conv3d default fallback only supports F32 — got input={input.DType}, output={output.DType}, weight={weight.DType}.");
        if (input.Shape.Rank != 5 || output.Shape.Rank != 5 || weight.Shape.Rank != 5)
            throw new ArgumentException($"Conv3d requires 5D [N,C,D,H,W] tensors; got input {input.Shape}, output {output.Shape}, weight {weight.Shape}.");

        int n = (int)input.Shape[0], cIn = (int)input.Shape[1];
        int iD = (int)input.Shape[2], iH = (int)input.Shape[3], iW = (int)input.Shape[4];
        int cOut = (int)output.Shape[1];
        int oD = (int)output.Shape[2], oH = (int)output.Shape[3], oW = (int)output.Shape[4];
        int kD = (int)weight.Shape[2], kH = (int)weight.Shape[3], kW = (int)weight.Shape[4];
        if (weight.Shape[0] != cOut || weight.Shape[1] != cIn)
            throw new ArgumentException($"Conv3d weight [{weight.Shape[0]},{weight.Shape[1]},...] must equal [C_out={cOut}, C_in={cIn}, ...].");

        float* srcBase = (float*)input.DataPointer, dstBase = (float*)output.DataPointer;
        float* wBase = (float*)weight.DataPointer, bBase = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < n; b++)
            for (int co = 0; co < cOut; co++)
            {
                float biasVal = bBase is null ? 0f : bBase[co];
                float* dst = dstBase + (((long)b * cOut + co) * oD) * oH * oW;
                for (int od = 0; od < oD; od++)
                    for (int oh = 0; oh < oH; oh++)
                        for (int ow = 0; ow < oW; ow++)
                        {
                            float acc = biasVal;
                            int id0 = od * strideD - padD, ih0 = oh * strideH - padH, iw0 = ow * strideW - padW;
                            for (int ci = 0; ci < cIn; ci++)
                            {
                                float* src = srcBase + (((long)b * cIn + ci) * iD) * iH * iW;
                                long wOff = (((long)co * cIn + ci) * kD) * kH * kW;
                                for (int kd = 0; kd < kD; kd++)
                                {
                                    int id = id0 + kd; if (id < 0 || id >= iD) continue;
                                    for (int kh = 0; kh < kH; kh++)
                                    {
                                        int ih = ih0 + kh; if (ih < 0 || ih >= iH) continue;
                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int iw = iw0 + kw; if (iw < 0 || iw >= iW) continue;
                                            acc += src[(id * iH + ih) * iW + iw] * wBase[wOff + (kd * kH + kh) * kW + kw];
                                        }
                                    }
                                }
                            }
                            dst[(od * oH + oh) * oW + ow] = acc;
                        }
            }
    }

    /// <summary>3D pixel-shuffle (TRELLIS SS-VAE upsample): <c>[N,C·r³,D,H,W] → [N,C,rD,rH,rW]</c>, matching reshape+permute.</summary>
    unsafe void PixelShuffle3d(Tensor output, Tensor input, int ratio)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("PixelShuffle3d default fallback only supports F32.");
        int n = (int)input.Shape[0], cIn = (int)input.Shape[1];
        int iD = (int)input.Shape[2], iH = (int)input.Shape[3], iW = (int)input.Shape[4];
        int r = ratio, r3 = r * r * r, cOut = cIn / r3;
        int oD = iD * r, oH = iH * r, oW = iW * r;
        float* src = (float*)input.DataPointer, dst = (float*)output.DataPointer;
        for (int b = 0; b < n; b++)
            for (int c = 0; c < cOut; c++)
                for (int d = 0; d < iD; d++)
                    for (int h = 0; h < iH; h++)
                        for (int w = 0; w < iW; w++)
                            for (int a = 0; a < r; a++)
                                for (int bb = 0; bb < r; bb++)
                                    for (int e = 0; e < r; e++)
                                    {
                                        long ci = (long)c * r3 + a * r * r + bb * r + e;
                                        long si = (((((long)b * cIn + ci) * iD + d) * iH + h) * iW + w);
                                        long di = (((((long)b * cOut + c) * oD + (d * r + a)) * oH + (h * r + bb)) * oW + (w * r + e));
                                        dst[di] = src[si];
                                    }
    }

    /// <summary>Channel-axis LayerNorm over <c>[N,C,D,H,W]</c> (TRELLIS <c>ChannelLayerNorm32</c>): normalizes across C, affine <c>[C]</c>.</summary>
    unsafe void ChannelLayerNorm3d(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32 || bias.DType != DType.F32)
            throw new NotSupportedException("ChannelLayerNorm3d default fallback only supports F32.");
        int n = (int)input.Shape[0], c = (int)input.Shape[1];
        long spatial = input.Shape[2] * input.Shape[3] * input.Shape[4];
        float* src = (float*)input.DataPointer, dst = (float*)output.DataPointer;
        float* wt = (float*)weight.DataPointer, bs = (float*)bias.DataPointer;
        for (int b = 0; b < n; b++)
            for (long p = 0; p < spatial; p++)
            {
                long baseIdx = (long)b * c * spatial + p;
                double mean = 0; for (int ch = 0; ch < c; ch++) mean += src[baseIdx + (long)ch * spatial]; mean /= c;
                double variance = 0; for (int ch = 0; ch < c; ch++) { double dd = src[baseIdx + (long)ch * spatial] - mean; variance += dd * dd; } variance /= c;
                float invStd = (float)(1.0 / Math.Sqrt(variance + eps));
                for (int ch = 0; ch < c; ch++)
                {
                    long idx = baseIdx + (long)ch * spatial;
                    dst[idx] = ((float)(src[idx] - mean)) * invStd * wt[ch] + bs[ch];
                }
            }
    }

    /// <summary>Scatters active-voxel features <c>[N,C]</c> onto a pre-zeroed grid <c>[1,C,R,R,R]</c> for submanifold conv.</summary>
    unsafe void SparseScatterToGrid(Tensor grid, Tensor feats, Tensor coords, int channels, int resolution)
    {
        int n = (int)(feats.ElementCount / channels); long r3 = (long)resolution * resolution * resolution;
        float* g = (float*)grid.DataPointer, f = (float*)feats.DataPointer; int* cp = (int*)coords.DataPointer;
        for (int v = 0; v < n; v++)
        {
            long cell = (long)cp[v * 4 + 1] * resolution * resolution + (long)cp[v * 4 + 2] * resolution + cp[v * 4 + 3];
            for (int c = 0; c < channels; c++) g[(long)c * r3 + cell] = f[(long)v * channels + c];
        }
    }

    /// <summary>Gathers a dense grid to active-voxel features <c>[N,C]</c>: inverse of <see cref="SparseScatterToGrid"/>.</summary>
    unsafe void SparseGatherFromGrid(Tensor feats, Tensor grid, Tensor coords, int channels, int resolution)
    {
        int n = (int)(feats.ElementCount / channels); long r3 = (long)resolution * resolution * resolution;
        float* g = (float*)grid.DataPointer, f = (float*)feats.DataPointer; int* cp = (int*)coords.DataPointer;
        for (int v = 0; v < n; v++)
        {
            long cell = (long)cp[v * 4 + 1] * resolution * resolution + (long)cp[v * 4 + 2] * resolution + cp[v * 4 + 3];
            for (int c = 0; c < channels; c++) f[(long)v * channels + c] = g[(long)c * r3 + cell];
        }
    }

    /// <summary>Row gather: <c>output[j] = input[indices[j]]</c> (rows of length channels) — used by the sparse-conv rulebook.</summary>
    unsafe void RowGather(Tensor output, Tensor input, Tensor indices, int m, int channels)
    {
        float* o = (float*)output.DataPointer, i = (float*)input.DataPointer; int* idx = (int*)indices.DataPointer;
        for (int j = 0; j < m; j++) for (int c = 0; c < channels; c++) o[(long)j * channels + c] = i[(long)idx[j] * channels + c];
    }

    /// <summary>Row scatter-add: <c>output[indices[j]] += input[j]</c> (indices unique per call), accumulating onto existing output.</summary>
    unsafe void RowScatterAdd(Tensor output, Tensor input, Tensor indices, int m, int channels)
    {
        float* o = (float*)output.DataPointer, i = (float*)input.DataPointer; int* idx = (int*)indices.DataPointer;
        for (int j = 0; j < m; j++) for (int c = 0; c < channels; c++) o[(long)idx[j] * channels + c] += i[(long)j * channels + c];
    }

    /// <summary>Depthwise 2D convolution (groups=C), weight <c>[C,1,kH,kW]</c> — used by YOLO11's class branch and C2PSA encoding.</summary>
    unsafe void Conv2dDepthwise(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException($"Conv2dDepthwise default fallback only supports F32 — got input={input.DType}, output={output.DType}, weight={weight.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"Conv2dDepthwise requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");
        if (weight.Shape.Rank != 4 || weight.Shape[1] != 1)
            throw new ArgumentException($"Conv2dDepthwise weight must be [C, 1, kH, kW]; got {weight.Shape}.");
        if (input.Shape[1] != weight.Shape[0] || output.Shape[1] != weight.Shape[0])
            throw new ArgumentException("Conv2dDepthwise requires input/output channel count to equal weight channel count.");

        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;
        float* wBase = (float*)weight.DataPointer;
        float* bBase = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                float* srcPlane = srcBase + ((long)b * c + ch) * iH * iW;
                float* dstPlane = dstBase + ((long)b * c + ch) * oH * oW;
                float* kernel = wBase + (long)ch * kH * kW;
                float biasVal = bBase is null ? 0f : bBase[ch];

                for (int oy = 0; oy < oH; oy++)
                {
                    int iy0 = oy * strideH - padH;
                    for (int ox = 0; ox < oW; ox++)
                    {
                        int ix0 = ox * strideW - padW;
                        float v = biasVal;
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int iy = iy0 + ky;
                            if (iy < 0 || iy >= iH) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int ix = ix0 + kx;
                                if (ix < 0 || ix >= iW) continue;
                                v += kernel[ky * kW + kx] * srcPlane[iy * iW + ix];
                            }
                        }
                        dstPlane[oy * oW + ox] = v;
                    }
                }
            }
        }
    }

    /// <summary>2D max-pooling with explicit kernel/stride/zero-padding — used by YOLO's SPPF block (k=5,s=1,p=2 preserves dims).</summary>
    unsafe void MaxPool2D(Tensor output, Tensor input, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException($"MaxPool2D default fallback only supports F32 — got input={input.DType}, output={output.DType}. Override in the backend if you need other dtypes.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"MaxPool2D requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");

        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;

        // NCHW: outer indices [n, c] address a contiguous H*W plane.
        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                long planeOffset = ((long)b * c + ch) * iH * iW;
                long outPlaneOffset = ((long)b * c + ch) * oH * oW;
                float* srcPlane = srcBase + planeOffset;
                float* dstPlane = dstBase + outPlaneOffset;

                for (int oy = 0; oy < oH; oy++)
                {
                    int iy0 = oy * strideH - padH;
                    for (int ox = 0; ox < oW; ox++)
                    {
                        int ix0 = ox * strideW - padW;
                        float maxVal = float.NegativeInfinity;
                        bool hasValue = false;
                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            int iy = iy0 + ky;
                            if (iy < 0 || iy >= iH) continue;
                            for (int kx = 0; kx < kernelW; kx++)
                            {
                                int ix = ix0 + kx;
                                if (ix < 0 || ix >= iW) continue;
                                hasValue = true;
                                float v = srcPlane[iy * iW + ix];
                                if (v > maxVal) maxVal = v;
                            }
                        }
                        // An empty padded window emits zero. A valid -infinity input remains -infinity.
                        dstPlane[oy * oW + ox] = hasValue ? maxVal : 0f;
                    }
                }
            }
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

    // ── Data Movement ───────────────────────────────────────────────────

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

    /// <summary>Count of cross-device copies that took a DIRECT peer path (P2P/NVLink), for placement diagnostics;
    /// 0 on backends without one. Host-staged fallbacks show up in <see cref="GetD2hSyncCount"/> instead.</summary>
    long GetPeerCopyCount() => 0;

    /// <summary>Fill a tensor with a constant value.</summary>
    void Fill(Tensor tensor, float value);

    // ── Audio (optional — backends may throw NotSupportedException) ──

    /// <summary>Radix-2 FFT for audio processing.</summary>
    void Fft(Tensor output, Tensor input);

    /// <summary>Short-time Fourier transform.</summary>
    void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window);

    /// <summary>Apply mel filterbank to FFT magnitude spectrogram.</summary>
    void MelFilterbank(Tensor output, Tensor input, Tensor filters);

    // ── Fused Operations ────────────────────────────────────────────────

    /// <summary>Fused GroupNorm + SiLU: normalize, apply affine, then SiLU in one pass, eliminating an intermediate allocation.</summary>
    void GroupNormSilu(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        GroupNorm(output, input, weight, bias, groups, eps);
        Silu(output, output);
    }

    // ── Dtype Casting ────────────────────────────────────────────────────

    /// <summary>Cast tensor from FP32 to FP16. Default: CPU scalar loop.</summary>
    unsafe void CastToF16(Tensor output, Tensor input)
    {
        float* src = (float*)input.DataPointer;
        Half* dst = (Half*)output.DataPointer;
        int count = (int)input.Shape.ElementCount;
        for (int i = 0; i < count; i++)
            dst[i] = (Half)src[i];
    }

    /// <summary>Cast tensor from FP16 to FP32. Default: CPU scalar loop.</summary>
    unsafe void CastToF32(Tensor output, Tensor input)
    {
        Half* src = (Half*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        int count = (int)input.Shape.ElementCount;
        for (int i = 0; i < count; i++)
            dst[i] = (float)src[i];
    }

    /// <summary>Cast tensor from FP32 to BF16. Default: CPU fallback via Tensor.CastTo.</summary>
    void CastToBf16(Tensor output, Tensor input)
    {
        Tensor casted = input.CastTo(DType.BF16);
        try
        {
            unsafe
            {
                Buffer.MemoryCopy((void*)casted.DataPointer, (void*)output.DataPointer,
                    output.ElementCount * 2, casted.ElementCount * 2);
            }
        }
        finally { casted.Dispose(); }
    }

    /// <summary>Cast tensor from FP8 E4M3 to FP16. Default: CPU via F32 intermediate.</summary>
    void CastF8E4M3ToF16(Tensor output, Tensor input)
    {
        Tensor f32 = input.CastTo(DType.F32);
        CastToF16(output, f32);
        f32.Dispose();
    }

    /// <summary>Cast tensor from FP16 to FP8 E4M3. Default: CPU via Tensor.CastTo.</summary>
    void CastF16ToF8E4M3(Tensor output, Tensor input)
    {
        Tensor f8 = input.CastTo(DType.F8E4M3);
        unsafe
        {
            long byteCount = output.Shape.ElementCount; // 1 byte per F8 element
            Buffer.MemoryCopy(f8.DataPointer, output.DataPointer, byteCount, byteCount);
        }
        f8.Dispose();
    }

    // ── Synchronization ──────────────────────────────────────────────────

    /// <summary>Waits for all pending GPU work (no-op on CPU) — call at phase boundaries so deferred frees land before large allocations.</summary>
    void Sync() { }

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

    /// <summary>Free device memory in bytes (0 if not a device backend). For diagnostics / adaptive tiling.</summary>
    long FreeMemoryBytes() => 0;

    /// <summary>Pre-uploads weights into the weight cache so ops avoid re-uploading; pair with <see cref="FreeWeights"/>.</summary>
    void PreloadWeights(IEnumerable<Tensor> weights) { }

    /// <summary>Streaming cache overlapping weight uploads with compute; <c>null</c> means fall back to <see cref="PreloadWeights"/>.</summary>
    IStreamingWeightCache? StreamingCache => null;

    /// <summary>Row-indexed affine broadcast writing e4m3 straight into <paramref name="outputFp8"/>, scaled for
    /// <paramref name="consumerWeight"/>'s Linear. Returns false if the backend cannot fuse, leaving the caller to
    /// run the ordinary path.</summary>
    /// <remarks>The modulated activation's only consumer is the fp8 Linear that follows, so materializing it in F32
    /// costs a full-width write and read to hand the quantize kernel something it compresses 4:1 anyway. Requires a
    /// static <c>Fp8InputScaleFactor</c> on the weight: with a dynamic absmax the scale is not known until a
    /// reduction over the finished tensor has retired, so it cannot be applied from registers.</remarks>
    bool TryAffineBroadcastRowIndexedToFp8(Tensor outputFp8, Tensor input, Tensor scaleTable, Tensor? shiftTable,
        Tensor rowIndex, Tensor consumerWeight) => false;

    /// <summary>Clears the per-op profile accumulator so a later <see cref="DumpOpProfile"/> covers only work after this point.</summary>
    /// <remarks>Lets a pipeline window the profiler onto its steady-state loop. Without it the table folds in
    /// one-time setup — text encode, weight residency warm-up — whose cost varies enough run to run that
    /// differencing two runs to cancel it does not work (measured on MiniMax-H3: a 3-step run spent MORE total
    /// time in <c>Linear</c> than a 6-step run, purely from differing H2D-miss counts). No-op when profiling is off.</remarks>
    void ResetOpProfile() { }

    /// <summary>Writes the per-op profile accumulated so far, tagging the output with <paramref name="label"/>. No-op when profiling is off.</summary>
    void DumpOpProfile(string label) { }
}
