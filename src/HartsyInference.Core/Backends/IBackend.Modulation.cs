using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// DiT modulation and gating primitives (affine broadcast, gated residual, modulated norms, chunked activations) with host reference defaults.
public partial interface IBackend
{
    // These keep the per-block modulation math GPU-resident in diffusion transformers
    // (Ideogram 4, etc.). Default implementations are scalar-F32 CPU loops that also serve
    // as the numerical reference; CudaBackend overrides them with PTX kernels.

    /// <summary>Broadcast affine over the last dim: <c>out = in*scale + (shift ?? 0)</c>; null shift = Ideogram 4's scale-only adaLN.</summary>
    unsafe void AffineBroadcastLastDim(Tensor output, Tensor input, Tensor scale, Tensor? shift)
        => AffineBroadcastLastDimReference(output, input, scale, shift);

    /// <summary>The managed <see cref="AffineBroadcastLastDim"/> body, callable directly. A backend override must call
    /// THIS to fall back — <c>((IBackend)this).AffineBroadcastLastDim(...)</c> re-enters the override through
    /// interface dispatch and recurses until the stack overflows.</summary>
    static unsafe void AffineBroadcastLastDimReference(Tensor output, Tensor input, Tensor scale, Tensor? shift)
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
        => GatedResidualLastDimReference(output, residual, value, gate);

    /// <summary>The managed <see cref="GatedResidualLastDim"/> body, callable directly. A backend override must call
    /// THIS to fall back — <c>((IBackend)this).GatedResidualLastDim(...)</c> re-enters the override through interface
    /// dispatch and recurses until the stack overflows.</summary>
    static unsafe void GatedResidualLastDimReference(Tensor output, Tensor residual, Tensor value, Tensor gate)
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
        => AffineBroadcastRowIndexedReference(output, input, scaleTable, shiftTable, rowIndex);

    /// <summary>The managed <see cref="AffineBroadcastRowIndexed"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).AffineBroadcastRowIndexed(...)</c> re-enters the override through
    /// interface dispatch and recurses until the stack overflows.</summary>
    static unsafe void AffineBroadcastRowIndexedReference(Tensor output, Tensor input, Tensor scaleTable, Tensor? shiftTable, Tensor rowIndex)
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
        => GatedResidualRowIndexedReference(output, residual, value, gateTable, rowIndex);

    /// <summary>The managed <see cref="GatedResidualRowIndexed"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).GatedResidualRowIndexed(...)</c> re-enters the override through
    /// interface dispatch and recurses until the stack overflows.</summary>
    static unsafe void GatedResidualRowIndexedReference(Tensor output, Tensor residual, Tensor value, Tensor gateTable, Tensor rowIndex)
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
        => LayerNormModulateReference(output, input, scale, shift, eps);

    /// <summary>The managed <see cref="LayerNormModulate"/> body, callable directly. A backend override must call
    /// THIS to fall back — <c>((IBackend)this).LayerNormModulate(...)</c> re-enters the override through interface
    /// dispatch and recurses until the stack overflows.</summary>
    static unsafe void LayerNormModulateReference(Tensor output, Tensor input, Tensor scale, Tensor shift, float eps)
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
        => QkvSplitNormReference(q, k, v, qkv, qWeight, kWeight, eps);

    /// <summary>The managed <see cref="QkvSplitNorm"/> body, callable directly. A backend override must call THIS
    /// to fall back — <c>((IBackend)this).QkvSplitNorm(...)</c> re-enters the override through interface dispatch
    /// and recurses until the stack overflows.</summary>
    static unsafe void QkvSplitNormReference(Tensor q, Tensor k, Tensor v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
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
        => QkvSplitNormHeadMajorReference(q, k, v, qkv, qWeight, kWeight, eps);

    /// <summary>The managed <see cref="QkvSplitNormHeadMajor"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).QkvSplitNormHeadMajor(...)</c> re-enters the override through
    /// interface dispatch and recurses until the stack overflows.</summary>
    static unsafe void QkvSplitNormHeadMajorReference(Tensor? q, Tensor? k, Tensor? v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
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

    /// <summary>AdaLN modulation split: chunks <c>proj [B,4D]</c> into 4 <c>[B,D]</c>, <c>1+x</c> for scales, <c>tanh(x)</c> for gates.</summary>
    unsafe void ModulationSplit4(Tensor scaleMsa, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp, Tensor proj)
        => ModulationSplit4Reference(scaleMsa, gateMsa, scaleMlp, gateMlp, proj);

    /// <summary>The managed <see cref="ModulationSplit4"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).ModulationSplit4(...)</c> re-enters the override through
    /// interface dispatch and recurses until the stack overflows.</summary>
    static unsafe void ModulationSplit4Reference(Tensor scaleMsa, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp, Tensor proj)
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

    /// <summary>Elementwise affine mix: <c>output = xScale·x + yScale·y</c>. Inputs may alias each other, but neither input storage may overlap <paramref name="output"/>. All tensors are F32, nonempty, and identically shaped.</summary>
    unsafe void AffineMix(Tensor output, Tensor x, Tensor y, float xScale, float yScale)
        => AffineMixReference(output, x, y, xScale, yScale);

    /// <summary>The managed <see cref="AffineMix"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).AffineMix(...)</c> re-enters the override and
    /// recurses until the stack overflows.</summary>
    static unsafe void AffineMixReference(Tensor output, Tensor x, Tensor y, float xScale, float yScale)
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

    /// <summary>In-place masked affine replacement: <c>target = mask·target + (1−mask)·(sourceScale·source + noiseScale·noise)</c>. When <paramref name="noiseScale"/> is zero, <paramref name="noise"/> must be null and the noise term is omitted. Mask values are consumed exactly as supplied (never clamped). Source, noise, and mask remain unchanged.</summary>
    /// <remarks>Packed layouts accept <c>[B,S,F]</c>/<c>[B,S,P]</c> or implicit-batch <c>[S,F]</c>/<c>[S,P]</c>.</remarks>
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
                long patchIndex = layout is MaskBroadcastLayout.PackedChannelOuter or MaskBroadcastLayout.Rows
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

    /// <summary>Row-indexed affine broadcast writing e4m3 straight into <paramref name="outputFp8"/>, scaled for
    /// <paramref name="consumerWeight"/>'s Linear. Returns false if the backend cannot fuse, leaving the caller to
    /// run the ordinary path.</summary>
    /// <remarks>The modulated activation's only consumer is the fp8 Linear that follows, so materializing it in F32
    /// costs a full-width write and read to hand the quantize kernel something it compresses 4:1 anyway. Requires a
    /// static <c>Fp8InputScaleFactor</c> on the weight: with a dynamic absmax the scale is not known until a
    /// reduction over the finished tensor has retired, so it cannot be applied from registers.</remarks>
    bool TryAffineBroadcastRowIndexedToFp8(Tensor outputFp8, Tensor input, Tensor scaleTable, Tensor? shiftTable,
        Tensor rowIndex, Tensor consumerWeight) => false;
}
