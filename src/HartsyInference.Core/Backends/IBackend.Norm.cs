using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Normalization: plain, affine, channel and fused residual/norm forms, with host reference defaults where a backend declares none.
public partial interface IBackend
{
    /// <summary>Splits channels into <paramref name="groups"/> groups, normalizes each over channel+spatial, then applies the per-channel affine weight/bias.</summary>
    void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps);

    /// <summary>Normalizes over the last dim (zero mean, unit variance) then applies the affine weight/bias.</summary>
    void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps);

    /// <summary>RMS-normalizes over the last dim (no mean subtraction) then scales by weight: <c>out = x / sqrt(mean(x²)+eps) · weight</c>.</summary>
    void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps);

    /// <summary>AdaIN 1D: per-(batch,channel) normalize <c>[B,C,T]</c> over T, scale by <c>(1+gamma)</c>, shift by beta.</summary>
    /// <param name="gamma"><c>[B, C]</c> (or <c>[C]</c>, broadcast across batch), typically a Linear projection of a style/speaker embedding.</param>
    /// <param name="beta"><c>[B, C]</c> (or <c>[C]</c>, broadcast across batch), typically a Linear projection of a style/speaker embedding.</param>
    void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps);

    /// <summary>Wan2.2 VAE channel RMS norm: <c>scale = sqrt(C)/max(L2_over_C, eps)</c>, <c>out = x·scale·gamma</c>.</summary>
    unsafe void WanRmsNormChannel(Tensor output, Tensor input, Tensor? gamma, float eps)
        => WanRmsNormChannelReference(output, input, gamma, eps);

    /// <summary>The managed <see cref="WanRmsNormChannel"/> body, callable directly. A backend override must call
    /// THIS to fall back — <c>((IBackend)this).WanRmsNormChannel(...)</c> re-enters the override through interface
    /// dispatch and recurses until the stack overflows.</summary>
    static unsafe void WanRmsNormChannelReference(Tensor output, Tensor input, Tensor? gamma, float eps)
    {
        // Stated rather than assumed: the loop below reads every operand as float*, so a non-F32 tensor would be
        // reinterpreted bit-for-bit and produce plausible garbage. The other references in this file guard the
        // same way.
        if (output.DType != DType.F32 || input.DType != DType.F32 || (gamma is not null && gamma.DType != DType.F32))
            throw new NotSupportedException("WanRmsNormChannel default fallback only supports F32.");
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

    /// <summary>Non-affine LayerNorm over the last dim: per row, zero mean and unit variance (biased var), no learned scale/bias.</summary>
    unsafe void LayerNormNoAffine(Tensor output, Tensor input, float eps)
        => LayerNormNoAffineReference(output, input, eps);

    /// <summary>The managed <see cref="LayerNormNoAffine"/> body, callable directly. A backend override must call
    /// THIS to fall back — <c>((IBackend)this).LayerNormNoAffine(...)</c> re-enters the override through interface
    /// dispatch and recurses until the stack overflows.</summary>
    static unsafe void LayerNormNoAffineReference(Tensor output, Tensor input, float eps)
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

    /// <summary>Fused GroupNorm + SiLU: normalize, apply affine, then SiLU in one pass, eliminating an intermediate allocation.</summary>
    void GroupNormSilu(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        GroupNorm(output, input, weight, bias, groups, eps);
        Silu(output, output);
    }
}
