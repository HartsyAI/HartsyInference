using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for the hyper-connection (mHC) mixing primitives of DeepSeek-V4.1.</summary>
public static class HcReference
{
    /// <summary>Widest hyper-connection stream count the fused kernel keeps per thread.</summary>
    public const int MaxHc = 8;

    /// <summary>Checks the operands of <see cref="SplitSinkhorn"/>; shared by every backend.</summary>
    public static void ValidateSplit(Tensor pre, Tensor post, Tensor comb, Tensor mixes, Tensor scale, Tensor bias, int hc,
        int iters)
    {
        if (hc < 1 || hc > MaxHc) throw new ArgumentOutOfRangeException(nameof(hc), $"hc must be in [1,{MaxHc}]; got {hc}.");
        if (iters < 1) throw new ArgumentOutOfRangeException(nameof(iters), "iters must be at least 1.");
        if (pre.DType != DType.F32 || post.DType != DType.F32 || comb.DType != DType.F32 || mixes.DType != DType.F32 ||
            scale.DType != DType.F32 || bias.DType != DType.F32)
            throw new NotSupportedException("HcSplitSinkhorn supports F32 only.");
        int width = (2 + hc) * hc;
        if (mixes.ElementCount == 0 || mixes.ElementCount % width != 0)
            throw new ArgumentException($"mixes ({mixes.ElementCount} elements) must be a non-empty multiple of {width}.", nameof(mixes));
        long tokens = mixes.ElementCount / width;
        if (pre.ElementCount != tokens * hc || post.ElementCount != tokens * hc || comb.ElementCount != tokens * hc * hc)
            throw new ArgumentException($"pre/post must hold {tokens}x{hc} and comb {tokens}x{hc}x{hc} entries.");
        if (scale.ElementCount != 3) throw new ArgumentException("scale must hold 3 entries.", nameof(scale));
        if (bias.ElementCount != width) throw new ArgumentException($"bias must hold {width} entries.", nameof(bias));
    }

    /// <summary>Splits each token's mixes into pre and post gates and a doubly-stochastic comb matrix.</summary>
    /// <remarks><c>pre = sigmoid(m0*s0+b) + eps</c>, <c>post = 2*sigmoid(m1*s1+b)</c>, <c>comb = softmax_row(m2*s2+b) + eps</c>
    /// followed by a column normalization and <paramref name="iters"/>-1 row-then-column rounds, each dividing by sum + eps.</remarks>
    public static unsafe void SplitSinkhorn(Tensor pre, Tensor post, Tensor comb, Tensor mixes, Tensor scale, Tensor bias,
        int hc, int iters, float eps)
    {
        ValidateSplit(pre, post, comb, mixes, scale, bias, hc, iters);
        int width = (2 + hc) * hc;
        long tokens = mixes.ElementCount / width;
        float* m = (float*)mixes.DataPointer, s = (float*)scale.DataPointer, b = (float*)bias.DataPointer;
        float* pr = (float*)pre.DataPointer, po = (float*)post.DataPointer, co = (float*)comb.DataPointer;
        Span<float> c = stackalloc float[hc * hc];
        for (long t = 0; t < tokens; t++)
        {
            float* x = m + t * width;
            for (int i = 0; i < hc; i++)
            {
                pr[t * hc + i] = Sigmoid(x[i] * s[0] + b[i]) + eps;
                po[t * hc + i] = 2f * Sigmoid(x[hc + i] * s[1] + b[hc + i]);
            }
            for (int i = 0; i < hc * hc; i++) c[i] = x[2 * hc + i] * s[2] + b[2 * hc + i];
            for (int r = 0; r < hc; r++)
            {
                float max = c[r * hc];
                for (int k = 1; k < hc; k++) max = MathF.Max(max, c[r * hc + k]);
                float sum = 0f;
                for (int k = 0; k < hc; k++)
                {
                    c[r * hc + k] = MathF.Exp(c[r * hc + k] - max);
                    sum += c[r * hc + k];
                }
                for (int k = 0; k < hc; k++) c[r * hc + k] = c[r * hc + k] / sum + eps;
            }
            NormalizeColumns(c, hc, eps);
            for (int it = 1; it < iters; it++)
            {
                NormalizeRows(c, hc, eps);
                NormalizeColumns(c, hc, eps);
            }
            for (int i = 0; i < hc * hc; i++) co[t * hc * hc + i] = c[i];
        }
    }

    /// <summary>Checks the operands of <see cref="PreMix"/>.</summary>
    public static void ValidatePreMix(Tensor output, Tensor x, Tensor pre)
    {
        if (output.DType != DType.F32 || x.DType != DType.F32 || pre.DType != DType.F32)
            throw new NotSupportedException("HcPreMix supports F32 only.");
        if (x.Shape.Rank != 3) throw new ArgumentException("x must be [tokens, hc, dim].", nameof(x));
        long tokens = x.Shape[0], hc = x.Shape[1], dim = x.Shape[2];
        if (hc > MaxHc || pre.ElementCount != tokens * hc) throw new ArgumentException($"pre must hold {tokens}x{hc} entries.", nameof(pre));
        if (output.ElementCount != tokens * dim) throw new ArgumentException($"output must hold {tokens}x{dim} entries.", nameof(output));
    }

    /// <summary>Collapses the hc streams: <c>output[t] = sum_i pre[t,i] * x[t,i]</c>, streams summed in order.</summary>
    public static unsafe void PreMix(Tensor output, Tensor x, Tensor pre)
    {
        ValidatePreMix(output, x, pre);
        int tokens = (int)x.Shape[0], hc = (int)x.Shape[1], dim = (int)x.Shape[2];
        float* xp = (float*)x.DataPointer, pp = (float*)pre.DataPointer, o = (float*)output.DataPointer;
        for (int t = 0; t < tokens; t++)
            for (int d = 0; d < dim; d++)
            {
                float acc = 0f;
                for (int i = 0; i < hc; i++) acc += pp[t * hc + i] * xp[((long)t * hc + i) * dim + d];
                o[(long)t * dim + d] = acc;
            }
    }

    /// <summary>Checks the operands of <see cref="PostMix"/>.</summary>
    public static void ValidatePostMix(Tensor output, Tensor x, Tensor residual, Tensor post, Tensor comb)
    {
        if (output.DType != DType.F32 || x.DType != DType.F32 || residual.DType != DType.F32 || post.DType != DType.F32 ||
            comb.DType != DType.F32)
            throw new NotSupportedException("HcPostMix supports F32 only.");
        if (residual.Shape.Rank != 3) throw new ArgumentException("residual must be [tokens, hc, dim].", nameof(residual));
        long tokens = residual.Shape[0], hc = residual.Shape[1], dim = residual.Shape[2];
        if (hc > MaxHc) throw new ArgumentException($"hc must not exceed {MaxHc}.", nameof(residual));
        if (x.ElementCount != tokens * dim) throw new ArgumentException($"x must hold {tokens}x{dim} entries.", nameof(x));
        if (post.ElementCount != tokens * hc) throw new ArgumentException($"post must hold {tokens}x{hc} entries.", nameof(post));
        if (comb.ElementCount != tokens * hc * hc) throw new ArgumentException($"comb must hold {tokens}x{hc}x{hc} entries.", nameof(comb));
        if (!output.Shape.Equals(residual.Shape)) throw new ArgumentException("output must match the residual shape.", nameof(output));
    }

    /// <summary>Expands and mixes: <c>output[t,i] = post[t,i]*x[t] + sum_j comb[t,j,i] * residual[t,j]</c>, j ascending.</summary>
    /// <remarks><paramref name="output"/> must not alias <paramref name="residual"/>.</remarks>
    public static unsafe void PostMix(Tensor output, Tensor x, Tensor residual, Tensor post, Tensor comb)
    {
        ValidatePostMix(output, x, residual, post, comb);
        int tokens = (int)residual.Shape[0], hc = (int)residual.Shape[1], dim = (int)residual.Shape[2];
        float* xp = (float*)x.DataPointer, rp = (float*)residual.DataPointer, pp = (float*)post.DataPointer;
        float* cp = (float*)comb.DataPointer, o = (float*)output.DataPointer;
        for (int t = 0; t < tokens; t++)
            for (int i = 0; i < hc; i++)
                for (int d = 0; d < dim; d++)
                {
                    float mix = 0f;
                    for (int j = 0; j < hc; j++) mix += cp[((long)t * hc + j) * hc + i] * rp[((long)t * hc + j) * dim + d];
                    o[((long)t * hc + i) * dim + d] = pp[t * hc + i] * xp[(long)t * dim + d] + mix;
                }
    }

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    private static void NormalizeRows(Span<float> c, int hc, float eps)
    {
        for (int r = 0; r < hc; r++)
        {
            float sum = 0f;
            for (int k = 0; k < hc; k++) sum += c[r * hc + k];
            for (int k = 0; k < hc; k++) c[r * hc + k] /= sum + eps;
        }
    }

    private static void NormalizeColumns(Span<float> c, int hc, float eps)
    {
        for (int k = 0; k < hc; k++)
        {
            float sum = 0f;
            for (int r = 0; r < hc; r++) sum += c[r * hc + k];
            for (int r = 0; r < hc; r++) c[r * hc + k] /= sum + eps;
        }
    }
}
