using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Small F32 host kernels and tensor marshalling shared by the V4.1 reference modules.</summary>
internal static class DeepSeekV41HostMath
{
    /// <summary><c>y[t,o] = sum_i x[t,i] * w[o,i]</c> for <paramref name="rows"/> rows.</summary>
    public static float[] Linear(ReadOnlySpan<float> x, ReadOnlySpan<float> w, int rows, int inDim, int outDim)
    {
        if (x.Length != (long)rows * inDim || w.Length != (long)outDim * inDim) throw new ArgumentException("Linear operands do not match the stated shape.");
        float[] y = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> xr = x.Slice(r * inDim, inDim);
            for (int o = 0; o < outDim; o++)
            {
                ReadOnlySpan<float> wr = w.Slice(o * inDim, inDim);
                float sum = 0f;
                for (int i = 0; i < inDim; i++) sum += xr[i] * wr[i];
                y[r * outDim + o] = sum;
            }
        }
        return y;
    }

    /// <summary>In-place RMS norm per row of width <paramref name="dim"/>: <c>w * x * rsqrt(mean(x^2) + eps)</c>.</summary>
    public static void RmsNormRows(Span<float> x, ReadOnlySpan<float> weight, int dim, float eps)
    {
        if (weight.Length != dim || x.Length % dim != 0) throw new ArgumentException("RmsNorm operands do not match the stated width.");
        for (int off = 0; off < x.Length; off += dim)
        {
            Span<float> row = x.Slice(off, dim);
            float sq = 0f;
            for (int i = 0; i < dim; i++) sq += row[i] * row[i];
            float scale = 1f / MathF.Sqrt(sq / dim + eps);
            for (int i = 0; i < dim; i++) row[i] = weight[i] * (row[i] * scale);
        }
    }

    /// <summary>An F32 tensor holding a copy of <paramref name="values"/> with the given shape.</summary>
    public static Tensor Tensor(ReadOnlySpan<float> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>An I32 tensor holding a copy of <paramref name="values"/> with the given shape.</summary>
    public static Tensor Tensor(ReadOnlySpan<int> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.I32);
        values.CopyTo(t.AsSpan<int>());
        return t;
    }
}
