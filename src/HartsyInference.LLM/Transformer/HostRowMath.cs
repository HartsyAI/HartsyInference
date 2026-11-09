using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Small F32 host row kernels and tensor marshalling shared by the host reference attention paths.</summary>
internal static class HostRowMath
{
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
    public static Tensor F32Tensor(ReadOnlySpan<float> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
