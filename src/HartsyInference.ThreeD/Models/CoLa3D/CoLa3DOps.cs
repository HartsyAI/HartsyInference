using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Allocation, layout and layer helpers shared by the CoLa3D components. Device math goes through <c>IBackend</c>.
/// Layer inputs and outputs are <c>[1, tokens, width]</c>; the caller owns every tensor it allocates or receives.</summary>
internal static class CoLa3DOps
{
    /// <summary>Frequency channels of the sinusoidal slot and part-index embeddings (64 in the checkpoints).</summary>
    internal const int PartFrequencyDim = 64;

    /// <summary>Epsilon of the <c>LayerNorm32</c> layers in the transformer blocks.</summary>
    internal const float BlockNormEps = 1e-6f;

    internal static Tensor Alloc(params long[] dims) => new(new TensorShape(dims), DType.F32);

    internal static Affine LoadAffine(IReadOnlyDictionary<string, Tensor> w, string name) => new(w[name + ".weight"], w[name + ".bias"]);

    internal static Dense LoadDense(IReadOnlyDictionary<string, Tensor> w, string name) => new(w[name + ".weight"], w[name + ".bias"]);

    /// <summary>Layer norm over the last dim with affine weight and bias; the input is borrowed.</summary>
    internal static Tensor Norm(IBackend backend, Tensor x, Affine norm, float eps)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.LayerNorm(o, x, norm.Weight, norm.Bias, eps);
        return o;
    }

    /// <summary>Linear layer over <c>[1, S, in]</c> giving <c>[1, S, outDim]</c>; the input is borrowed.</summary>
    internal static Tensor Linear(IBackend backend, Tensor x, Dense layer, long outDim)
    {
        Tensor o = Alloc(1, x.Shape[1], outDim);
        backend.Linear(o, x, layer.Weight, layer.Bias);
        return o;
    }

    /// <summary>Returns <c>x + delta</c> and releases both inputs.</summary>
    internal static Tensor Residual(IBackend backend, Tensor x, Tensor delta)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.Add(o, x, delta);
        x.Dispose();
        delta.Dispose();
        return o;
    }

    /// <summary>Writes <c>sin</c> then <c>cos</c> of <c>value·freq_i</c> per value, the layout of <c>PartIndexEmbedder.sinusoidal_embedding</c>.</summary>
    internal static void SinusoidalEmbedding(ReadOnlySpan<float> values, int dim, Span<float> dst)
    {
        int half = dim / 2;
        for (int v = 0; v < values.Length; v++)
        {
            Span<float> row = dst.Slice(v * dim, dim);
            for (int i = 0; i < half; i++)
            {
                float angle = values[v] * MathF.Exp(-MathF.Log(10000f) * i / half);
                row[i] = MathF.Sin(angle);
                row[half + i] = MathF.Cos(angle);
            }
        }
    }

    /// <summary>Copies <paramref name="rowCount"/> rows from <paramref name="src"/> row <paramref name="srcRow"/> into <paramref name="dst"/> row <paramref name="dstRow"/>.</summary>
    internal static void CopyRows(Tensor src, int srcRow, int rowCount, Tensor dst, int dstRow)
    {
        int width = (int)src.Shape[src.Shape.Rank - 1];
        src.AsReadOnlySpan<float>().Slice(srcRow * width, rowCount * width)
            .CopyTo(dst.AsSpan<float>().Slice(dstRow * width, rowCount * width));
    }

    /// <summary>Copies rows <c>srcBaseRow + srcRows[i]</c> of <paramref name="src"/> into rows <c>dstBaseRow + i</c> of <paramref name="dst"/>.</summary>
    internal static void GatherRows(Tensor src, int srcBaseRow, ReadOnlySpan<int> srcRows, Tensor dst, int dstBaseRow)
    {
        int width = (int)src.Shape[src.Shape.Rank - 1];
        ReadOnlySpan<float> from = src.AsReadOnlySpan<float>();
        Span<float> to = dst.AsSpan<float>();
        for (int i = 0; i < srcRows.Length; i++)
            from.Slice((srcBaseRow + srcRows[i]) * width, width).CopyTo(to.Slice((dstBaseRow + i) * width, width));
    }

    /// <summary>Copies consecutive rows of <paramref name="src"/> into rows <c>dstBaseRow + dstRows[i]</c> of <paramref name="dst"/>; other rows are untouched.</summary>
    internal static void ScatterRows(Tensor src, int srcBaseRow, ReadOnlySpan<int> dstRows, Tensor dst, int dstBaseRow)
    {
        int width = (int)dst.Shape[dst.Shape.Rank - 1];
        ReadOnlySpan<float> from = src.AsReadOnlySpan<float>();
        Span<float> to = dst.AsSpan<float>();
        for (int i = 0; i < dstRows.Length; i++)
            from.Slice((srcBaseRow + i) * width, width).CopyTo(to.Slice((dstBaseRow + dstRows[i]) * width, width));
    }

    /// <summary>Row-wise <c>x / max(||x||, 1e-12)</c> (the <c>F.normalize</c> rule), in place.</summary>
    internal static void NormalizeRows(Span<float> data, int rows, int width)
    {
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = data.Slice(r * width, width);
            float sumSquares = 0f;
            for (int i = 0; i < width; i++)
                sumSquares += row[i] * row[i];
            float inv = 1f / MathF.Max(MathF.Sqrt(sumSquares), 1e-12f);
            for (int i = 0; i < width; i++)
                row[i] *= inv;
        }
    }

    internal readonly record struct Affine(Tensor Weight, Tensor Bias);

    internal readonly record struct Dense(Tensor Weight, Tensor Bias);
}
