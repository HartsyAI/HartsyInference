using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>Elementwise and layout helpers of the HairDeck runtime. Matrix work goes through <see cref="IBackend"/>; these
/// helpers only move and combine the activations of one sample (batch 1), in float32 like the reference.</summary>
internal static unsafe class HairDeckNumerics
{
    /// <summary>Width of the sinusoidal timestep embedding that feeds the timestep MLP.</summary>
    public const int TimestepWidth = 256;

    public static Tensor Create(long rows, long cols) => new(new TensorShape((int)rows, (int)cols), DType.F32);

    public static Tensor Create(long d0, long d1, long d2, long d3) =>
        new(new TensorShape((int)d0, (int)d1, (int)d2, (int)d3), DType.F32);

    public static Tensor FromArray(ReadOnlySpan<float> values, long rows, long cols)
    {
        if (values.Length != rows * cols)
            throw new ArgumentException($"Expected {rows * cols} values for [{rows}, {cols}], got {values.Length}.", nameof(values));
        Tensor t = Create(rows, cols);
        values.CopyTo(new Span<float>(t.DataPointer, values.Length));
        return t;
    }

    public static float[] ToArray(Tensor t)
    {
        float[] result = new float[t.ElementCount];
        new ReadOnlySpan<float>(t.DataPointer, result.Length).CopyTo(result);
        return result;
    }

    public static Tensor Clone(Tensor t)
    {
        Tensor copy = new(t.Shape, DType.F32);
        new ReadOnlySpan<float>(t.DataPointer, checked((int)t.ElementCount)).CopyTo(new Span<float>(copy.DataPointer, checked((int)t.ElementCount)));
        return copy;
    }

    /// <summary>Row <paramref name="index"/> of width <paramref name="dim"/> of a <c>[1, k * dim]</c> projection, as <c>[1, dim]</c>.</summary>
    public static Tensor Chunk(Tensor projection, int index, int dim)
    {
        Tensor dst = Create(1, dim);
        float* src = (float*)projection.DataPointer + (long)index * dim;
        float* d = (float*)dst.DataPointer;
        for (int i = 0; i < dim; i++) d[i] = src[i];
        return dst;
    }

    /// <summary><c>[S, srcCols]</c> token-major rows, columns from <paramref name="colOffset"/> on, to head-major
    /// <c>[1, heads, S, headDim]</c>: head <c>h</c> reads columns <c>colOffset + h * headDim + d</c>.</summary>
    public static void SplitHeads(Tensor src, int colOffset, Tensor dst, int heads, int headDim)
    {
        int s = (int)src.Shape[0], srcCols = (int)src.Shape[1];
        float* sp = (float*)src.DataPointer, dp = (float*)dst.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < s; i++)
            {
                float* from = sp + (long)i * srcCols + colOffset + (long)h * headDim;
                float* to = dp + ((long)h * s + i) * headDim;
                for (int d = 0; d < headDim; d++) to[d] = from[d];
            }
    }

    /// <summary>Head-major <c>[1, heads, S, headDim]</c> back to token-major <c>[S, heads * headDim]</c>.</summary>
    public static void MergeHeads(Tensor src, Tensor dst, int heads, int headDim)
    {
        int s = (int)dst.Shape[0], width = (int)dst.Shape[1];
        float* sp = (float*)src.DataPointer, dp = (float*)dst.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < s; i++)
            {
                float* from = sp + ((long)h * s + i) * headDim;
                float* to = dp + (long)i * width + (long)h * headDim;
                for (int d = 0; d < headDim; d++) to[d] = from[d];
            }
    }

    /// <summary><c>residual + gate * value</c> with <paramref name="gate"/> <c>[1, D]</c> broadcast over the rows of <c>[N, D]</c>.</summary>
    public static Tensor GatedAdd(Tensor residual, Tensor value, Tensor gate)
    {
        int n = (int)residual.Shape[0], d = (int)residual.Shape[1];
        Tensor dst = Create(n, d);
        float* r = (float*)residual.DataPointer, v = (float*)value.DataPointer, g = (float*)gate.DataPointer, o = (float*)dst.DataPointer;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < d; k++)
            {
                long at = (long)i * d + k;
                o[at] = r[at] + g[k] * v[at];
            }
        return dst;
    }

    /// <summary>Adds the <c>[1, D]</c> row to every row of <c>[N, D]</c> in place.</summary>
    public static void AddRowInPlace(Tensor matrix, Tensor row)
    {
        int n = (int)matrix.Shape[0], d = (int)matrix.Shape[1];
        float* m = (float*)matrix.DataPointer, r = (float*)row.DataPointer;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < d; k++) m[(long)i * d + k] += r[k];
    }

    /// <summary>Mean over the rows of <c>[M, D]</c> as <c>[1, D]</c>, accumulated in double like the reference's mean.</summary>
    public static Tensor MeanRows(Tensor matrix)
    {
        int m = (int)matrix.Shape[0], d = (int)matrix.Shape[1];
        Tensor dst = Create(1, d);
        float* p = (float*)matrix.DataPointer, o = (float*)dst.DataPointer;
        for (int k = 0; k < d; k++)
        {
            double sum = 0;
            for (int i = 0; i < m; i++) sum += p[(long)i * d + k];
            o[k] = (float)(sum / m);
        }
        return dst;
    }

    /// <summary>The sinusoidal embedding of one timestep as <c>[1, 256]</c>: cos then sin of <c>t * 1000 * exp(-ln(1e4) i / 128)</c>.</summary>
    public static Tensor Timestep(float t)
    {
        const int half = TimestepWidth / 2;
        Tensor dst = Create(1, TimestepWidth);
        float* o = (float*)dst.DataPointer;
        float logPeriod = MathF.Log(10000f);
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-logPeriod * i / half);
            float a = t * 1000f * freq;
            o[i] = MathF.Cos(a);
            o[half + i] = MathF.Sin(a);
        }
        return dst;
    }

    /// <summary><c>input · weightᵀ + bias</c> for <c>[N, K]</c> input as a new <c>[N, out]</c> tensor.</summary>
    public static Tensor Project(IBackend backend, Tensor input, HairDeckSetDitCheckpointConverter.Linear layer)
    {
        Tensor output = Create(input.Shape[0], layer.Weight.Shape[0]);
        backend.Linear(output, input, layer.Weight, layer.Bias);
        return output;
    }

    public static Tensor Silu(IBackend backend, Tensor input)
    {
        Tensor output = new(input.Shape, DType.F32);
        backend.Silu(output, input);
        return output;
    }

    /// <summary>GELU with the tanh approximation (the set DiT MLP).</summary>
    public static Tensor GeluTanh(IBackend backend, Tensor input)
    {
        Tensor output = new(input.Shape, DType.F32);
        backend.Gelu(output, input);
        return output;
    }

    /// <summary>Exact erf GELU (the card VAE decoder MLP).</summary>
    public static Tensor GeluExact(IBackend backend, Tensor input)
    {
        Tensor output = new(input.Shape, DType.F32);
        backend.GeluErf(output, input);
        return output;
    }

    /// <summary>adaLN: <c>LayerNorm(x) · (1 + scale) + shift</c> with a parameter-free LayerNorm; <paramref name="scale"/> and
    /// <paramref name="shift"/> are <c>[1, D]</c>.</summary>
    public static Tensor NormModulate(IBackend backend, Tensor input, Tensor scale, Tensor shift, float eps)
    {
        Tensor output = new(input.Shape, DType.F32);
        backend.LayerNormModulate(output, input, scale, shift, eps);
        return output;
    }

    /// <summary>LayerNorm with affine weight and bias over the last dimension.</summary>
    public static Tensor LayerNorm(IBackend backend, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        Tensor output = new(input.Shape, DType.F32);
        backend.LayerNorm(output, input, weight, bias, eps);
        return output;
    }

    /// <summary>Chunk <paramref name="index"/> of a split projection, as an owned <c>[1, dim]</c> tensor.</summary>
    public static Tensor[] Chunks(Tensor projection, int count, int dim)
    {
        Tensor[] result = new Tensor[count];
        for (int i = 0; i < count; i++) result[i] = Chunk(projection, i, dim);
        return result;
    }

    /// <summary>The <paramref name="i"/>-th of <paramref name="points"/> values of torch.linspace(start, end, points) in float32.</summary>
    public static float Linspace(int i, int points, float start, float end)
    {
        float step = (end - start) / (points - 1);
        return i < points / 2 ? start + step * i : end - step * (points - i - 1);
    }
}
