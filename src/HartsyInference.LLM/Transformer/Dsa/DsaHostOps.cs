using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Backend calls over host F32 arrays for the DSA host reference: each wraps the array in a tensor, runs one
/// <see cref="IBackend"/> op and copies the result back.</summary>
internal static class DsaHostOps
{
    /// <summary><c>y = x @ weight^T</c> through <see cref="IBackend.Linear"/>.</summary>
    public static float[] Project(IBackend backend, ReadOnlySpan<float> x, int rows, int inDim, Tensor weight, int outDim)
    {
        if (x.Length != (long)rows * inDim) throw new ArgumentException($"x must hold {rows}x{inDim} values.", nameof(x));
        using Tensor input = F32(x, rows, inDim);
        using Tensor output = new(new TensorShape(rows, outDim), DType.F32);
        backend.Linear(output, input, weight, null);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>RMS norm of each <paramref name="width"/>-wide row in place.</summary>
    public static void RmsNorm(IBackend backend, Span<float> data, int width, Tensor weight, float eps)
    {
        using Tensor input = F32(data, data.Length / width, width);
        using Tensor output = new(input.Shape, DType.F32);
        backend.RmsNorm(output, input, weight, eps);
        output.AsReadOnlySpan<float>().CopyTo(data);
    }

    /// <summary>LayerNorm of each <paramref name="width"/>-wide row in place; a null bias is zero.</summary>
    public static void LayerNorm(IBackend backend, Span<float> data, int width, Tensor weight, Tensor? bias, float eps)
    {
        using Tensor input = F32(data, data.Length / width, width);
        using Tensor output = new(input.Shape, DType.F32);
        using Tensor? zeros = bias is null ? new Tensor(new TensorShape(width), DType.F32) : null;
        zeros?.AsSpan<float>().Clear();
        backend.LayerNorm(output, input, weight, bias ?? zeros!, eps);
        output.AsReadOnlySpan<float>().CopyTo(data);
    }

    /// <summary>Orthonormal Walsh-Hadamard transform of each <paramref name="width"/>-wide row in place.</summary>
    public static void Hadamard(IBackend backend, Span<float> data, int width)
    {
        using Tensor x = F32(data, data.Length / width, width);
        backend.Hadamard(x, x, 1f / MathF.Sqrt(width));
        x.AsReadOnlySpan<float>().CopyTo(data);
    }

    /// <summary>Replaces each row with its quantize-dequantize round trip in <paramref name="encoding"/>; F32 is a no-op.</summary>
    public static void RoundTrip(IBackend backend, Span<float> data, int width, LatentEncoding encoding)
    {
        if (encoding == LatentEncoding.F32) return;
        using Tensor x = F32(data, data.Length / width, width);
        backend.ActQuantDequantInPlace(x, encoding);
        x.AsReadOnlySpan<float>().CopyTo(data);
    }

    /// <summary>An F32 tensor holding a copy of <paramref name="values"/>.</summary>
    public static Tensor F32(ReadOnlySpan<float> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>An I32 tensor holding a copy of <paramref name="values"/>.</summary>
    public static Tensor I32(ReadOnlySpan<int> values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.I32);
        values.CopyTo(t.AsSpan<int>());
        return t;
    }

    /// <summary>Positions <c>start .. start + count - 1</c>.</summary>
    public static int[] Range(int start, int count)
    {
        int[] r = new int[count];
        for (int i = 0; i < count; i++) r[i] = start + i;
        return r;
    }
}
