using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Backend calls over host F32 arrays for the reference transformer modules: each wraps the array in a tensor, runs one
/// <see cref="IBackend"/> op and copies the result back.</summary>
internal static class HostOps
{
    /// <summary><c>y = x @ weight^T</c> through <see cref="IBackend.Linear"/>.</summary>
    public static float[] Project(IBackend backend, ReadOnlySpan<float> x, int rows, int inDim, Tensor weight, int outDim)
    {
        if (x.Length != (long)rows * inDim) throw new ArgumentException($"x must hold {rows}x{inDim} values.", nameof(x));
        using Tensor input = HostMath.Tensor(x, rows, inDim);
        using Tensor output = new(new TensorShape(rows, outDim), DType.F32);
        backend.Linear(output, input, weight, null);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The grouped projection <c>y[t, g, :] = weight[g] @ x[t, g, :]</c> through <see cref="IBackend.BlockDiagonalLinear"/>;
    /// <paramref name="weight"/> is <c>[groups * rank, groupDim]</c>, F32 or F8E4M3 with <paramref name="weightScaleInv"/>.</summary>
    public static float[] BlockDiagonal(IBackend backend, ReadOnlySpan<float> x, int rows, int groups, Tensor weight, Tensor? weightScaleInv)
    {
        using Tensor input = HostMath.Tensor(x, rows, x.Length / rows);
        using Tensor output = new(new TensorShape(rows, weight.Shape[0]), DType.F32);
        backend.BlockDiagonalLinear(output, input, weight, weightScaleInv, groups);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>RMS norm of each <paramref name="width"/>-wide row in place.</summary>
    public static void RmsNorm(IBackend backend, Span<float> data, int width, Tensor weight, float eps)
    {
        using Tensor input = HostMath.Tensor(data, data.Length / width, width);
        using Tensor output = new(input.Shape, DType.F32);
        backend.RmsNorm(output, input, weight, eps);
        output.AsReadOnlySpan<float>().CopyTo(data);
    }

    /// <summary>LayerNorm of each <paramref name="width"/>-wide row in place.</summary>
    public static void LayerNorm(IBackend backend, Span<float> data, int width, Tensor weight, Tensor bias, float eps)
    {
        using Tensor input = HostMath.Tensor(data, data.Length / width, width);
        using Tensor output = new(input.Shape, DType.F32);
        backend.LayerNorm(output, input, weight, bias, eps);
        output.AsReadOnlySpan<float>().CopyTo(data);
    }

    /// <summary>Orthonormal Walsh-Hadamard transform of each <paramref name="width"/>-wide row in place.</summary>
    public static void Hadamard(IBackend backend, Span<float> data, int width)
    {
        using Tensor x = HostMath.Tensor(data, data.Length / width, width);
        backend.Hadamard(x, x, 1f / MathF.Sqrt(width));
        x.AsReadOnlySpan<float>().CopyTo(data);
    }

    /// <summary>Replaces each row with its quantize-dequantize round trip in <paramref name="encoding"/>; F32 is a no-op.</summary>
    public static void RoundTrip(IBackend backend, Span<float> data, int width, LatentEncoding encoding)
    {
        if (encoding == LatentEncoding.F32) return;
        using Tensor x = HostMath.Tensor(data, data.Length / width, width);
        backend.ActQuantDequantInPlace(x, encoding);
        x.AsReadOnlySpan<float>().CopyTo(data);
    }

    /// <summary>Positions <c>start .. start + count - 1</c>.</summary>
    public static int[] Range(int start, int count)
    {
        int[] r = new int[count];
        for (int i = 0; i < count; i++) r[i] = start + i;
        return r;
    }
}
