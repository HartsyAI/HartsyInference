using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Looped;

/// <summary>The parameter-free normalized-GPT primitives, composed from shared backend ops: the unit-norm <c>x / sqrt(sum(x²) + eps)</c>
/// over the last dimension, and the residual interpolation <c>norm(norm(a) + alpha ⊙ (norm(b) − norm(a)))</c>.</summary>
/// <remarks>The unit-norm is <see cref="IBackend.RmsNorm"/> with a constant <c>1/sqrt(n)</c> weight and <c>eps/n</c>:
/// <c>x / sqrt(mean(x²) + eps/n) · 1/sqrt(n) = x / sqrt(sum(x²) + eps)</c> exactly, so every backend's fused norm kernel serves it.</remarks>
internal static class HypersphereOps
{
    /// <summary>Unit-norms every row of <paramref name="input"/> over its last dimension into <paramref name="output"/>;
    /// <paramref name="normWeight"/> is <see cref="InverseSqrtVector"/> of that dimension's width.</summary>
    public static void Normalize(IBackend backend, Tensor output, Tensor input, Tensor normWeight, float eps)
        => backend.RmsNorm(output, input, normWeight, eps / normWeight.ElementCount);

    /// <summary>Unit-norms <paramref name="input"/> into a new tensor of the same shape; consumes nothing.</summary>
    public static Tensor Normalized(IBackend backend, Tensor input, Tensor normWeight, float eps)
    {
        Tensor output = new(input.Shape, DType.F32);
        Normalize(backend, output, input, normWeight, eps);
        return output;
    }

    /// <summary>Joins <paramref name="sublayer"/> to <paramref name="residual"/> on the hypersphere:
    /// <c>norm((1 − alpha) ⊙ norm(residual) + alpha ⊙ norm(sublayer))</c>, with the two learned per-channel vectors precomputed.
    /// Both inputs are <c>[1, T, width]</c>; consumes neither.</summary>
    public static Tensor Interpolate(IBackend backend, Tensor residual, Tensor sublayer, Tensor alpha, Tensor oneMinusAlpha,
        Tensor normWeight, float eps)
    {
        using Tensor residualNormed = Normalized(backend, residual, normWeight, eps);
        using Tensor sublayerNormed = Normalized(backend, sublayer, normWeight, eps);
        using Tensor scaled = new(residual.Shape, DType.F32);
        backend.AffineBroadcastLastDim(scaled, residualNormed, oneMinusAlpha, null);
        using Tensor mixed = new(residual.Shape, DType.F32);
        backend.GatedResidualLastDim(mixed, scaled, sublayerNormed, alpha);
        return Normalized(backend, mixed, normWeight, eps);
    }

    /// <summary>An F32 vector of <paramref name="n"/> copies of <c>1/sqrt(n)</c>: the weight that turns <see cref="IBackend.RmsNorm"/> into the unit-norm.</summary>
    public static unsafe Tensor InverseSqrtVector(int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), n, "Width must be positive.");
        Tensor t = new(new TensorShape(n), DType.F32);
        t.AsSpan<float>().Fill(1f / MathF.Sqrt(n));
        return t;
    }
}
