using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Small allocate-and-run wrappers over <see cref="IBackend"/> ops for the UniRig modules; every result is owned by the caller.</summary>
internal static class UniRigOps
{
    public const float LayerNormEps = 1e-5f;

    /// <summary>Tensor of <paramref name="shape"/> filled from <paramref name="values"/>.</summary>
    public static Tensor FromFloats(ReadOnlySpan<float> values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    public static Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor? bias)
    {
        long[] dims = new long[input.Shape.Rank];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = input.Shape[i];
        }

        dims[^1] = weight.Shape[0];
        Tensor output = new(new TensorShape(dims), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    public static Tensor LayerNorm(IBackend backend, Tensor input, Tensor weight, Tensor bias)
    {
        Tensor output = new(input.Shape, DType.F32);
        backend.LayerNorm(output, input, weight, bias, LayerNormEps);
        return output;
    }

    public static Tensor GeluErf(IBackend backend, Tensor input)
    {
        Tensor output = new(input.Shape, DType.F32);
        backend.GeluErf(output, input);
        return output;
    }

    public static Tensor Add(IBackend backend, Tensor a, Tensor b)
    {
        Tensor output = new(a.Shape, DType.F32);
        backend.Add(output, a, b);
        return output;
    }

    /// <summary>The same tensor viewed with a new shape of equal element count; copies when the backend cannot alias.</summary>
    public static Tensor Reshaped(Tensor tensor, TensorShape shape) => tensor.Reshape(shape);
}
