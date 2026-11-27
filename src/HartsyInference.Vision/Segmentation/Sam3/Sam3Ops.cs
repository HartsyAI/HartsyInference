using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Weight lookup, shape checks and small tensor helpers shared by the SAM 3 detector transformer modules.</summary>
internal static class Sam3Ops
{
    /// <summary>Checks that <paramref name="tensor"/> is F32 with exactly <paramref name="expected"/> dimensions.</summary>
    public static void RequireShape(Tensor tensor, string name, params long[] expected)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.DType != DType.F32)
            throw new ArgumentException($"SAM 3 tensor '{name}' must be F32, found {tensor.DType}.", nameof(tensor));
        if (tensor.Shape.Rank != expected.Length)
            throw new ArgumentException($"SAM 3 tensor '{name}' has rank {tensor.Shape.Rank}, expected {expected.Length}.", nameof(tensor));
        for (int i = 0; i < expected.Length; i++)
        {
            if (tensor.Shape[i] != expected[i])
                throw new ArgumentException($"SAM 3 tensor '{name}' has shape {tensor.Shape}, expected dimension {i} = {expected[i]}.", nameof(tensor));
        }
    }

    /// <summary>Checks a batch-1 token sequence <c>[1, S, width]</c>.</summary>
    public static void RequireSequence(Tensor tensor, string name, long width)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.Shape.Rank != 3 || tensor.Shape[0] != 1 || tensor.Shape[2] != width)
            throw new ArgumentException($"SAM 3 tensor '{name}' must be [1, S, {width}], found {tensor.Shape}.", nameof(tensor));
        if (tensor.DType != DType.F32)
            throw new ArgumentException($"SAM 3 tensor '{name}' must be F32, found {tensor.DType}.", nameof(tensor));
    }

    /// <summary>Allocates a zero-initialized F32 tensor.</summary>
    public static Tensor NewF32(params long[] dims) => new(new TensorShape(dims), DType.F32);

    /// <summary>LayerNorm over the last dimension of a <c>[1, S, D]</c> tensor; returns a new tensor.</summary>
    public static Tensor LayerNorm(IBackend backend, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        Tensor output = new(input.Shape, DType.F32);
        backend.LayerNorm(output, input, weight, bias, eps);
        return output;
    }

    /// <summary>Elementwise sum of two same-shaped tensors; returns a new tensor.</summary>
    public static Tensor Add(IBackend backend, Tensor a, Tensor b)
    {
        Tensor output = new(a.Shape, DType.F32);
        backend.Add(output, a, b);
        return output;
    }

    /// <summary>Linear projection of a <c>[1, S, In]</c> input to <c>[1, S, Out]</c>; returns a new tensor.</summary>
    public static Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor? bias)
    {
        Tensor output = new(new TensorShape(input.Shape[0], input.Shape[1], weight.Shape[0]), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    /// <summary>In-place ReLU.</summary>
    public static void Relu(IBackend backend, Tensor tensor) => backend.Clamp(tensor, tensor, 0f, float.MaxValue);

    /// <summary>Concatenates two <c>[1, S, D]</c> sequences along the sequence axis; returns a new tensor.</summary>
    public static Tensor ConcatRows(IBackend backend, Tensor first, Tensor second)
    {
        Tensor output = NewF32(first.Shape[0], first.Shape[1] + second.Shape[1], first.Shape[2]);
        Tensor[] parts = [first, second];
        backend.Concat(output, parts, 1);
        return output;
    }
}
