using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>Weight lookups shared by the SAM 3D modules: every tensor must be present, F32 and of the expected shape.</summary>
internal static class SamWeights
{
    /// <summary>The tensor under <paramref name="key"/>, borrowed.</summary>
    /// <exception cref="InvalidOperationException">The key is missing, the tensor is not F32 (widen 16-bit weights first) or has another shape.</exception>
    public static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? tensor))
            throw new InvalidOperationException($"SAM 3D weight '{key}' is missing.");
        if (tensor.DType != DType.F32)
            throw new InvalidOperationException($"SAM 3D weight '{key}' is {tensor.DType}; the module takes F32 weights, so widen 16-bit weights first.");
        if (!tensor.Shape.Equals(new TensorShape(shape)))
            throw new InvalidOperationException($"SAM 3D weight '{key}' has shape {tensor.Shape}, expected {new TensorShape(shape)}.");
        return tensor;
    }
}
