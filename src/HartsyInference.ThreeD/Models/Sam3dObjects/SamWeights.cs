using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>Weight lookups shared by the SAM 3D modules: every tensor must be present, of the expected dtype and of the expected shape. Name every dimension: an empty shape checks none.</summary>
internal static class SamWeights
{
    private const string Family = "SAM 3D";

    /// <summary>The tensor under <paramref name="key"/>, borrowed.</summary>
    /// <exception cref="KeyNotFoundException">The checkpoint has no such tensor.</exception>
    /// <exception cref="InvalidDataException">The tensor has another shape.</exception>
    /// <exception cref="InvalidOperationException">The tensor is not F32: widen 16-bit weights first.</exception>
    public static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        Tensor tensor = CheckpointTensors.Require(weights, Family, key, shape);
        if (tensor.DType != DType.F32)
            throw new InvalidOperationException($"SAM 3D weight '{key}' is {tensor.DType}; the module takes F32 weights, so widen 16-bit weights first.");
        return tensor;
    }

    /// <summary>
    /// As <see cref="Require"/> for a linear weight a backend GEMM reads: F32, F16 or BF16 (see <see cref="SamFlowWeights"/>).
    /// </summary>
    public static Tensor RequireLinear(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        Tensor tensor = CheckpointTensors.Require(weights, Family, key, shape);
        if (tensor.DType != DType.F32 && tensor.DType != DType.F16 && tensor.DType != DType.BF16)
            throw new InvalidOperationException($"SAM 3D weight '{key}' is {tensor.DType}; the module takes F32, F16 or BF16 linear weights.");
        return tensor;
    }
}
