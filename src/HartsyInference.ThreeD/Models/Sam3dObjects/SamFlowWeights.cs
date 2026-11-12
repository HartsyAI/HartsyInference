using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// Narrows the transformer-block linear weights of a SAM 3D flow (the structure flow's <see cref="SamMotFlow"/> and the SLAT flow's <c>SlatFlowModel</c>, both of which keep the block weights a
/// backend GEMM reads as they come) to F16 or BF16: half the device memory for those weights, the GEMM running in 16 bits with F32 accumulation. Norm gains, biases, the timestep and latent
/// embedders and the sparse res-blocks stay as they are.
/// </summary>
public static class SamFlowWeights
{
    /// <summary>The weights <see cref="Narrow"/> converts: the rank-2 <c>.weight</c> tensors under <c>blocks.</c>.</summary>
    public static bool IsBlockLinear(string key, Tensor tensor) =>
        key.StartsWith("blocks.", StringComparison.Ordinal) && key.EndsWith(".weight", StringComparison.Ordinal) && tensor.Shape.Rank == 2;

    /// <summary>
    /// A copy of <paramref name="weights"/> whose block linears are <paramref name="linearDType"/>; every other tensor is the same object. The converted tensors are added to
    /// <paramref name="owned"/>, which the caller disposes once the flow is done. <see cref="DType.F32"/> returns the input unchanged.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="linearDType"/> is not F32, F16 or BF16.</exception>
    public static Dictionary<string, Tensor> Narrow(IReadOnlyDictionary<string, Tensor> weights, DType linearDType, List<Tensor> owned)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(owned);
        if (linearDType != DType.F32 && linearDType != DType.F16 && linearDType != DType.BF16)
            throw new ArgumentOutOfRangeException(nameof(linearDType), linearDType, "Flow linear weights are held as F32, F16 or BF16.");
        Dictionary<string, Tensor> result = new(weights.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in weights)
        {
            Tensor tensor = entry.Value;
            if (linearDType != DType.F32 && tensor.DType != linearDType && IsBlockLinear(entry.Key, tensor))
            {
                tensor = tensor.CastTo(linearDType);
                owned.Add(tensor);
            }

            result[entry.Key] = tensor;
        }

        return result;
    }

    /// <summary>
    /// Ends a flow's use of its weights: drains the backend, frees its device copies of <paramref name="resident"/>, then disposes the converted tensors in <paramref name="owned"/>. The order is the
    /// contract: a converted tensor's host memory must outlive every transfer and kernel the backend still has queued against it, and disposing it first freed memory the backend was still using (an
    /// access violation in the next stage on one seed in two).
    /// </summary>
    public static void Release(IBackend backend, IEnumerable<Tensor> resident, List<Tensor> owned)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(resident);
        ArgumentNullException.ThrowIfNull(owned);
        backend.Sync();
        backend.FreeWeights(resident);
        foreach (Tensor tensor in owned)
            tensor.Dispose();
        owned.Clear();
    }
}
