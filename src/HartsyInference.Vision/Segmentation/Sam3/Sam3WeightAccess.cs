using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Resolves named checkpoint tensors and checks their shapes before any op uses them.</summary>
internal static class Sam3WeightAccess
{
    /// <summary>Returns the tensor under <paramref name="key"/> as F32; throws naming the key when it is absent or mis-shaped.</summary>
    /// <param name="weights">Checkpoint tensors keyed relative to their component prefix.</param>
    /// <param name="key">Weight key to resolve.</param>
    /// <param name="expectedShape">Required dimensions in order.</param>
    public static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] expectedShape)
    {
        return TensorCasts.EnsureF32(CheckpointTensors.Require(weights, "Sam3", key, expectedShape));
    }
}
