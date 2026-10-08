using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Boundary checks shared by the vision tower and aligner weight holders.</summary>
internal static class DeepSeekV41VisionTensors
{
    /// <summary>Throws unless <paramref name="tensor"/> is an F32 tensor of exactly <paramref name="shape"/>; <paramref name="name"/> is the checkpoint key it stands for.</summary>
    /// <exception cref="HartsyInferenceException">The tensor is missing, not F32 or has another shape.</exception>
    public static void Require(Tensor? tensor, string name, params long[] shape)
    {
        if (tensor is null) throw new HartsyInferenceException($"Vision weight '{name}' is missing.");
        if (tensor.DType != DType.F32) throw new HartsyInferenceException($"Vision weight '{name}' is {tensor.DType}; the CPU reference needs F32.");
        if (tensor.Shape != new TensorShape(shape))
            throw new HartsyInferenceException($"Vision weight '{name}' is {tensor.Shape}, expected {new TensorShape(shape)}.");
    }
}
