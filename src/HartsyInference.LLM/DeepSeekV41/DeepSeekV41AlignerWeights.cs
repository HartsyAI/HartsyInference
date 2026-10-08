using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The F32 weights of the aligner, named after the checkpoint's <c>aligner.*</c> keys; matrices are <c>[out, in]</c>.</summary>
/// <param name="W1">First projection, <c>[outDim, dim * ratio * ratio]</c>.</param>
/// <param name="B1">Its bias, <c>[outDim]</c>.</param>
/// <param name="W2">Second projection, <c>[outDim, outDim]</c>.</param>
/// <param name="B2">Its bias, <c>[outDim]</c>.</param>
public sealed record DeepSeekV41AlignerWeights(Tensor W1, Tensor B1, Tensor W2, Tensor B2)
{
    /// <summary>Checks every tensor against <paramref name="config"/> and the language model's hidden width <paramref name="outDim"/>.</summary>
    internal void Validate(DeepSeekV41VisionConfig config, int outDim)
    {
        DeepSeekV41VisionTensors.Require(W1, "aligner.w1.weight", outDim, config.AlignerInputDim);
        DeepSeekV41VisionTensors.Require(B1, "aligner.w1.bias", outDim);
        DeepSeekV41VisionTensors.Require(W2, "aligner.w2.weight", outDim, outDim);
        DeepSeekV41VisionTensors.Require(B2, "aligner.w2.bias", outDim);
    }

    internal void Dispose()
    {
        foreach (Tensor tensor in new[] { W1, B1, W2, B2 }) tensor?.Dispose();
    }
}
