namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Dequantized F32 weights of a layer's KV compressor.</summary>
/// <param name="Wkv">Latent projection, <c>[headDim, dim]</c>.</param>
/// <param name="Wgate">Pooling-gate projection, <c>[headDim, dim]</c>; null for ratio 1, which does not pool.</param>
/// <param name="Norm">Output RMS norm weight, <c>[headDim]</c>.</param>
public sealed record DeepSeekV41CompressorWeights(DeepSeekV41Weight Wkv, DeepSeekV41Weight? Wgate, float[] Norm);
