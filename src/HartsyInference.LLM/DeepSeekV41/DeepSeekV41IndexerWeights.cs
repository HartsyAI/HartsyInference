namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Dequantized F32 weights of a layer's sparse-attention indexer.</summary>
/// <param name="WqB">Index query projection, <c>[indexHeads * indexHeadDim, qLoraRank]</c>.</param>
/// <param name="WeightsProj">Per-head score weights, <c>[indexHeads, dim]</c>.</param>
/// <param name="Wk">Index key projection from the compressed latent, <c>[indexHeadDim, headDim]</c>; null when another layer owns the keys.</param>
/// <param name="KNorm">Index key RMS norm weight, <c>[indexHeadDim]</c>; null with <paramref name="Wk"/>.</param>
public sealed record DeepSeekV41IndexerWeights(float[] WqB, float[] WeightsProj, float[]? Wk, float[]? KNorm);
