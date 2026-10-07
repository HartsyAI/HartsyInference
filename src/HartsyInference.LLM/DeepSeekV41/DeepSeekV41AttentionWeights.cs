namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Dequantized F32 weights of one V4.1 attention layer.</summary>
/// <param name="WqA">Query down projection, <c>[qLoraRank, dim]</c>.</param>
/// <param name="QNorm">Query latent norm, <c>[qLoraRank]</c>.</param>
/// <param name="WqB">Query up projection, <c>[heads * headDim, qLoraRank]</c>.</param>
/// <param name="Wkv">Shared key/value latent projection, <c>[headDim, dim]</c>.</param>
/// <param name="KvNorm">Latent norm, <c>[headDim]</c>.</param>
/// <param name="WoA">Grouped output projection, <c>[oGroups * oLoraRank, heads * headDim / oGroups]</c>.</param>
/// <param name="WoB">Output up projection, <c>[dim, oGroups * oLoraRank]</c>.</param>
/// <param name="Sink">Per-head attention sink logit, <c>[heads]</c>.</param>
/// <param name="Compressor">Present only on a layer that compresses its own KV.</param>
/// <param name="Indexer">Present only on an index-source layer.</param>
public sealed record DeepSeekV41AttentionWeights(float[] WqA, float[] QNorm, float[] WqB, float[] Wkv, float[] KvNorm, float[] WoA, float[] WoB,
    float[] Sink, DeepSeekV41CompressorWeights? Compressor, DeepSeekV41IndexerWeights? Indexer);
