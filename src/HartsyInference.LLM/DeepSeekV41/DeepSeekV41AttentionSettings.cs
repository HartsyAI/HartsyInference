namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The shape and role of one V4.1 attention layer.</summary>
/// <param name="Dim">Hidden width.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="HeadDim">Latent width, which is also each head's width.</param>
/// <param name="RopeDim">Trailing slice of each vector that is rotated.</param>
/// <param name="QLoraRank">Query latent width.</param>
/// <param name="OGroups">Output projection groups.</param>
/// <param name="OLoraRank">Per-group output latent width.</param>
/// <param name="Window">Sliding-window length.</param>
/// <param name="CompressRatio">0 for window-only attention, else tokens per compressed latent.</param>
/// <param name="IsKvSource">Whether this layer compresses KV that later layers reuse.</param>
/// <param name="IsIndexSource">Whether this layer runs an indexer whose indices later layers reuse.</param>
/// <param name="IsCandidateSource">Whether this layer's indexer picks the level-one candidate blocks.</param>
/// <param name="UsesCandidates">Whether this layer's indexer is restricted to those blocks.</param>
/// <param name="IndexHeads">Indexer heads.</param>
/// <param name="IndexHeadDim">Indexer key/query width.</param>
/// <param name="IndexTopk">Compressed positions kept per query.</param>
/// <param name="CandidateTopkBlocks">Blocks kept at level one.</param>
/// <param name="CandidateBlockSize">Positions per candidate block.</param>
/// <param name="NormEps">RMS norm epsilon.</param>
public sealed record DeepSeekV41AttentionSettings(int Dim, int Heads, int HeadDim, int RopeDim, int QLoraRank, int OGroups, int OLoraRank, int Window,
    int CompressRatio, bool IsKvSource, bool IsIndexSource, bool IsCandidateSource, bool UsesCandidates, int IndexHeads, int IndexHeadDim,
    int IndexTopk, int CandidateTopkBlocks, int CandidateBlockSize, float NormEps)
{
    /// <summary>Width of one head-group slice fed to <c>wo_a</c>.</summary>
    public int GroupDim => Heads * HeadDim / OGroups;
}
