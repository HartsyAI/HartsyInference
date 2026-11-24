using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Maxwell;

/// <summary>Geometry of Maxwell block-sparse attention (MBSA): per KV group a query attends to a fixed prefix and a local window of blocks
/// plus the remote blocks its indexer scores highest.</summary>
public sealed record MaxwellSparseConfig
{
    /// <summary>Indexer groups; one per key/value head.</summary>
    public required int IndexerGroups { get; init; }

    /// <summary>Width of the indexer's queries and keys.</summary>
    public required int IndexerHeadDim { get; init; }

    /// <summary>Tokens per block.</summary>
    public required int BlockSize { get; init; }

    /// <summary>Blocks at the start of the sequence every query keeps.</summary>
    public required int PrefixBlocks { get; init; }

    /// <summary>Blocks ending at the query's own block that every query keeps.</summary>
    public required int LocalBlocks { get; init; }

    /// <summary>Blocks the indexer adds, chosen among those outside both windows.</summary>
    public required int RemoteBlocks { get; init; }

    /// <summary>The first position whose attention can differ from dense causal attention: until then every causal block is in the prefix, the
    /// local window or the remote picks, so the layer needs no indexer.</summary>
    public int DenseCausalLimit => (PrefixBlocks + LocalBlocks + RemoteBlocks) * BlockSize;

    internal static MaxwellSparseConfig Parse(HfConfigFields r)
    {
        int block = r.Int("block_sparse_gqa_block_size");
        int prefixTokens = r.Int("block_sparse_gqa_prefix_window_size"), localTokens = r.Int("block_sparse_gqa_local_window_size");
        if (block < 1 || prefixTokens % block != 0 || localTokens % block != 0)
            throw new HartsyInferenceException("Maxwell config: the prefix and local windows must be whole blocks.");
        MaxwellSparseConfig sparse = new()
        {
            IndexerGroups = r.Int("block_sparse_gqa_indexer_n_heads"),
            IndexerHeadDim = r.Int("block_sparse_gqa_indexer_head_dim"),
            BlockSize = block,
            PrefixBlocks = prefixTokens / block,
            LocalBlocks = localTokens / block,
            RemoteBlocks = r.Int("block_sparse_gqa_num_remote_blocks"),
        };
        if (r.OptionalString("block_sparse_gqa_group_reduction") != "per_group_max")
            throw new HartsyInferenceException("Maxwell config: only block_sparse_gqa_group_reduction per_group_max is wired.");
        int support = r.Int("block_sparse_gqa_total_support_blocks");
        if (support != sparse.PrefixBlocks + sparse.LocalBlocks + sparse.RemoteBlocks)
            throw new HartsyInferenceException($"Maxwell config: total_support_blocks {support} is not prefix + local + remote blocks.");
        return sparse;
    }
}
