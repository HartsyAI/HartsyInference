namespace HartsyInference.ModelAssets.Duo;

/// <summary>Row order of the fused <c>qkv_proj</c> in the backbone shards being repacked.</summary>
public enum DuoQkvInput
{
    /// <summary>The released MiniMax-H3 and Duo shards: <c>[head0: q k v, head1: q k v, ...]</c>.</summary>
    HeadInterleaved,

    /// <summary>Already <c>[Q; K; V]</c> blocks, as the Comfy-Org H3 builds the engine runs today.</summary>
    Blocks,
}
