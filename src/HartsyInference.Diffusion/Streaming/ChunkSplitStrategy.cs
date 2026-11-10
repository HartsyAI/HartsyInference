namespace HartsyInference.Diffusion.Streaming;

/// <summary>How a causal sequence of frames is cut into chunks of a base size.</summary>
public enum ChunkSplitStrategy
{
    /// <summary>Every chunk has the base size. A bounded sequence merges a short remainder into its last chunk.</summary>
    Uniform,

    /// <summary>Chunk 0 is one frame, then base-size chunks. A bounded sequence merges a short remainder into its last chunk.</summary>
    FirstFrameAlone,

    /// <summary>Chunk 0 is the base size plus one (a conditioning frame and a full chunk), then base-size chunks. A bounded
    /// sequence merges a short remainder into its last chunk.</summary>
    FirstChunkPlusOne,

    /// <summary>Chunk 0 is the base size plus the remainder of the total, so every later chunk has the base size. Needs a
    /// bounded sequence.</summary>
    FirstAbsorbsRemainder,
}
