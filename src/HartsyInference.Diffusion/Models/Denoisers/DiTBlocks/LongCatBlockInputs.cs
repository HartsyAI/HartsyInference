using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>What every block of one LongCat forward shares: the token grid, the conditioning split, the RoPE tables, the attention plan and the
/// token-to-frame modulation index. Built once per forward by the transformer; the tables and the index are owned by it.</summary>
/// <param name="Frames">Latent frames.</param>
/// <param name="TokensPerFrame">Token rows per frame (<c>gridH · gridW</c>).</param>
/// <param name="CondFrames">Leading frames that are conditioning (they take no text update and, in the plan, attend only their own group).</param>
/// <param name="Cos">RoPE cosines <c>[S, headDim]</c>, one value per interleaved pair duplicated.</param>
/// <param name="Sin">RoPE sines, same layout.</param>
/// <param name="Segments">The self-attention plan over the <c>S</c> tokens.</param>
/// <param name="FrameRowIndex">I32 <c>[S]</c>, each token row's frame.</param>
/// <param name="Sparse">Null for dense self-attention; else the block-sparse options of each segment, in the segments' order.</param>
public readonly record struct LongCatBlockInputs(int Frames, int TokensPerFrame, int CondFrames, Tensor Cos, Tensor Sin,
    AttentionSegment[] Segments, Tensor FrameRowIndex, BlockSparseAttentionOptions[]? Sparse = null);
