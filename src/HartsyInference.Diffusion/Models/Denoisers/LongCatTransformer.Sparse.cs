using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Layers.Attention;

namespace HartsyInference.Diffusion.Models.Denoisers;

public sealed unsafe partial class LongCatTransformer
{
    /// <summary>The block-sparse options of every segment of a refinement forward: the segment's own queries against its keys, both
    /// whole latent frames of <paramref name="gridH"/> × <paramref name="gridW"/> tokens. As in the reference, the conditioning
    /// queries read the conditioning keys and the noisy queries read every key, each through its own selection.</summary>
    private BlockSparseAttentionOptions[] SparsePlan(IBackend backend, AttentionSegment[] segments, int tokensPerFrame, int gridH,
        int gridW, in LongCatConditioning conditioning, bool avatar)
    {
        if (avatar || conditioning.NumRefLatents > 0)
            throw new ArgumentException("Block-sparse self-attention is the base model's refinement; an avatar forward runs dense.");
        if (!backend.HasBlockSparseAttention)
            throw new AttentionUnsupportedException(
                $"LongCat's refinement needs block-sparse attention, which the {backend.Device} backend does not run.");
        BlockSparseAttentionOptions[] options = new BlockSparseAttentionOptions[segments.Length];
        for (int i = 0; i < segments.Length; i++)
        {
            AttentionSegment s = segments[i];
            if (s.KeyStart != 0 || s.QueryCount % tokensPerFrame != 0 || s.KeyCount % tokensPerFrame != 0)
                throw new InvalidOperationException($"Block-sparse self-attention takes whole frames with keys from row 0; got {s}.");
            options[i] = _config.SparseAttention.Options(s.QueryCount / tokensPerFrame, s.KeyCount / tokensPerFrame, gridH, gridW);
        }
        return options;
    }

    /// <summary>The heads one block-sparse call takes in every block, 0 for the block's own choice (a test seam).</summary>
    internal void SetSparseHeadGroup(int heads)
    {
        foreach (DiTBlocks.LongCatBlock block in _blocks)
            block.SetSparseHeadGroup(heads);
    }
}
