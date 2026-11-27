using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Typed weights of the HairDeck set DiT (<c>hairdeck/models/setdit.py</c>): the token projections, timestep and
/// pooling MLPs, the adaLN-modulated blocks (self-attention, cross-attention to the context, GELU MLP) and the output head.
/// Every block has a cross-attention; the layers the released models do not run (<c>head_norm</c>, <c>head_proj</c>,
/// <c>cond_*</c> of the chain model) are not read.</summary>
public static class HairDeckSetDitCheckpointConverter
{
    /// <summary>Linear layer weights; <see cref="Bias"/> is null for the bias-free attention projections.</summary>
    public sealed record Linear(Tensor Weight, Tensor? Bias);

    /// <summary>QK-norm RMSNorm scales of one attention; the projections are bias-free except the output.</summary>
    public sealed record Attention(Linear Query, Linear KeyValue, Linear Output, Tensor QueryNorm, Tensor KeyNorm);

    /// <summary>One block: 9-way adaLN projection, self-attention, cross-attention and the tanh-GELU MLP.</summary>
    public sealed record Block(Linear Ada, Attention SelfAttention, Attention CrossAttention, Linear MlpIn, Linear MlpOut);

    /// <summary>The set DiT with its width, head count, token width and null context.</summary>
    public sealed record Weights(
        int Dim,
        int Heads,
        int TokenDim,
        int ContextTokens,
        Linear Input,
        Linear TimeMlpIn,
        Linear TimeMlpOut,
        Linear PoolMlpIn,
        Linear PoolMlpOut,
        IReadOnlyList<Block> Blocks,
        Linear FinalAda,
        Linear Output,
        Tensor NullContext);

    private const int TimestepDim = 256;

    /// <summary>Reads the set DiT under <paramref name="prefix"/> (<c>""</c> or <c>"dit."</c>); the shapes give the width,
    /// the depth and the head count, and every other tensor is checked against them.</summary>
    public static Weights Read(IReadOnlyDictionary<string, Tensor> all, string prefix)
    {
        Tensor inp = HairDeckTensorReader.TakeRank(all, prefix + "inp.weight", 2);
        long dim = inp.Shape[0], tokenDim = inp.Shape[1];
        Tensor headNorm = HairDeckTensorReader.TakeRank(all, prefix + "blocks.0.attn.qn.w", 1);
        long headDim = headNorm.Shape[0];
        if (dim % headDim != 0)
            throw new InvalidDataException($"HairDeck width {dim} is not a multiple of the head dim {headDim}.");
        int depth = 0;
        while (HairDeckTensorReader.Exists(all, $"{prefix}blocks.{depth}.ada.weight")) depth++;
        if (depth == 0)
            throw new InvalidDataException($"HairDeck checkpoint under '{prefix}' has no blocks.");

        Tensor nullCtx = HairDeckTensorReader.TakeRank(all, prefix + "null_ctx", 2);
        if (nullCtx.Shape[1] != dim)
            throw new InvalidDataException($"HairDeck null_ctx width {nullCtx.Shape[1]} does not match dim {dim}.");

        long hidden = 4 * dim;
        List<Block> blocks = new(depth);
        for (int i = 0; i < depth; i++)
        {
            string b = $"{prefix}blocks.{i}.";
            blocks.Add(new Block(
                Ada: Lin(all, b + "ada", 9 * dim, dim),
                SelfAttention: Attn(all, b + "attn.", dim, headDim),
                CrossAttention: Attn(all, b + "xattn.", dim, headDim),
                MlpIn: Lin(all, b + "mlp.0", hidden, dim),
                MlpOut: Lin(all, b + "mlp.2", dim, hidden)));
        }

        return new Weights(
            Dim: (int)dim,
            Heads: (int)(dim / headDim),
            TokenDim: (int)tokenDim,
            ContextTokens: (int)nullCtx.Shape[0],
            Input: Lin(all, prefix + "inp", dim, tokenDim),
            TimeMlpIn: Lin(all, prefix + "t_mlp.0", dim, TimestepDim),
            TimeMlpOut: Lin(all, prefix + "t_mlp.2", dim, dim),
            PoolMlpIn: Lin(all, prefix + "pool_mlp.0", dim, dim),
            PoolMlpOut: Lin(all, prefix + "pool_mlp.2", dim, dim),
            Blocks: blocks,
            FinalAda: Lin(all, prefix + "f_ada", 2 * dim, dim),
            Output: Lin(all, prefix + "out", tokenDim, dim),
            NullContext: nullCtx);
    }

    /// <summary>A linear layer with weight <c>[outDim, inDim]</c> and bias <c>[outDim]</c>.</summary>
    public static Linear Lin(IReadOnlyDictionary<string, Tensor> all, string name, long outDim, long inDim)
    {
        Tensor w = HairDeckTensorReader.Take(all, name + ".weight", outDim, inDim);
        Tensor b = HairDeckTensorReader.Take(all, name + ".bias", outDim);
        return new Linear(w, b);
    }

    private static Attention Attn(IReadOnlyDictionary<string, Tensor> all, string prefix, long dim, long headDim)
    {
        Linear q = new(HairDeckTensorReader.Take(all, prefix + "q.weight", dim, dim), null);
        Linear kv = new(HairDeckTensorReader.Take(all, prefix + "kv.weight", 2 * dim, dim), null);
        Linear o = Lin(all, prefix + "o", dim, dim);
        Tensor qn = HairDeckTensorReader.Take(all, prefix + "qn.w", headDim);
        Tensor kn = HairDeckTensorReader.Take(all, prefix + "kn.w", headDim);
        return new Attention(q, kv, o, qn, kn);
    }
}
