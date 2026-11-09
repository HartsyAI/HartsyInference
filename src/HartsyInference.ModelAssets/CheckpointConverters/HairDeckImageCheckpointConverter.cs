using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Typed weights of the image-conditioned set DiT (<c>hairdeck/models/imgmodel.py</c>): the set DiT plus the
/// projection of the DINOv3 context and its learned per-token position embedding.</summary>
public static class HairDeckImageCheckpointConverter
{
    /// <summary>The image DiT: <see cref="Dit"/> with the context projection <see cref="CondProj"/> (after LayerNorm
    /// <see cref="CondNormWeight"/>/<see cref="CondNormBias"/>) and the position embedding <see cref="CtxPos"/>.</summary>
    public sealed record Weights(
        HairDeckSetDitCheckpointConverter.Weights Dit,
        int DinoWidth,
        Tensor CondNormWeight,
        Tensor CondNormBias,
        HairDeckSetDitCheckpointConverter.Linear CondProj,
        Tensor CtxPos);

    /// <summary>Reads the image DiT under <paramref name="prefix"/> (<c>""</c> for the released image model, <c>"dit."</c> for the colour model).</summary>
    public static Weights Convert(IReadOnlyDictionary<string, Tensor> all, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(all);
        HairDeckSetDitCheckpointConverter.Weights dit = HairDeckSetDitCheckpointConverter.Read(all, prefix);
        Tensor condNorm = HairDeckTensorReader.TakeRank(all, prefix + "cond_norm.weight", 1);
        long dino = condNorm.Shape[0];
        Tensor ctxPos = HairDeckTensorReader.Take(all, prefix + "ctx_pos", dit.ContextTokens, dit.Dim);
        return new Weights(
            Dit: dit,
            DinoWidth: (int)dino,
            CondNormWeight: condNorm,
            CondNormBias: HairDeckTensorReader.Take(all, prefix + "cond_norm.bias", dino),
            CondProj: HairDeckSetDitCheckpointConverter.Lin(all, prefix + "cond_proj", dit.Dim, dino),
            CtxPos: ctxPos);
    }
}
