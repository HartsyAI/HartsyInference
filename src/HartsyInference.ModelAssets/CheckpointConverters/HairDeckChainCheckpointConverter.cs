using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Typed weights of the spring-bone chain DiT (<c>hairdeck/spring/chaindit.py</c>): the set DiT over 27-channel
/// chain tokens, conditioned on 192 card tokens through their LayerNorm, projection and a slot-wide position.</summary>
public static class HairDeckChainCheckpointConverter
{
    /// <summary>The chain DiT. The card context is <c>card_proj(card_norm(cards)) + card_pos</c>.</summary>
    public sealed record Weights(
        HairDeckSetDitCheckpointConverter.Weights Dit,
        int CardWidth,
        Tensor CardNormWeight,
        Tensor CardNormBias,
        HairDeckSetDitCheckpointConverter.Linear CardProj,
        Tensor CardPos);

    /// <summary>Reads the chain DiT; the unused <c>cond_*</c> and <c>head_*</c> tensors are not read.</summary>
    public static Weights Convert(IReadOnlyDictionary<string, Tensor> all)
    {
        ArgumentNullException.ThrowIfNull(all);
        HairDeckSetDitCheckpointConverter.Weights dit = HairDeckSetDitCheckpointConverter.Read(all, string.Empty);
        const long cardWidth = 40;
        return new Weights(
            Dit: dit,
            CardWidth: (int)cardWidth,
            CardNormWeight: HairDeckTensorReader.Take(all, "card_norm.weight", cardWidth),
            CardNormBias: HairDeckTensorReader.Take(all, "card_norm.bias", cardWidth),
            CardProj: HairDeckSetDitCheckpointConverter.Lin(all, "card_proj", dit.Dim, cardWidth),
            CardPos: HairDeckTensorReader.Take(all, "card_pos", 1, 1, dit.Dim));
    }
}
