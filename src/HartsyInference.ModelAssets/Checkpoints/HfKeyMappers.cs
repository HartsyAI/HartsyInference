using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>Chooses the key mapper for a safetensors checkpoint from the flavor its config declares.</summary>
public static class HfKeyMappers
{
    /// <summary>The mapper for <paramref name="flavor"/>; a config with no quantization block is the official naming.</summary>
    public static IHfKeyMapper ForSafeTensors(QuantFlavor? flavor) => flavor switch
    {
        QuantFlavor.Mlx => new MlxV41KeyMapper(),
        QuantFlavor.Exl3 => new Exl3V41KeyMapper(),
        _ => new OfficialV41KeyMapper(),
    };
}
