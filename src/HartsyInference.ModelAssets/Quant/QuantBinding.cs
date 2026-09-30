using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.ModelAssets.Quant;

/// <summary>A weight paired with its companion keys and inferred geometry, decided from headers alone; <see cref="ToRecipe"/> attaches the tensors once they are mapped.</summary>
public sealed record QuantBinding(
    string WeightKey,
    QuantEncoding Encoding,
    BlockGeometry Geometry,
    DType ScaleDType,
    long LogicalRows,
    long LogicalCols,
    string? ScaleKey,
    string? GlobalScaleKey = null,
    string? InputScaleKey = null,
    string? BiasKey = null,
    Exl3Keys? Exl3 = null)
{
    /// <summary>Builds the recipe, resolving each companion key through <paramref name="getTensor"/> (borrowed views, as <c>ShardedSafeTensorSet.GetTensor</c> returns).</summary>
    public QuantRecipe ToRecipe(Func<string, Tensor> getTensor)
    {
        ArgumentNullException.ThrowIfNull(getTensor);
        Exl3Companions? exl3 = Exl3 is null ? null
            : new Exl3Companions(getTensor(Exl3.Suh), getTensor(Exl3.Svh), getTensor(Exl3.Mcg), Exl3.Bits);
        return new QuantRecipe
        {
            Encoding = Encoding,
            Geometry = Geometry,
            ScaleDType = ScaleDType,
            LogicalRows = LogicalRows,
            LogicalCols = LogicalCols,
            Scale = ScaleKey is null ? null : getTensor(ScaleKey),
            GlobalScale = GlobalScaleKey is null ? null : getTensor(GlobalScaleKey),
            InputScale = InputScaleKey is null ? null : getTensor(InputScaleKey),
            Bias = BiasKey is null ? null : getTensor(BiasKey),
            Exl3 = exl3,
        };
    }

    /// <summary>The weight's <see cref="QuantWeightInfo"/> carrying the recipe under its own format string.</summary>
    public QuantWeightInfo ToWeightInfo(Func<string, Tensor> getTensor)
    {
        QuantRecipe recipe = ToRecipe(getTensor);
        return new QuantWeightInfo { Format = recipe.FormatName, Recipe = recipe };
    }
}
