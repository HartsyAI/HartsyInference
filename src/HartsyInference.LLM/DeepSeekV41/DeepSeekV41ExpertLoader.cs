using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Dequantizes one routed expert of a checkpoint to F32 and checks its three matrices against the layer's widths.</summary>
public static class DeepSeekV41ExpertLoader
{
    /// <summary>Reads <paramref name="expert"/> from <paramref name="bank"/> as <see cref="DeepSeekV41SwigluWeights"/>.</summary>
    /// <param name="bank">The layer's borrowed expert views.</param>
    /// <param name="expert">Routed expert index.</param>
    /// <param name="dim">Hidden width (<c>hidden_size</c>).</param>
    /// <param name="inter">Intermediate width (<c>moe_intermediate_size</c>).</param>
    /// <exception cref="HartsyInferenceException">A projection's logical shape is not the expected <c>[inter, dim]</c> or <c>[dim, inter]</c>.</exception>
    public static DeepSeekV41SwigluWeights Load(DeepSeekV41ExpertBank bank, int expert, int dim, int inter) =>
        new(dim, inter,
            Read(bank, expert, DeepSeekV41ExpertProjection.W1, inter, dim),
            Read(bank, expert, DeepSeekV41ExpertProjection.W2, dim, inter),
            Read(bank, expert, DeepSeekV41ExpertProjection.W3, inter, dim));

    private static float[] Read(DeepSeekV41ExpertBank bank, int expert, DeepSeekV41ExpertProjection projection, long rows, long cols)
    {
        Tensor weight = bank.Weight(expert, projection);
        QuantWeightInfo? quant = bank.Quant(expert, projection);
        long actualRows = quant?.Recipe?.LogicalRows ?? weight.Shape[0];
        long actualCols = quant?.Recipe?.LogicalCols ?? weight.Shape[1];
        if (actualRows != rows || actualCols != cols)
            throw new HartsyInferenceException(
                $"{bank.WeightKey(expert, projection)} is [{actualRows}, {actualCols}], expected [{rows}, {cols}].");
        return WeightDequantizer.ToF32(weight, quant);
    }
}
