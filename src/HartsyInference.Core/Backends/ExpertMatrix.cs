using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Core.Backends;

/// <summary>One expert projection: the (possibly packed) weight and the recipe whose scale, global scale and bias tensors ride with it; a null recipe means an unquantized weight.</summary>
public sealed record ExpertMatrix(Tensor Weight, QuantRecipe? Recipe = null)
{
    /// <summary>The block scales, or null for an unquantized weight.</summary>
    public Tensor? Scale => Recipe?.Scale;

    /// <summary>The scale, global scale and bias the recipe's dequant reads, in that order; empty for an unquantized weight.</summary>
    public IEnumerable<Tensor> Companions => Recipe?.DequantCompanions() ?? [];
}
