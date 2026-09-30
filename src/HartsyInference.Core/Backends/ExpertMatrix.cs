using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Core.Backends;

/// <summary>One expert projection: the (possibly packed) weight and the recipe whose scale tensor rides with it; a null recipe means an unquantized weight.</summary>
public sealed record ExpertMatrix(Tensor Weight, QuantRecipe? Recipe = null)
{
    /// <summary>The block scales, or null for an unquantized weight.</summary>
    public Tensor? Scale => Recipe?.Scale;
}
