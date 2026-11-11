using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Core.Backends;

// A linear layer whose weight stays MXFP4-packed: the packed expert projections of MoE layers.
public partial interface IBackend
{
    /// <summary>F32 <c>y = x · Wᵀ</c> with <c>W</c> an MXFP4 weight kept packed (E2M1 nibbles, E8M0 scales per 32 columns). The host
    /// reference is the default; a backend with a device kernel overrides it.</summary>
    /// <param name="output">F32 <c>[n, rows]</c>.</param>
    /// <param name="input">F32 <c>[n, cols]</c>.</param>
    /// <param name="packedWeight">U8 <c>[rows, cols / 2]</c>.</param>
    /// <param name="recipe">The weight's <see cref="QuantEncoding.Mxfp4E8M0"/> recipe, whose scales are U8 <c>[rows, cols / 32]</c>.</param>
    void Mxfp4LinearF32(Tensor output, Tensor input, Tensor packedWeight, QuantRecipe recipe)
        => Mxfp4LinearReference.Apply(output, input, packedWeight, recipe);
}
