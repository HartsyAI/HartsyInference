using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.LLM.Transformer.Quant;

/// <summary>One MXFP4-packed projection of a routed expert: its U8 packed weight <c>[out, in / 2]</c> and the recipe that carries its E8M0
/// scales, with <c>LogicalRows = out</c> and <c>LogicalCols = in</c>.</summary>
/// <param name="Packed">U8 <c>[out, in / 2]</c>.</param>
/// <param name="Recipe">The <see cref="QuantEncoding.Mxfp4E8M0"/> recipe over <paramref name="Packed"/>.</param>
public readonly record struct PackedMxfp4Projection(Tensor Packed, QuantRecipe Recipe);

/// <summary>A routed expert with its gate, up and down projections packed (<c>w1</c>, <c>w3</c> and <c>w2</c> in the checkpoint).</summary>
public sealed record PackedMxfp4Expert(PackedMxfp4Projection Gate, PackedMxfp4Projection Up, PackedMxfp4Projection Down);
