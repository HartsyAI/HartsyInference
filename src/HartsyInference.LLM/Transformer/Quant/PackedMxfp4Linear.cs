using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.LLM.Transformer.Quant;

/// <summary>A linear layer whose weight stays MXFP4-packed: <c>y = x · Wᵀ</c> with <c>W</c> decoded row by row from its E2M1 nibbles and
/// E8M0 block scales. This is the host reference for the packed expert path; the device paths must match it.</summary>
public static class PackedMxfp4Linear
{
    /// <summary>Computes <paramref name="output"/> <c>[n, out]</c> from <paramref name="input"/> <c>[n, in]</c> and a packed weight.</summary>
    /// <param name="packedWeight">U8 <c>[out, in / 2]</c>: two E2M1 codes per byte, low nibble the even column.</param>
    /// <param name="recipe">The weight's <see cref="QuantEncoding.Mxfp4E8M0"/> recipe: <c>LogicalRows = out</c>, <c>LogicalCols = in</c>, and its
    /// E8M0 scale grid.</param>
    public static void Apply(Tensor output, Tensor input, Tensor packedWeight, QuantRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(packedWeight);
        ArgumentNullException.ThrowIfNull(recipe);
        if (recipe.Encoding != QuantEncoding.Mxfp4E8M0)
            throw new NotSupportedException($"PackedMxfp4Linear takes an MXFP4 recipe, got {recipe.Encoding}.");
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("PackedMxfp4Linear takes F32 activations and output.");
        int outRows = checked((int)recipe.LogicalRows), inCols = checked((int)recipe.LogicalCols);
        if (input.ElementCount % inCols != 0 || output.ElementCount != input.ElementCount / inCols * outRows)
            throw new ArgumentException($"Input {input.Shape} and output {output.Shape} do not match a weight of {outRows}×{inCols}.");
        int n = checked((int)(input.ElementCount / inCols));

        // One output feature at a time: decode its row, then dot it with every input row. Decoding is per row, so the
        // resident F32 scratch is one row, not the whole expert.
        float[] row = new float[inCols];
        ReadOnlySpan<float> x = input.AsReadOnlySpan<float>();
        Span<float> y = output.AsSpan<float>();
        ReadOnlySpan<byte> packed = packedWeight.AsReadOnlySpan<byte>();
        for (int o = 0; o < outRows; o++)
        {
            Mxfp4E8M0Codec.DequantRows(packed, recipe, o, 1, row);
            for (int r = 0; r < n; r++)
            {
                float acc = 0f;
                ReadOnlySpan<float> xr = x.Slice(r * inCols, inCols);
                for (int c = 0; c < inCols; c++) acc += xr[c] * row[c];
                y[r * outRows + o] = acc;
            }
        }
    }
}
