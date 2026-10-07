using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Host dequantization of a whole checkpoint weight to F32, picking the codec from the bound quantization recipe.</summary>
/// <remarks>For a reference path and for loading small tensors; it materializes the full matrix, so large weights should be decoded by row window with the codecs directly.</remarks>
public static unsafe class WeightDequantizer
{
    /// <summary>Returns <c>[rows, cols]</c> row-major F32 values of <paramref name="weight"/>.</summary>
    /// <param name="weight">The stored tensor; a borrowed view is fine and is not retained.</param>
    /// <param name="quant">The weight's bound recipe, or null when it is stored unquantized (F32, BF16 or F16).</param>
    /// <exception cref="NotSupportedException">The recipe's encoding has no host codec wired here.</exception>
    public static float[] ToF32(Tensor weight, QuantWeightInfo? quant)
    {
        ArgumentNullException.ThrowIfNull(weight);
        QuantRecipe? recipe = quant?.Recipe;
        if (recipe is null) return UnquantizedToF32(weight);

        long byteCount = weight.DType.ComputeByteCount(weight.ElementCount);
        ReadOnlySpan<byte> packed = new(weight.DataPointer, checked((int)byteCount));
        float[] dest = new float[checked((int)(recipe.LogicalRows * recipe.LogicalCols))];
        switch (recipe.Encoding)
        {
            case QuantEncoding.Fp8E4M3BlockE8M0:
                Fp8BlockE8M0Codec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dest);
                break;
            case QuantEncoding.Mxfp4E8M0:
                Mxfp4E8M0Codec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dest);
                break;
            case QuantEncoding.Nvfp4:
                ModelOptNvfp4Codec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dest);
                break;
            case QuantEncoding.AffineInt4:
            case QuantEncoding.AffineInt8:
                AffineIntCodec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dest);
                break;
            case QuantEncoding.Exl3Trellis:
                Exl3Codec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dest);
                break;
            default:
                throw new NotSupportedException($"No host dequantizer is wired for {recipe.Encoding}.");
        }
        return dest;
    }

    private static float[] UnquantizedToF32(Tensor weight)
    {
        if (weight.DType == DType.F32) return weight.AsReadOnlySpan<float>().ToArray();
        using Tensor f32 = weight.CastTo(DType.F32);
        return f32.AsReadOnlySpan<float>().ToArray();
    }
}
