using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Host dequantizer for MLX affine weights: <c>w = q * scale + bias</c> per column group, q an unsigned 4- or 8-bit field.</summary>
/// <remarks>MLX packs each U32 word lowest field first. The product is rounded to F32 before the bias is added, as <c>mx.dequantize</c> does.</remarks>
public static unsafe class AffineIntCodec
{
    private static readonly DType[] ScaleTypes = [DType.F32];

    /// <summary>Dequantizes <paramref name="rowCount"/> rows from <paramref name="rowOffset"/> to F32.</summary>
    /// <param name="packed">The recipe's whole packed matrix as bytes: <c>LogicalRows * LogicalCols / ElementsPerByte</c> of them.</param>
    /// <param name="recipe">Encoding <see cref="QuantEncoding.AffineInt4"/> or <see cref="QuantEncoding.AffineInt8"/> with F32 scale and bias of one shape.</param>
    /// <param name="rowOffset">First row of the recipe to decode.</param>
    /// <param name="rowCount">Rows to decode.</param>
    /// <param name="dest">Receives <c>rowCount * LogicalCols</c> floats.</param>
    public static void DequantRows(ReadOnlySpan<byte> packed, QuantRecipe recipe, long rowOffset, long rowCount, Span<float> dest)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        if (recipe.Encoding is not (QuantEncoding.AffineInt4 or QuantEncoding.AffineInt8))
            throw new HartsyInferenceException($"Recipe encoding is {recipe.Encoding}, this codec decodes AffineInt4 or AffineInt8.");
        long scaleStride = BlockScaleCodecChecks.Validate(recipe, recipe.Encoding, ScaleTypes, packed.Length, rowOffset, rowCount, dest.Length);
        Tensor bias = recipe.Bias ?? throw new HartsyInferenceException($"{recipe.Encoding} recipe has no bias tensor.");
        Tensor scale = recipe.Scale!;
        if (bias.DType != recipe.ScaleDType || bias.Shape != scale.Shape)
            throw new HartsyInferenceException($"{recipe.Encoding} bias must match the scale ({scale.DType} {scale.Shape}); got {bias.DType} {bias.Shape}.");
        if (rowCount == 0) return;

        bool fourBit = recipe.Encoding == QuantEncoding.AffineInt4;
        long cols = recipe.LogicalCols;
        if (fourBit && cols % 2 != 0)
            throw new NotSupportedException($"AffineInt4 needs an even column count; got {cols}.");
        long rowBytes = fourBit ? cols / 2 : cols;
        int group = recipe.Geometry.BlockCols;
        int blockRows = recipe.Geometry.BlockRows;
        long scaleColOffset = recipe.ScaleColOffset;
        float* scaleBase = (float*)scale.DataPointer;
        float* biasBase = (float*)bias.DataPointer;

        fixed (byte* src = packed)
        fixed (float* dst = dest)
        {
            nint srcAddr = (nint)src, dstAddr = (nint)dst, scaleAddr = (nint)scaleBase, biasAddr = (nint)biasBase;
            Parallel.For(0, (int)rowCount, r =>
            {
                long row = rowOffset + r;
                byte* rowSrc = (byte*)srcAddr + row * rowBytes;
                float* rowDst = (float*)dstAddr + (long)r * cols;
                long groupRow = (row / blockRows) * scaleStride + scaleColOffset;
                float* rowScale = (float*)scaleAddr + groupRow;
                float* rowBias = (float*)biasAddr + groupRow;
                for (long c = 0; c < cols; c++)
                {
                    int q = fourBit ? (c & 1) == 0 ? rowSrc[c >> 1] & 0x0F : rowSrc[c >> 1] >> 4 : rowSrc[c];
                    long g = c / group;
                    float product = q * rowScale[g];
                    rowDst[c] = product + rowBias[g];
                }
            });
        }
    }
}
