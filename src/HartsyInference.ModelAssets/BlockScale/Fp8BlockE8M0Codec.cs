using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Host dequantizer for FP8 E4M3 weights with F8E8M0 block scales: <c>w = e4m3(q) * 2^(scale - 127)</c> (DeepSeek-V4.1 dense and shared-expert weights).</summary>
public static unsafe class Fp8BlockE8M0Codec
{
    /// <summary>Dequantizes <paramref name="rowCount"/> rows from <paramref name="rowOffset"/> to F32.</summary>
    /// <param name="packed">The recipe's whole <c>[LogicalRows, LogicalCols]</c> FP8 matrix, one byte per element.</param>
    /// <param name="recipe">Encoding <see cref="QuantEncoding.Fp8E4M3BlockE8M0"/> with a row-major E8M0 scale.</param>
    /// <param name="rowOffset">First row of the recipe to decode.</param>
    /// <param name="rowCount">Rows to decode.</param>
    /// <param name="dest">Receives <c>rowCount * LogicalCols</c> floats.</param>
    public static void DequantRows(ReadOnlySpan<byte> packed, QuantRecipe recipe, long rowOffset, long rowCount, Span<float> dest)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        long scaleStride = BlockScaleCodecChecks.Validate(recipe, QuantEncoding.Fp8E4M3BlockE8M0, packed.Length, rowOffset, rowCount, dest.Length);
        if (rowCount == 0) return;

        long cols = recipe.LogicalCols;
        int blockRows = recipe.Geometry.BlockRows;
        int blockCols = recipe.Geometry.BlockCols;
        long scaleColOffset = recipe.ScaleColOffset;
        float[] e4m3 = CheckpointConvertUtils.E4M3Table;
        float[] e8m0 = E8M0Scale.Table;
        byte* scaleBase = (byte*)recipe.Scale!.DataPointer;

        fixed (byte* src = packed)
        fixed (float* dst = dest)
        {
            nint srcAddr = (nint)src, dstAddr = (nint)dst, scaleAddr = (nint)scaleBase;
            Parallel.For(0, (int)rowCount, r =>
            {
                long row = rowOffset + r;
                byte* rowSrc = (byte*)srcAddr + row * cols;
                float* rowDst = (float*)dstAddr + (long)r * cols;
                byte* rowScale = (byte*)scaleAddr + (row / blockRows) * scaleStride + scaleColOffset;
                for (long c0 = 0; c0 < cols; c0 += blockCols)
                {
                    float scale = e8m0[rowScale[c0 / blockCols]];
                    long end = Math.Min(c0 + blockCols, cols);
                    for (long c = c0; c < end; c++)
                        rowDst[c] = e4m3[rowSrc[c]] * scale;
                }
            });
        }
    }
}
