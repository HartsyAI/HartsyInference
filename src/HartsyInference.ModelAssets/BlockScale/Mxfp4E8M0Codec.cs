using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.Mxfp4;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Host dequantizer for E2M1 weights packed two per byte with F8E8M0 scales per 32 inputs: <c>w = fp4(nibble) * 2^(scale - 127)</c> (DeepSeek-V4.1 routed experts, stored as I8).</summary>
/// <remarks>Low nibble is the even element, high nibble the odd one, as in the reference convert.py. Uses <see cref="Mxfp4Codec.Fp4Lut"/> except that nibble 8 decodes to +0.0 like the reference table.</remarks>
public static unsafe class Mxfp4E8M0Codec
{
    // The reference FP4_TABLE has +0.0 at nibble 8, where Mxfp4Codec.Fp4Lut has -0.0.
    private static readonly float[] Lut = ZeroPositiveLut();

    /// <summary>A fresh copy of the E2M1 table with nibble 8 as +0.0, the convention of every V4.1 derivative's reference decoder.</summary>
    internal static float[] ZeroPositiveLut()
    {
        float[] lut = (float[])Mxfp4Codec.Fp4Lut.Clone();
        lut[8] = 0.0f;
        return lut;
    }

    /// <summary>Dequantizes <paramref name="rowCount"/> rows from <paramref name="rowOffset"/> to F32.</summary>
    /// <param name="packed">The recipe's whole packed matrix, <c>LogicalRows * LogicalCols / 2</c> bytes.</param>
    /// <param name="recipe">Encoding <see cref="QuantEncoding.Mxfp4E8M0"/> with a row-major E8M0 scale.</param>
    /// <param name="rowOffset">First row of the recipe to decode.</param>
    /// <param name="rowCount">Rows to decode.</param>
    /// <param name="dest">Receives <c>rowCount * LogicalCols</c> floats.</param>
    public static void DequantRows(ReadOnlySpan<byte> packed, QuantRecipe recipe, long rowOffset, long rowCount, Span<float> dest)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        long scaleStride = BlockScaleCodecChecks.Validate(recipe, QuantEncoding.Mxfp4E8M0, packed.Length, rowOffset, rowCount, dest.Length);
        if (rowCount == 0) return;

        long cols = recipe.LogicalCols;
        long rowBytes = cols / 2;
        int blockCols = recipe.Geometry.BlockCols;
        if (blockCols % 2 != 0 || cols % 2 != 0)
            throw new NotSupportedException($"Mxfp4 needs even block width and columns; got {recipe.Geometry} over {cols} columns.");
        long bytesPerBlock = blockCols / 2;
        int blockRows = recipe.Geometry.BlockRows;
        long scaleColOffset = recipe.ScaleColOffset;
        float[] lut = Lut;
        float[] e8m0 = E8M0Scale.Table;
        byte* scaleBase = (byte*)recipe.Scale!.DataPointer;

        fixed (byte* src = packed)
        fixed (float* dst = dest)
        {
            nint srcAddr = (nint)src, dstAddr = (nint)dst, scaleAddr = (nint)scaleBase;
            Parallel.For(0, (int)rowCount, r =>
            {
                long row = rowOffset + r;
                byte* rowSrc = (byte*)srcAddr + row * rowBytes;
                float* rowDst = (float*)dstAddr + (long)r * cols;
                byte* rowScale = (byte*)scaleAddr + (row / blockRows) * scaleStride + scaleColOffset;
                for (long blk = 0, j0 = 0; j0 < rowBytes; blk++, j0 += bytesPerBlock)
                {
                    float scale = e8m0[rowScale[blk]];
                    long end = Math.Min(j0 + bytesPerBlock, rowBytes);
                    for (long j = j0; j < end; j++)
                    {
                        byte b = rowSrc[j];
                        rowDst[2 * j] = lut[b & 0x0F] * scale;
                        rowDst[2 * j + 1] = lut[b >> 4] * scale;
                    }
                }
            });
        }
    }
}
