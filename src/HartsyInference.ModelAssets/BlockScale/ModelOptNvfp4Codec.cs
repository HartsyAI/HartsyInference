using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Mxfp4;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Host dequantizer for ModelOpt NVFP4: <c>w = e2m1(nibble) * (e4m3(scale) * weight_scale_2)</c>, low nibble first, one E4M3 scale per 16.</summary>
/// <remarks>Not <see cref="Nvfp4.Nvfp4Codec"/>, which reads the Comfy layout. Nibble 8 decodes to +0.0; the input scale is not read and a zero, NaN or inf global scale decodes as written.</remarks>
public static unsafe class ModelOptNvfp4Codec
{
    private static readonly DType[] ScaleTypes = [DType.F8E4M3];
    private static readonly float[] Lut = Mxfp4E8M0Codec.ZeroPositiveLut();

    /// <summary>Dequantizes <paramref name="rowCount"/> rows from <paramref name="rowOffset"/> to F32.</summary>
    /// <param name="packed">The recipe's whole packed matrix, <c>LogicalRows * LogicalCols / 2</c> bytes.</param>
    /// <param name="recipe">Encoding <see cref="QuantEncoding.Nvfp4"/> with a row-major E4M3 scale and a scalar F32 global scale.</param>
    /// <param name="rowOffset">First row of the recipe to decode.</param>
    /// <param name="rowCount">Rows to decode.</param>
    /// <param name="dest">Receives <c>rowCount * LogicalCols</c> floats.</param>
    public static void DequantRows(ReadOnlySpan<byte> packed, QuantRecipe recipe, long rowOffset, long rowCount, Span<float> dest)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        long scaleStride = BlockScaleCodecChecks.Validate(recipe, QuantEncoding.Nvfp4, ScaleTypes, packed.Length, rowOffset, rowCount, dest.Length);
        Tensor global = recipe.GlobalScale ?? throw new HartsyInferenceException("Nvfp4 recipe has no global scale (weight_scale_2).");
        if (global.DType != DType.F32 || global.ElementCount != 1)
            throw new HartsyInferenceException($"Nvfp4 global scale must be a scalar F32; got {global.DType} {global.Shape}.");
        if (rowCount == 0) return;

        long cols = recipe.LogicalCols;
        int blockCols = recipe.Geometry.BlockCols;
        if (blockCols % 2 != 0 || cols % 2 != 0)
            throw new NotSupportedException($"Nvfp4 needs even block width and columns; got {recipe.Geometry} over {cols} columns.");
        long rowBytes = cols / 2;
        long bytesPerBlock = blockCols / 2;
        int blockRows = recipe.Geometry.BlockRows;
        long scaleColOffset = recipe.ScaleColOffset;
        float globalScale = *(float*)global.DataPointer;
        float[] lut = Lut;
        float[] e4m3 = CheckpointConvertUtils.E4M3Table;
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
                    float scale = e4m3[rowScale[blk]] * globalScale;
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
