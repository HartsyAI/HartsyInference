using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Numerics;
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
            CpuParallel.For((int)rowCount, rowCount * cols * 2, r =>
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

    /// <summary>Quantizes a row-major F32 matrix to ModelOpt NVFP4: one E4M3 scale per 16 consecutive elements of a row and one F32 global scale (<c>amax / (6 * 448)</c>).</summary>
    /// <returns>Low-nibble-first packed e2m1 (<c>rows * cols / 2</c> bytes), row-major E4M3 scale codes (<c>rows * cols / 16</c> bytes) and the global scale. Decoding with <see cref="DequantRows"/> reproduces each element within <see cref="MaxRoundTripError"/> of its block maximum.</returns>
    public static (byte[] Packed, byte[] Scales, float GlobalScale) Quantize(ReadOnlySpan<float> values, long rows, long cols)
    {
        if (cols % BlockWidth != 0)
            throw new ArgumentException($"Nvfp4 needs columns in multiples of {BlockWidth}; got {cols}.", nameof(cols));
        if (values.Length != rows * cols)
            throw new ArgumentException($"Expected {rows * cols} values for a {rows}x{cols} matrix; got {values.Length}.", nameof(values));

        float amax = 0f;
        foreach (float v in values)
            amax = MathF.Max(amax, MathF.Abs(v));
        float global = amax > 0f ? amax / (E2m1Max * E4M3Max) : 1f;

        long blocksPerRow = cols / BlockWidth;
        byte[] packed = new byte[rows * cols / 2];
        byte[] scales = new byte[rows * blocksPerRow];
        float[] e4m3 = CheckpointConvertUtils.E4M3Table;
        for (long r = 0; r < rows; r++)
        {
            for (long blk = 0; blk < blocksPerRow; blk++)
            {
                long start = r * cols + blk * BlockWidth;
                float blockMax = 0f;
                for (int j = 0; j < BlockWidth; j++)
                    blockMax = MathF.Max(blockMax, MathF.Abs(values[(int)start + j]));

                byte scaleCode = NearestE4M3Code(blockMax / E2m1Max / global);
                scales[r * blocksPerRow + blk] = scaleCode;
                float scale = e4m3[scaleCode] * global;
                float inv = scale > 0f ? 1f / scale : 0f;
                for (int j = 0; j < BlockWidth; j++)
                {
                    long i = start + j;
                    byte nibble = E2m1Nibble(values[(int)i] * inv);
                    packed[i >> 1] |= (byte)((i & 1) == 0 ? nibble : nibble << 4);
                }
            }
        }

        return (packed, scales, global);
    }

    /// <summary>Largest per-element error of <see cref="Quantize"/> then <see cref="DequantRows"/>, as a fraction of the block maximum: half the widest e2m1 step (1 scale unit = block maximum / 6) plus the E4M3 rounding of the block scale (1/16).</summary>
    public const float MaxRoundTripError = 1f / 6f + 1f / 16f;

    private const int BlockWidth = 16;
    private const float E2m1Max = 6f;
    private const float E4M3Max = 448f;
    private static readonly float[] PositiveE4M3Values;
    private static readonly byte[] PositiveE4M3Codes;

    static ModelOptNvfp4Codec()
    {
        float[] table = CheckpointConvertUtils.E4M3Table;
        PositiveE4M3Codes = [.. Enumerable.Range(0, 0x7F).Select(c => (byte)c)];
        PositiveE4M3Values = [.. PositiveE4M3Codes.Select(c => table[c])];
        Array.Sort(PositiveE4M3Values, PositiveE4M3Codes);
    }

    private static byte NearestE4M3Code(float magnitude)
    {
        int idx = Array.BinarySearch(PositiveE4M3Values, magnitude);
        if (idx >= 0)
            return PositiveE4M3Codes[idx];
        int insert = ~idx;
        if (insert >= PositiveE4M3Values.Length)
            return PositiveE4M3Codes[^1];
        if (insert == 0)
            return PositiveE4M3Codes[0];
        float below = magnitude - PositiveE4M3Values[insert - 1];
        float above = PositiveE4M3Values[insert] - magnitude;
        return below <= above ? PositiveE4M3Codes[insert - 1] : PositiveE4M3Codes[insert];
    }

    private static byte E2m1Nibble(float normalized)
    {
        float magnitude = MathF.Abs(normalized);
        int best = 0;
        float bestError = float.MaxValue;
        for (int m = 0; m < E2m1Magnitudes.Length; m++)
        {
            float error = MathF.Abs(magnitude - E2m1Magnitudes[m]);
            if (error < bestError)
            {
                bestError = error;
                best = m;
            }
        }
        if (best == 0)
            return 0;
        return (byte)(normalized < 0f ? 8 | best : best);
    }

    private static readonly float[] E2m1Magnitudes = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];
}
