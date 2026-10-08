using HartsyInference.Core.Numerics;
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
        if (recipe is null)
        {
            if (quant is not null) throw new NotSupportedException($"Quantization format '{quant.Format}' carries no block-scale recipe; refusing to read it as plain values.");
            return UnquantizedToF32(weight);
        }
        float[] dest = new float[checked((int)(recipe.LogicalRows * recipe.LogicalCols))];
        ToF32Rows(weight, quant, 0, recipe.LogicalRows, dest);
        return dest;
    }

    /// <summary>Decodes <paramref name="rowCount"/> rows from <paramref name="rowOffset"/> of a two-dimensional weight to F32, so a large matrix can be read through a small window.</summary>
    /// <param name="weight">The stored tensor; borrowed, not retained.</param>
    /// <param name="quant">The bound recipe, or null for an unquantized F32, BF16 or F16 matrix.</param>
    /// <param name="rowOffset">First logical row.</param>
    /// <param name="rowCount">Rows to decode.</param>
    /// <param name="dest">Receives <c>rowCount * cols</c> floats.</param>
    /// <exception cref="NotSupportedException">The encoding or element type has no host reader.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The window leaves the matrix or <paramref name="dest"/> is the wrong size.</exception>
    public static void ToF32Rows(Tensor weight, QuantWeightInfo? quant, long rowOffset, long rowCount, Span<float> dest)
    {
        ArgumentNullException.ThrowIfNull(weight);
        QuantRecipe? recipe = quant?.Recipe;
        if (recipe is null)
        {
            if (quant is not null) throw new NotSupportedException($"Quantization format '{quant.Format}' carries no block-scale recipe; refusing to read it as plain values.");
            UnquantizedRows(weight, rowOffset, rowCount, dest);
            return;
        }

        long byteCount = weight.DType.ComputeByteCount(weight.ElementCount);
        ReadOnlySpan<byte> packed = new(weight.DataPointer, checked((int)byteCount));
        switch (recipe.Encoding)
        {
            case QuantEncoding.Fp8E4M3BlockE8M0:
                Fp8BlockE8M0Codec.DequantRows(packed, recipe, rowOffset, rowCount, dest);
                break;
            case QuantEncoding.Mxfp4E8M0:
                Mxfp4E8M0Codec.DequantRows(packed, recipe, rowOffset, rowCount, dest);
                break;
            case QuantEncoding.Nvfp4:
                ModelOptNvfp4Codec.DequantRows(packed, recipe, rowOffset, rowCount, dest);
                break;
            case QuantEncoding.AffineInt4:
            case QuantEncoding.AffineInt8:
                AffineIntCodec.DequantRows(packed, recipe, rowOffset, rowCount, dest);
                break;
            case QuantEncoding.Exl3Trellis:
                Exl3Codec.DequantRows(packed, recipe, rowOffset, rowCount, dest);
                break;
            default:
                throw new NotSupportedException($"No host dequantizer is wired for {recipe.Encoding}.");
        }
    }

    private static void UnquantizedRows(Tensor weight, long rowOffset, long rowCount, Span<float> dest)
    {
        if (weight.Shape.Rank != 2) throw new NotSupportedException($"An unquantized weight must be a matrix to be read by rows; this one has rank {weight.Shape.Rank}.");
        long rows = weight.Shape[0], cols = weight.Shape[1];
        if (rowOffset < 0 || rowCount < 0 || rowOffset + rowCount > rows)
            throw new ArgumentOutOfRangeException(nameof(rowOffset), $"Rows [{rowOffset}, {rowOffset + rowCount}) leave a {rows}-row matrix.");
        if (dest.Length != rowCount * cols) throw new ArgumentOutOfRangeException(nameof(dest), $"dest holds {dest.Length} values, expected {rowCount * cols}.");
        long first = rowOffset * cols;
        if (weight.DType == DType.F32)
        {
            weight.AsReadOnlySpan<float>().Slice((int)first, dest.Length).CopyTo(dest);
        }
        else if (weight.DType == DType.BF16)
        {
            // a head or embedding window is hundreds of millions of values per token, so convert rows in parallel
            ushort* bits = (ushort*)weight.DataPointer + first;
            fixed (float* dst = dest)
            {
                nint src = (nint)bits, dstAddr = (nint)dst;
                CpuParallel.For((int)rowCount, rowCount * cols, r =>
                {
                    ushort* rowSrc = (ushort*)src + r * cols;
                    float* rowDst = (float*)dstAddr + r * cols;
                    for (long c = 0; c < cols; c++) rowDst[c] = BitConverter.UInt32BitsToSingle((uint)rowSrc[c] << 16);
                });
            }
        }
        else if (weight.DType == DType.F16)
        {
            ReadOnlySpan<Half> halves = weight.AsReadOnlySpan<Half>().Slice((int)first, dest.Length);
            for (int i = 0; i < dest.Length; i++) dest[i] = (float)halves[i];
        }
        else
        {
            throw new NotSupportedException($"A {weight.DType} weight without a quantization recipe cannot be read as plain values.");
        }
    }

    private static float[] UnquantizedToF32(Tensor weight)
    {
        if (weight.DType != DType.F32 && weight.DType != DType.BF16 && weight.DType != DType.F16)
            throw new NotSupportedException($"A {weight.DType} weight without a quantization recipe cannot be read as plain values.");
        if (weight.DType == DType.F32) return weight.AsReadOnlySpan<float>().ToArray();
        using Tensor f32 = weight.CastTo(DType.F32);
        return f32.AsReadOnlySpan<float>().ToArray();
    }
}
