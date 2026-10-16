using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Host dequantizer for FP8 E4M3 weights with F32 block scales, the DeepSeek-V3 / Qwen <c>fp8</c> layout that Fish Audio's
/// <c>bsgemm-moe</c> experts also use: <c>w[r, c] = e4m3(q[r, c]) * weight_scale_inv[r / blockRows, c / blockCols]</c>, the scale
/// grid <c>[ceil(rows / blockRows), ceil(cols / blockCols)]</c>.</summary>
public static unsafe class Fp8BlockF32Codec
{
    /// <summary>The block edge of every checkpoint this layout is known in.</summary>
    public const int DefaultBlock = 128;

    /// <summary>Dequantizes <paramref name="weight"/> (<c>[rows, cols]</c> F8E4M3) with <paramref name="scaleInv"/> (F32 or BF16 scale grid)
    /// into a new F32 tensor the caller owns.</summary>
    public static Tensor Dequantize(Tensor weight, Tensor scaleInv, int blockRows = DefaultBlock, int blockCols = DefaultBlock)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(scaleInv);
        if (weight.DType != DType.F8E4M3 || weight.Shape.Rank != 2)
        {
            throw new ArgumentException($"A block-FP8 weight is a rank-2 F8_E4M3 tensor, not {weight.Shape} {weight.DType.Name}.", nameof(weight));
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockCols);
        long rows = weight.Shape[0], cols = weight.Shape[1];
        long gridRows = (rows + blockRows - 1) / blockRows, gridCols = (cols + blockCols - 1) / blockCols;
        if (scaleInv.Shape.Rank != 2 || scaleInv.Shape[0] != gridRows || scaleInv.Shape[1] != gridCols)
        {
            throw new ArgumentException(
                $"A [{rows}, {cols}] FP8 weight with {blockRows}x{blockCols} blocks needs a [{gridRows}, {gridCols}] scale, not {scaleInv.Shape}.",
                nameof(scaleInv));
        }
        Tensor scales = scaleInv.DType == DType.F32 ? scaleInv : scaleInv.CastTo(DType.F32);
        Tensor output = new(new TensorShape(rows, cols), DType.F32);
        try
        {
            float[] e4m3 = CheckpointConvertUtils.E4M3Table;
            nint src = (nint)weight.DataPointer, dst = (nint)output.DataPointer, scale = (nint)scales.DataPointer;
            CpuParallel.For((int)rows, rows * cols * 2, r =>
            {
                byte* rowSrc = (byte*)src + (long)r * cols;
                float* rowDst = (float*)dst + (long)r * cols;
                float* rowScale = (float*)scale + (r / blockRows) * gridCols;
                for (long c0 = 0, block = 0; c0 < cols; c0 += blockCols, block++)
                {
                    float s = rowScale[block];
                    long end = Math.Min(c0 + blockCols, cols);
                    for (long c = c0; c < end; c++) rowDst[c] = e4m3[rowSrc[c]] * s;
                }
            });
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
        finally
        {
            if (!ReferenceEquals(scales, scaleInv)) scales.Dispose();
        }
    }
}
