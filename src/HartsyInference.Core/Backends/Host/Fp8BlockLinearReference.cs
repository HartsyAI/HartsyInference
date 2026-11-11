using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>CPU reference for <see cref="IBackend.Fp8BlockLinear"/>: a linear over an FP8 e4m3 weight with F32 block scales.</summary>
/// <remarks>Each output row decodes its weight once into a float row, <c>e4m3(q) * scale</c> rounded to float, then the dot
/// products accumulate in double and round once, so the result equals dequantizing the weight and running a plain GEMM.</remarks>
public static class Fp8BlockLinearReference
{
    /// <summary>The weight decoded to a new F32 <c>[N, K]</c> tensor, each element <c>e4m3(q) * scale</c> as <see cref="Apply"/> decodes it.
    /// Rows decode in parallel, so the CPU linear can run a vectorized GEMM over the result. The caller owns the tensor.</summary>
    public static unsafe Tensor DequantizeF32(Tensor weight, Tensor weightScaleInv, int blockRows, int blockCols)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(weightScaleInv);
        if (weight.DType != DType.F8E4M3 || weight.Shape.Rank != 2)
            throw new ArgumentException($"A block-FP8 weight is a rank-2 F8_E4M3 tensor, not {weight.Shape} {weight.DType.Name}.", nameof(weight));
        int n = checked((int)weight.Shape[0]), k = checked((int)weight.Shape[1]);
        int gridRows = (n + blockRows - 1) / blockRows, gridCols = (k + blockCols - 1) / blockCols;
        if (weightScaleInv.DType != DType.F32 || weightScaleInv.Shape.Rank != 2 || weightScaleInv.Shape[0] != gridRows || weightScaleInv.Shape[1] != gridCols)
            throw new ArgumentException($"A [{n}, {k}] FP8 weight with {blockRows}x{blockCols} blocks needs an F32 [{gridRows}, {gridCols}] scale.", nameof(weightScaleInv));
        Tensor dense = new(new TensorShape(n, k), DType.F32);
        try
        {
            nint src = (nint)weight.DataPointer, dst = (nint)dense.DataPointer, scale = (nint)weightScaleInv.DataPointer;
            CpuParallel.For(n, (long)n * k, j =>
            {
                byte* srcRow = (byte*)src + (long)j * k;
                float* dstRow = (float*)dst + (long)j * k;
                float* scaleRow = (float*)scale + (long)(j / blockRows) * gridCols;
                for (int c = 0; c < k; c++) dstRow[c] = LatentCodec.DecodeE4M3(srcRow[c]) * scaleRow[c / blockCols];
            });
            return dense;
        }
        catch
        {
            dense.Dispose();
            throw;
        }
    }

    /// <summary>Checks the operands of <see cref="Apply"/> and returns the rows M, the outputs N and the inputs K.</summary>
    public static (int M, int N, int K) Validate(Tensor output, Tensor input, Tensor weight, Tensor weightScaleInv, Tensor? bias,
        int blockRows, int blockCols)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockCols);
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("Fp8BlockLinear supports F32 input and output only.");
        if (weight.DType != DType.F8E4M3 || weight.Shape.Rank != 2)
            throw new ArgumentException($"Fp8BlockLinear needs a rank-2 F8_E4M3 weight, not {weight.Shape} {weight.DType.Name}.", nameof(weight));
        if (weightScaleInv.DType != DType.F32)
            throw new NotSupportedException($"Fp8BlockLinear needs an F32 weight_scale_inv grid, not {weightScaleInv.DType.Name}.");
        long n = weight.Shape[0], k = weight.Shape[1];
        long gridRows = (n + blockRows - 1) / blockRows, gridCols = (k + blockCols - 1) / blockCols;
        if (weightScaleInv.Shape.Rank != 2 || weightScaleInv.Shape[0] != gridRows || weightScaleInv.Shape[1] != gridCols)
            throw new ArgumentException(
                $"A [{n}, {k}] weight with {blockRows}x{blockCols} blocks needs a [{gridRows}, {gridCols}] scale, not {weightScaleInv.Shape}.",
                nameof(weightScaleInv));
        if (input.Shape.Rank < 1 || input.Shape[input.Shape.Rank - 1] != k)
            throw new ArgumentException($"Fp8BlockLinear input must end in K={k}; got {input.Shape}.", nameof(input));
        if (k == 0) throw new ArgumentException("Fp8BlockLinear needs K > 0.", nameof(weight));
        long m = input.ElementCount / k;
        if (m * k > int.MaxValue) throw new NotSupportedException("Fp8BlockLinear input exceeds 32-bit indexing.");
        if (output.ElementCount != m * n || output.Shape.Rank < 1 || output.Shape[output.Shape.Rank - 1] != n)
            throw new ArgumentException($"Fp8BlockLinear output must be [..., {n}] with {m} rows; got {output.Shape}.", nameof(output));
        if (bias is not null && (bias.DType != DType.F32 || bias.ElementCount != n))
            throw new ArgumentException($"Fp8BlockLinear bias must be F32 with {n} elements.", nameof(bias));
        if (m * n > int.MaxValue || n * k > int.MaxValue) throw new NotSupportedException("Fp8BlockLinear operands exceed int indexing.");
        return ((int)m, (int)n, (int)k);
    }

    /// <summary>Computes <c>output = input @ dequant(weight)^T (+ bias)</c>; the input and output must not alias.</summary>
    public static unsafe void Apply(Tensor output, Tensor input, Tensor weight, Tensor weightScaleInv, Tensor? bias,
        int blockRows, int blockCols)
    {
        (int m, int n, int k) = Validate(output, input, weight, weightScaleInv, bias, blockRows, blockCols);
        float* x = (float*)input.DataPointer, y = (float*)output.DataPointer;
        byte* w = (byte*)weight.DataPointer;
        float* scale = (float*)weightScaleInv.DataPointer, b = bias is null ? null : (float*)bias.DataPointer;
        int gridCols = (k + blockCols - 1) / blockCols;
        float[] row = new float[k];
        for (int j = 0; j < n; j++)
        {
            float* scaleRow = scale + (long)(j / blockRows) * gridCols;
            for (int c = 0; c < k; c++) row[c] = LatentCodec.DecodeE4M3(w[(long)j * k + c]) * scaleRow[c / blockCols];
            for (int i = 0; i < m; i++)
            {
                float* xi = x + (long)i * k;
                double sum = b is null ? 0.0 : b[j];
                for (int c = 0; c < k; c++) sum += (double)xi[c] * row[c];
                y[(long)i * n + j] = (float)sum;
            }
        }
    }
}
