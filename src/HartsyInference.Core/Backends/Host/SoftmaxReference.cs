using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>Host reference for <see cref="IBackend.SoftmaxRows"/>: a numerically stable softmax over the last axis of an F32
/// <c>[rows, cols]</c> tensor, each row shifted by its maximum before the exponential.</summary>
public static unsafe class SoftmaxReference
{
    /// <summary>Writes <c>exp(x − max) / Σ exp(x − max)</c> per row of <paramref name="input"/> into <paramref name="output"/>.</summary>
    public static void Rows(Tensor output, Tensor input, int rows, int cols)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException($"SoftmaxRows takes F32 tensors only, got {input.DType} and {output.DType}.");
        if (rows < 1 || cols < 1)
            throw new ArgumentException($"SoftmaxRows needs positive rows and cols, got rows={rows}, cols={cols}.");
        long count = (long)rows * cols;
        if (input.ElementCount != count || output.ElementCount != count)
            throw new ArgumentException($"SoftmaxRows input {input.Shape} and output {output.Shape} must both hold {rows}×{cols} values.", nameof(input));
        float* x = (float*)input.DataPointer;
        float* y = (float*)output.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            float* xr = x + (long)r * cols;
            float* yr = y + (long)r * cols;
            float max = float.NegativeInfinity;
            for (int c = 0; c < cols; c++) max = MathF.Max(max, xr[c]);
            float sum = 0f;
            for (int c = 0; c < cols; c++)
            {
                yr[c] = MathF.Exp(xr[c] - max);
                sum += yr[c];
            }
            float inv = 1f / sum;
            for (int c = 0; c < cols; c++) yr[c] *= inv;
        }
    }
}
