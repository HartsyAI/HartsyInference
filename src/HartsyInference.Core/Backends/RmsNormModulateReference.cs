using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Host reference for <see cref="IBackend.RmsNormModulate"/>:
/// <c>out[r,d] = x[r,d]·rsqrt(mean(x[r,:]²) + eps)·weight[d]·(1 + scale[d]) + shift[d]</c>.</summary>
public static class RmsNormModulateReference
{
    /// <summary>Validates the operands and returns the normalized width, the row count and the rows per modulation row.</summary>
    public static (int Dim, long Rows, long RowsPerBatch) Validate(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? shift)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(scale);
        if ((input.DType != DType.F32 && input.DType != DType.F16) || (output.DType != DType.F32 && output.DType != DType.F16))
            throw new NotSupportedException($"RmsNormModulate takes F32 or F16 activations, got input={input.DType}, output={output.DType}.");
        if (weight.DType != DType.F32 || scale.DType != DType.F32 || (shift is not null && shift.DType != DType.F32))
            throw new NotSupportedException("RmsNormModulate needs F32 weight, scale and shift.");
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        if (dim <= 0 || output.ElementCount != input.ElementCount || weight.ElementCount != dim)
            throw new ArgumentException($"RmsNormModulate input {input.Shape}, output {output.Shape} and weight {weight.Shape} disagree.", nameof(output));
        long rows = input.ElementCount / dim;
        long modRows = scale.ElementCount / dim;
        if (modRows <= 0 || scale.ElementCount != modRows * dim || rows % modRows != 0
            || (shift is not null && shift.ElementCount != scale.ElementCount))
        {
            throw new ArgumentException(
                $"RmsNormModulate scale {scale.Shape} (shift {shift?.Shape.ToString() ?? "none"}) must be [{dim}] or one row per batch of {input.Shape}.",
                nameof(scale));
        }
        return (dim, rows, rows / modRows);
    }

    /// <summary>Applies the fused norm and modulation on the host, accumulating each row's mean square in F64.</summary>
    public static unsafe void Apply(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? shift, float eps)
    {
        (int dim, long rows, long rowsPerBatch) = Validate(output, input, weight, scale, shift);
        void* src = (void*)input.DataPointer;
        void* dst = (void*)output.DataPointer;
        float* w = (float*)weight.DataPointer;
        float* sc = (float*)scale.DataPointer;
        float* sh = shift is null ? null : (float*)shift.DataPointer;
        bool inHalf = input.DType == DType.F16;
        bool outHalf = output.DType == DType.F16;
        Parallel.For(0L, rows, row =>
        {
            long at = row * dim;
            double sumSq = 0;
            for (int d = 0; d < dim; d++)
            {
                double v = Read(src, inHalf, at + d);
                sumSq += v * v;
            }
            float inv = (float)(1.0 / Math.Sqrt(sumSq / dim + eps));
            long mod = row / rowsPerBatch * dim;
            for (int d = 0; d < dim; d++)
            {
                float v = Read(src, inHalf, at + d) * inv * w[d] * (1f + sc[mod + d]);
                if (sh != null) v += sh[mod + d];
                if (outHalf) ((Half*)dst)[at + d] = (Half)v;
                else ((float*)dst)[at + d] = v;
            }
        });
    }

    private static unsafe float Read(void* data, bool half, long index) =>
        half ? (float)((Half*)data)[index] : ((float*)data)[index];
}
