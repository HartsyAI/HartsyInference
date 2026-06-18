using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

/// <summary>Per-row symmetric INT8 quantization, the standard scheme for feeding <see cref="VulkanBackend.MatMulInt8"/>.
/// For a [rows, K] F32 tensor it computes one scale per row (<c>scale[r] = max_k|x[r][k]| / 127</c>) and stores
/// <c>q[r][k] = round(x[r][k] / scale[r])</c> clamped to [-127, 127] as INT8. Works for both weights ([N, K], one
/// scale per output channel) and activations ([M, K], one scale per token). Dequantization is <c>q * scale</c>,
/// which the GEMM folds in after the exact int32 accumulation.</summary>
public static class Int8Quantizer
{
    /// <summary>Quantizes a 2-D F32 tensor to per-row symmetric INT8. Returns the INT8 tensor (same shape, I8) and
    /// the F32 per-row scale tensor (length = rows). The caller owns and disposes both.</summary>
    public static unsafe (Tensor quantized, Tensor scale) RowwiseSymmetric(Tensor src)
    {
        if (src is null) throw new ArgumentNullException(nameof(src));
        if (src.DType != DType.F32)
            throw new ArgumentException($"RowwiseSymmetric requires an F32 source; got {src.DType.Name}.");
        if (src.Shape.Rank != 2)
            throw new ArgumentException($"RowwiseSymmetric requires a 2-D tensor; got rank {src.Shape.Rank}.");

        int rows = (int)src.Shape[0];
        int k = (int)src.Shape[1];
        Tensor quantized = new Tensor(src.Shape, DType.I8);
        Tensor scale = new Tensor(new TensorShape(rows), DType.F32);

        float* sp = (float*)src.DataPointer;
        sbyte* qp = (sbyte*)quantized.DataPointer;
        float* scp = (float*)scale.DataPointer;

        for (int r = 0; r < rows; r++)
        {
            int rowBase = r * k;
            float maxAbs = 0f;
            for (int i = 0; i < k; i++)
            {
                float v = MathF.Abs(sp[rowBase + i]);
                if (v > maxAbs) maxAbs = v;
            }
            // An all-zero row has no scale; use 1.0 so the (all-zero) quantized values dequant back to zero.
            float s = maxAbs > 0f ? maxAbs / 127f : 1f;
            scp[r] = s;
            float inv = 1f / s;
            for (int i = 0; i < k; i++)
            {
                int qi = (int)MathF.Round(sp[rowBase + i] * inv);
                qi = Math.Clamp(qi, -127, 127);
                qp[rowBase + i] = (sbyte)qi;
            }
        }
        return (quantized, scale);
    }
}
