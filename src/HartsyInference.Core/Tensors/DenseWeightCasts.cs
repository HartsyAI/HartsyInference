using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;

namespace HartsyInference.Core.Tensors;

/// <summary>Picks the storage a 16-bit dense projection weight keeps on the backend that will run it, so a BF16 checkpoint is
/// not widened to F32 where the backend can multiply it at 16 bits.</summary>
/// <remarks>
/// <para>CPU (or no backend yet): one F32 copy. The CPU GEMM casts any non-F32 weight on every call, so keeping BF16 there
/// trades a one-time copy for a cast per token.</para>
/// <para>A backend with BF16 (CUDA on Ampere or newer): kept as stored.</para>
/// <para>A backend without BF16 (Vulkan, pre-Ampere CUDA): BF16 is cast once on the host to F16. Vulkan would otherwise cast it
/// on the device and cache the F16 copy beside the BF16 original, two copies in VRAM. BF16 values outside F16's range would
/// become infinity, so the cast refuses with the weight's name instead.</para>
/// <para>Only floating dtypes are handled. Packed quantized weights go through the quantized-weight policy.</para>
/// </remarks>
public static class DenseWeightCasts
{
    /// <summary>The largest finite F16 value.</summary>
    public const float F16Max = 65504f;

    /// <summary>Returns <paramref name="weight"/> in the dtype <paramref name="backend"/> should hold it in. When the result is a new
    /// tensor the caller owns it; when it is <paramref name="weight"/> itself nothing was copied.</summary>
    /// <param name="weight">A floating-point dense weight.</param>
    /// <param name="backend">The backend that will run the weight, or null when none is chosen yet (treated as CPU).</param>
    /// <param name="key">The weight's checkpoint name, used in the overflow refusal.</param>
    /// <exception cref="NotSupportedException">A BF16 weight holds values F16 cannot represent and the backend has no BF16.</exception>
    public static Tensor ForBackend(Tensor weight, IBackend? backend, string key)
    {
        ArgumentNullException.ThrowIfNull(weight);
        if (weight.DType.IsQuantized)
            throw new ArgumentException($"'{key}' is {weight.DType}; packed weights go through the quantized-weight policy.", nameof(weight));
        if (backend is null || backend is CpuBackend)
            return TensorCasts.EnsureF32(weight);
        if (weight.DType == DType.BF16 && !backend.Capabilities.SupportsBF16)
            return CastBf16ToF16Checked(weight, key);
        return weight;
    }

    /// <summary>Casts a BF16 tensor to a new F16 tensor, refusing when any value lies outside F16's finite range.</summary>
    /// <exception cref="NotSupportedException">A finite BF16 value exceeds <see cref="F16Max"/> in magnitude.</exception>
    public static unsafe Tensor CastBf16ToF16Checked(Tensor bf16, string key)
    {
        if (bf16.DType != DType.BF16)
            throw new ArgumentException($"'{key}' is {bf16.DType}, not BF16.", nameof(bf16));
        ushort* src = (ushort*)bf16.DataPointer;
        long n = bf16.ElementCount;
        float worst = 0f;
        for (long i = 0; i < n; i++)
        {
            float v = MathF.Abs(BitConverter.UInt32BitsToSingle((uint)src[i] << 16));
            if (v > worst && float.IsFinite(v)) worst = v;
        }
        if (worst > F16Max)
            throw new NotSupportedException(
                $"'{key}' holds BF16 values up to {worst:G4}, beyond F16's {F16Max}, and this backend has no BF16 GEMM. "
                + "Run it on a backend with BF16 (CUDA on Ampere or newer) or on CPU.");
        return bf16.CastTo(DType.F16);
    }
}
