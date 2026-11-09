namespace HartsyInference.Core.Backends.Cpu;

/// <summary>Provides runtime detection of SIMD instruction set support (AVX2, AVX-512, ARM NEON) and reports the best available vector width for dispatching kernel implementations.</summary>
public static class SimdDispatch
{
    /// <summary>Gets a value indicating whether the current CPU supports the AVX2 instruction set.</summary>
    public static bool IsAvx2Supported
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Avx2.IsSupported;
    }

    /// <summary>Gets a value indicating whether the current CPU supports the AVX-512F instruction set.</summary>
    public static bool IsAvx512Supported
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Avx512F.IsSupported;
    }

    /// <summary>Sums the 8 lanes of an AVX2 vector in ascending index order, matching the manual stackalloc+store reduction it replaces.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe float HorizontalSum(Vector256<float> v)
    {
        float* tmp = stackalloc float[Vector256<float>.Count];
        Avx.Store(tmp, v);
        float sum = 0f;
        for (int j = 0; j < Vector256<float>.Count; j++)
        {
            sum += tmp[j];
        }

        return sum;
    }
}
