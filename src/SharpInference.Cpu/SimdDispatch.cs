using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace SharpInference.Cpu;

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

    /// <summary>Gets a value indicating whether the current CPU supports the ARM NEON (AdvSimd) instruction set.</summary>
    public static bool IsNeonSupported
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => AdvSimd.IsSupported;
    }

    /// <summary>Gets the best available SIMD vector width in number of 32-bit floats. Returns 16 for AVX-512, 8 for AVX2, or 4 for NEON and scalar fallback.</summary>
    public static int SimdWidth
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (IsAvx512Supported)
            {
                return 16;
            }

            if (IsAvx2Supported)
            {
                return 8;
            }

            return 4;
        }
    }
}
