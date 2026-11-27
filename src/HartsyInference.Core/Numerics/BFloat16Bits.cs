namespace HartsyInference.Core.Numerics;

/// <summary>bfloat16 as its 16-bit pattern: the upper half of an IEEE-754 single. Conversion from single rounds to
/// nearest even, as the CUDA <c>__float2bfloat16</c> intrinsic does, so a host reference reproduces a kernel's rounding.</summary>
public static class BFloat16Bits
{
    /// <summary>The single whose upper 16 bits are <paramref name="bits"/>.</summary>
    public static float ToFloat(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    /// <summary>Rounds <paramref name="value"/> to the nearest bfloat16, ties to even; a NaN stays a (quiet) NaN.</summary>
    public static ushort FromFloat(float value)
    {
        uint u = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return (ushort)((u >> 16) | 0x40u);
        uint rounding = 0x7FFFu + ((u >> 16) & 1u);
        return (ushort)((u + rounding) >> 16);
    }
}
