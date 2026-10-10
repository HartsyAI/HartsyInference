namespace HartsyInference.Cpu.Moe;

/// <summary>Scalar integer block dot products and the packed block sizes of the Q8_0, Q4_K, Q5_K and Q6_K formats.</summary>
public static unsafe partial class CpuExpertKernels
{
    /// <summary>Bytes per Q8_0 block: a half scale and 32 int8 codes.</summary>
    private const int Q8BlockBytes = 34;

    /// <summary>Elements per Q4_K super-block.</summary>
    private const int Q4KSuperBlockElems = 256;

    /// <summary>Bytes per Q4_K super-block: two half scales, 12 bytes of packed 6-bit scales and minimums, 128 bytes of nibbles.</summary>
    private const int Q4KBlockBytes = 144;

    /// <summary>Bytes per Q5_K super-block: two half scales, 12 bytes of packed scales and minimums, 32 bytes of fifth bits, 128
    /// bytes of nibbles.</summary>
    private const int Q5KBlockBytes = 176;

    /// <summary>Bytes per Q6_K super-block: 128 bytes of low nibbles, 64 of high bit pairs, 16 signed scales and one half scale.</summary>
    private const int Q6KBlockBytes = 210;

    /// <summary>Exact int32 dot of 32 int8 weights and 32 int8 activation codes.</summary>
    private static int Q8BlockDotScalar(sbyte* w, sbyte* a)
    {
        int sum = 0;
        for (int k = 0; k < QuantBlock; k++) sum += w[k] * a[k];
        return sum;
    }

    /// <summary>Exact int32 dot of 32 unsigned 4-bit Q4_K values (the nibble selected by <paramref name="shift"/>) and 32 int8
    /// activation codes.</summary>
    private static int Q4SubDotScalar(byte* quants, int shift, sbyte* a)
    {
        int sum = 0;
        for (int k = 0; k < QuantBlock; k++) sum += ((quants[k] >> shift) & 0x0F) * a[k];
        return sum;
    }

    /// <summary>Exact int32 dot of 32 unsigned 5-bit Q5_K values (the nibble selected by <paramref name="shift"/>, plus 16 when bit
    /// <paramref name="bit"/> of the matching <paramref name="qh"/> byte is set) and 32 int8 activation codes.</summary>
    private static int Q5SubDotScalar(byte* quants, int shift, byte* qh, int bit, sbyte* a)
    {
        int sum = 0;
        for (int k = 0; k < QuantBlock; k++)
            sum += (((quants[k] >> shift) & 0x0F) | (((qh[k] >> bit) & 1) << 4)) * a[k];
        return sum;
    }

    /// <summary>The 6-bit value of element <paramref name="l"/> of group <paramref name="g"/> in one Q6_K half, minus 32.</summary>
    private static int Q6Value(byte* ql, byte* qh, int g, int l) =>
        (((ql[l + (g & 1) * 32] >> (g >= 2 ? 4 : 0)) & 0x0F) | (((qh[l] >> (2 * g)) & 3) << 4)) - 32;

    /// <summary>Exact int32 dots of one Q6_K group's 32 signed values with 32 int8 activation codes, as two 16-value sums, one per
    /// scale.</summary>
    private static void Q6GroupDotScalar(byte* ql, byte* qh, int g, sbyte* a, out int lo, out int hi)
    {
        lo = 0;
        hi = 0;
        for (int l = 0; l < 16; l++) lo += Q6Value(ql, qh, g, l) * a[l];
        for (int l = 16; l < 32; l++) hi += Q6Value(ql, qh, g, l) * a[l];
    }
}
