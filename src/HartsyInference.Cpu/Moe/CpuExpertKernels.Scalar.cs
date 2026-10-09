namespace HartsyInference.Cpu.Moe;

/// <summary>Scalar integer block dot products and the packed block sizes of the Q8_0 and Q4_K formats.</summary>
public static unsafe partial class CpuExpertKernels
{
    /// <summary>Bytes per Q8_0 block: a half scale and 32 int8 codes.</summary>
    private const int Q8BlockBytes = 34;

    /// <summary>Elements per Q4_K super-block.</summary>
    private const int Q4KSuperBlockElems = 256;

    /// <summary>Bytes per Q4_K super-block: two half scales, 12 bytes of packed 6-bit scales and minimums, 128 bytes of nibbles.</summary>
    private const int Q4KBlockBytes = 144;

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
}
