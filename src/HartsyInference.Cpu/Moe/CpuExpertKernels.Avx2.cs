using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace HartsyInference.Cpu.Moe;

/// <summary>AVX2 integer block dot products. Each widens its bytes to 16 bits and multiplies with <c>pmaddwd</c>, whose pair
/// sums cannot saturate (<c>pmaddubsw</c> can, with two same-sign products near the byte limits). Only exact integer sums come
/// from here, so the results equal the scalar ones bit for bit.</summary>
/// <remarks>Callers must check <see cref="Avx2.IsSupported"/>; the dispatch in <c>CpuExpertKernels.cs</c> does.</remarks>
public static unsafe partial class CpuExpertKernels
{
    private static int Q8BlockDotAvx2(sbyte* w, sbyte* a)
    {
        Vector256<int> acc = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Sse2.LoadVector128(w)),
            Avx2.ConvertToVector256Int16(Sse2.LoadVector128(a)));
        acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Sse2.LoadVector128(w + 16)),
            Avx2.ConvertToVector256Int16(Sse2.LoadVector128(a + 16))));
        return Vector256.Sum(acc);
    }

    private static int Q4SubDotAvx2(byte* quants, int shift, sbyte* a)
    {
        Vector256<byte> packed = Avx.LoadVector256(quants);
        Vector256<byte> nibbles = shift == 0 ? packed : Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte();
        nibbles = Avx2.And(nibbles, Vector256.Create((byte)0x0F));
        Vector256<int> acc = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Vector256.GetLower(nibbles)),
            Avx2.ConvertToVector256Int16(Sse2.LoadVector128(a)));
        acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Vector256.GetUpper(nibbles)),
            Avx2.ConvertToVector256Int16(Sse2.LoadVector128(a + 16))));
        return Vector256.Sum(acc);
    }

    private static int Q5SubDotAvx2(byte* quants, int shift, byte* qh, int bit, sbyte* a)
    {
        Vector256<byte> packed = Avx.LoadVector256(quants);
        Vector256<byte> nibbles = shift == 0 ? packed : Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte();
        nibbles = Avx2.And(nibbles, Vector256.Create((byte)0x0F));
        // Bit `bit` of each qh byte, moved to 0x10. Shifting 16-bit lanes is safe: the mask keeps one bit per byte, and a value of
        // at most 1 shifted left by 4 never reaches the neighbouring byte.
        Vector256<byte> high = Avx2.And(Avx2.ShiftRightLogical(Avx.LoadVector256(qh).AsUInt16(), Vector128.CreateScalar((ushort)bit)).AsByte(), Vector256.Create((byte)1));
        high = Avx2.ShiftLeftLogical(high.AsUInt16(), 4).AsByte();
        Vector256<byte> values = Avx2.Or(nibbles, high);
        Vector256<int> acc = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Vector256.GetLower(values)),
            Avx2.ConvertToVector256Int16(Sse2.LoadVector128(a)));
        acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Vector256.GetUpper(values)),
            Avx2.ConvertToVector256Int16(Sse2.LoadVector128(a + 16))));
        return Vector256.Sum(acc);
    }

    private static void Q6GroupDotAvx2(byte* ql, byte* qh, int g, sbyte* a, out int lo, out int hi)
    {
        Vector256<byte> low = Avx.LoadVector256(ql + (g & 1) * 32);
        if (g >= 2) low = Avx2.ShiftRightLogical(low.AsUInt16(), 4).AsByte();
        low = Avx2.And(low, Vector256.Create((byte)0x0F));
        // Two bits per byte moved to bits 4-5. As above, masking before the 16-bit shift keeps every byte's bits in that byte.
        Vector256<byte> high = Avx2.And(Avx2.ShiftRightLogical(Avx.LoadVector256(qh).AsUInt16(), Vector128.CreateScalar((ushort)(2 * g))).AsByte(),
            Vector256.Create((byte)3));
        high = Avx2.ShiftLeftLogical(high.AsUInt16(), 4).AsByte();
        Vector256<byte> values = Avx2.Or(low, high);
        Vector256<short> offset = Vector256.Create((short)32);
        Vector256<short> first = Avx2.Subtract(Avx2.ConvertToVector256Int16(Vector256.GetLower(values)), offset);
        Vector256<short> second = Avx2.Subtract(Avx2.ConvertToVector256Int16(Vector256.GetUpper(values)), offset);
        lo = Vector256.Sum(Avx2.MultiplyAddAdjacent(first, Avx2.ConvertToVector256Int16(Sse2.LoadVector128(a))));
        hi = Vector256.Sum(Avx2.MultiplyAddAdjacent(second, Avx2.ConvertToVector256Int16(Sse2.LoadVector128(a + 16))));
    }
}
