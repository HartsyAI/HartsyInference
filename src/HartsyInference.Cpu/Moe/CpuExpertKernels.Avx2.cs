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
}
