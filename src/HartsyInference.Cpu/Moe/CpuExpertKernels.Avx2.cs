using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Moe;

/// <summary>
/// AVX2 row dot products: one weight row against one quantized activation row, accumulated in float VECTORS with a single
/// horizontal sum per row.
/// </summary>
/// <remarks>
/// <para><b>Products.</b> Each 32-value block multiplies unsigned weight bytes by signed activation codes with <c>pmaddubsw</c>, then
/// pairs the 16-bit results with <c>pmaddwd</c>, as llama.cpp's AVX2 kernels do. The 16-bit pairs cannot saturate: unsigned
/// weights stay at or below 63 (Q4_K 15, Q5_K 31, Q6_K's magnitude 32) and Q8_0 reaches 128 only as a magnitude, against
/// activation codes in [-127, 127]. Signed weights (Q6_K after its offset, Q8_0) go through the sign trick,
/// <c>|w| x sign(a, w)</c>, which equals <c>w x a</c>. Every integer sum is exact.</para>
/// <para><b>Float order.</b> The scaled block sums are accumulated lane-wise and summed once at the end, so the float result is not
/// bit-identical to the scalar path's block-by-block fold; it differs at the level of float rounding, which the tests bound.</para>
/// <para>Callers check <see cref="Avx2.IsSupported"/>; the dispatch in <c>CpuExpertKernels.cs</c> does. FMA is used when present.</para>
/// </remarks>
public static unsafe partial class CpuExpertKernels
{
    private static readonly Vector256<short> Ones16 = Vector256.Create((short)1);
    private static readonly Vector256<byte> LowNibble = Vector256.Create((byte)0x0F);

    /// <summary>The row dot product on AVX2, for every format the kernels read.</summary>
    private static float DotRowAvx2(DType dtype, byte* row, sbyte* act, float* actScale, int* actSum, int dim)
    {
        if (dtype == DType.Q8_0) return DotRowQ8Avx2(row, act, actScale, dim);
        if (dtype == DType.Q6_K) return DotRowQ6KAvx2(row, act, actScale, dim);
        if (dtype == DType.Q5_K) return DotRowQ5KAvx2(row, act, actScale, actSum, dim);
        return DotRowQ4KAvx2(row, act, actScale, actSum, dim);
    }

    /// <summary><c>acc + products x scale</c>, with FMA when the CPU has it.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Accumulate(Vector256<float> acc, Vector256<int> products, Vector256<float> scale)
    {
        Vector256<float> p = Avx.ConvertToVector256Single(products);
        return Fma.IsSupported ? Fma.MultiplyAdd(p, scale, acc) : Avx.Add(acc, Avx.Multiply(p, scale));
    }

    /// <summary>Eight 32-bit sums of <c>u x a</c> over 32 unsigned weight bytes and 32 signed activation codes; lane k holds values
    /// 4k to 4k+3 of its 128-bit half, so lanes 0-3 cover the first 16 values and lanes 4-7 the last 16.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Dot32(Vector256<byte> u, sbyte* a) =>
        Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u, Avx.LoadVector256(a)), Ones16);

    /// <summary>As <see cref="Dot32"/> for SIGNED weights: <c>|w| x sign(a, w)</c> is <c>w x a</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Dot32Signed(Vector256<sbyte> w, sbyte* a) =>
        Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx2.Abs(w), Avx2.Sign(Avx.LoadVector256(a), w)), Ones16);

    /// <summary>The eight 6-bit scales and eight 6-bit minimums of a Q4_K/Q5_K super-block, unpacked in one pass (llama.cpp's
    /// <c>utmp</c> shuffle) instead of per sub-block: bytes 0-7 of <paramref name="out16"/> are the scales, 8-15 the minimums.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackScalesMinsK4(byte* packed, byte* out16)
    {
        const uint kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
        uint u0 = Unsafe.ReadUnaligned<uint>(packed), u1 = Unsafe.ReadUnaligned<uint>(packed + 4), u2 = Unsafe.ReadUnaligned<uint>(packed + 8);
        uint* o = (uint*)out16;
        o[3] = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
        uint mins01 = u1 & kmask1;
        o[1] = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
        o[2] = mins01;
        o[0] = u0 & kmask1;
    }

    private static float DotRowQ8Avx2(byte* row, sbyte* act, float* actScale, int dim)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        int blocks = dim / QuantBlock;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = row + b * Q8BlockBytes;
            Vector256<sbyte> w = Avx.LoadVector256((sbyte*)(block + 2));
            acc = Accumulate(acc, Dot32Signed(w, act + b * QuantBlock), Vector256.Create(ReadHalf(block) * actScale[b]));
        }
        return Vector256.Sum(acc);
    }

    private static float DotRowQ4KAvx2(byte* row, sbyte* act, float* actScale, int* actSum, int dim)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float minimum = 0f;
        byte* sm = stackalloc byte[16];
        int superBlocks = dim / Q4KSuperBlockElems;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = row + sb * Q4KBlockBytes;
            float d = ReadHalf(block);
            float dmin = ReadHalf(block + 2);
            UnpackScalesMinsK4(block + 4, sm);
            byte* quants = block + 16;
            for (int pair = 0; pair < 4; pair++)
            {
                // 32 bytes hold two sub-blocks: the low nibbles are sub-block 2p, the high nibbles 2p+1.
                Vector256<byte> q = Avx.LoadVector256(quants + pair * QuantBlock);
                Vector256<byte> lo = Avx2.And(q, LowNibble);
                Vector256<byte> hi = Avx2.And(Avx2.ShiftRightLogical(q.AsUInt16(), 4).AsByte(), LowNibble);
                int j0 = 2 * pair, b0 = sb * 8 + j0;
                int sc0 = sm[j0], sc1 = sm[j0 + 1], m0 = sm[8 + j0], m1 = sm[8 + j0 + 1];
                acc = Accumulate(acc, Dot32(lo, act + b0 * QuantBlock), Vector256.Create(d * sc0 * actScale[b0]));
                acc = Accumulate(acc, Dot32(hi, act + (b0 + 1) * QuantBlock), Vector256.Create(d * sc1 * actScale[b0 + 1]));
                minimum += dmin * (m0 * actSum[b0] * actScale[b0] + m1 * actSum[b0 + 1] * actScale[b0 + 1]);
            }
        }
        return Vector256.Sum(acc) - minimum;
    }

    private static float DotRowQ5KAvx2(byte* row, sbyte* act, float* actScale, int* actSum, int dim)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float minimum = 0f;
        byte* sm = stackalloc byte[16];
        Vector256<byte> one = Vector256.Create((byte)1);
        int superBlocks = dim / Q4KSuperBlockElems;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = row + sb * Q5KBlockBytes;
            float d = ReadHalf(block);
            float dmin = ReadHalf(block + 2);
            UnpackScalesMinsK4(block + 4, sm);
            Vector256<ushort> qh = Avx.LoadVector256(block + 16).AsUInt16();
            byte* qs = block + 48;
            for (int pair = 0; pair < 4; pair++)
            {
                Vector256<byte> q = Avx.LoadVector256(qs + pair * QuantBlock);
                int j0 = 2 * pair, b0 = sb * 8 + j0;
                // The fifth bit of sub-block j is bit j of each qh byte, moved to 0x10. Masking to one bit before the 16-bit shift
                // keeps every byte's bit inside that byte.
                Vector256<byte> h0 = Avx2.ShiftLeftLogical(
                    Avx2.And(Avx2.ShiftRightLogical(qh, Vector128.CreateScalar((ushort)j0)).AsByte(), one).AsUInt16(), 4).AsByte();
                Vector256<byte> h1 = Avx2.ShiftLeftLogical(
                    Avx2.And(Avx2.ShiftRightLogical(qh, Vector128.CreateScalar((ushort)(j0 + 1))).AsByte(), one).AsUInt16(), 4).AsByte();
                Vector256<byte> lo = Avx2.Or(Avx2.And(q, LowNibble), h0);
                Vector256<byte> hi = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q.AsUInt16(), 4).AsByte(), LowNibble), h1);
                int sc0 = sm[j0], sc1 = sm[j0 + 1], m0 = sm[8 + j0], m1 = sm[8 + j0 + 1];
                acc = Accumulate(acc, Dot32(lo, act + b0 * QuantBlock), Vector256.Create(d * sc0 * actScale[b0]));
                acc = Accumulate(acc, Dot32(hi, act + (b0 + 1) * QuantBlock), Vector256.Create(d * sc1 * actScale[b0 + 1]));
                minimum += dmin * (m0 * actSum[b0] * actScale[b0] + m1 * actSum[b0 + 1] * actScale[b0 + 1]);
            }
        }
        return Vector256.Sum(acc) - minimum;
    }

    private static float DotRowQ6KAvx2(byte* row, sbyte* act, float* actScale, int dim)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> threeMask = Vector256.Create((byte)3);
        Vector256<sbyte> offset = Vector256.Create((sbyte)32);
        int superBlocks = dim / Q4KSuperBlockElems;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = row + sb * Q6KBlockBytes;
            float d = ReadHalf(block + 208);
            sbyte* sc = (sbyte*)(block + 192);
            for (int half = 0; half < 2; half++)
            {
                byte* ql = block + half * 64;
                Vector256<ushort> qh = Avx.LoadVector256(block + 128 + half * 32).AsUInt16();
                Vector256<byte> ql0 = Avx.LoadVector256(ql);
                Vector256<byte> ql1 = Avx.LoadVector256(ql + 32);
                for (int g = 0; g < 4; g++)
                {
                    Vector256<byte> src = (g & 1) == 0 ? ql0 : ql1;
                    Vector256<byte> low = g >= 2 ? Avx2.ShiftRightLogical(src.AsUInt16(), 4).AsByte() : src;
                    low = Avx2.And(low, LowNibble);
                    Vector256<byte> high = Avx2.And(
                        Avx2.ShiftRightLogical(qh, Vector128.CreateScalar((ushort)(2 * g))).AsByte(), threeMask);
                    high = Avx2.ShiftLeftLogical(high.AsUInt16(), 4).AsByte();
                    Vector256<sbyte> w = Avx2.Subtract(Avx2.Or(low, high).AsSByte(), offset);
                    int b = sb * 8 + half * 4 + g;
                    int s = half * 8 + 2 * g;
                    float scale = d * actScale[b];
                    float first = sc[s] * scale, second = sc[s + 1] * scale;
                    // Lanes 0-3 hold the first 16 values (scale s), lanes 4-7 the last 16 (scale s+1).
                    Vector256<float> scales = Vector256.Create(first, first, first, first, second, second, second, second);
                    acc = Accumulate(acc, Dot32Signed(w, act + b * QuantBlock), scales);
                }
            }
        }
        return Vector256.Sum(acc);
    }
}
