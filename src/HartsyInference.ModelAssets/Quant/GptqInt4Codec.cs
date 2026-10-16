using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Quant;

/// <summary>Reads symmetric 4-bit GPTQ linears (AutoGPTQ / GPTQModel layout): <c>qweight</c> <c>[in/8, out]</c> I32 with eight input
/// rows per word, lowest nibble first; <c>scales</c> <c>[in/group, out]</c>; <c>qzeros</c> <c>[in/group, out/8]</c> I32; and an
/// optional <c>g_idx</c>. A symmetric group's zero point is 8, stored as 7 by AutoGPTQ (which adds one back) and as 8 by GPTQModel's
/// v2 format. With a group of a multiple of 32 such a weight is exactly a ggml Q4_0 weight — <c>w = d · (q − 8)</c> per 32-element
/// block, the group's F16 scale repeated over its blocks — so <see cref="ToQ4_0"/> repacks it without changing a single value.</summary>
public static unsafe class GptqInt4Codec
{
    private const int NibblesPerWord = 8;
    private const int Q4BlockElements = 32;
    private const int Q4BlockBytes = 18;

    /// <summary>Repacks a symmetric GPTQ linear into a Q4_0 <c>[out, in]</c> tensor the caller owns; bit-exact when the scales are F16.</summary>
    public static Tensor ToQ4_0(Tensor qweight, Tensor scales, Tensor? qzeros, Tensor? gIdx, int groupSize)
    {
        (int inFeatures, int outFeatures) = Validate(qweight, scales, qzeros, gIdx, groupSize, Q4BlockElements);
        Tensor halfScales = scales.DType == DType.F16 ? scales : scales.CastTo(DType.F16);
        Tensor output = new(new TensorShape(outFeatures, inFeatures), DType.Q4_0);
        try
        {
            nint words = (nint)qweight.DataPointer, scale = (nint)halfScales.DataPointer, dst = (nint)output.DataPointer;
            int blocksPerRow = inFeatures / Q4BlockElements;
            CpuParallel.For(outFeatures, (long)outFeatures * inFeatures, o =>
            {
                uint* w = (uint*)words;
                ushort* s = (ushort*)scale;
                byte* row = (byte*)dst + (long)o * blocksPerRow * Q4BlockBytes;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = row + (long)b * Q4BlockBytes;
                    int first = b * Q4BlockElements;
                    *(ushort*)block = s[(long)(first / groupSize) * outFeatures + o];
                    for (int k = 0; k < 16; k++)
                    {
                        uint lo = Nibble(w, outFeatures, o, first + k);
                        uint hi = Nibble(w, outFeatures, o, first + 16 + k);
                        block[2 + k] = (byte)(lo | (hi << 4));
                    }
                }
            });
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
        finally
        {
            if (!ReferenceEquals(halfScales, scales)) halfScales.Dispose();
        }
    }

    /// <summary>Dequantizes a symmetric GPTQ linear to an F32 <c>[out, in]</c> tensor the caller owns.</summary>
    public static Tensor Dequantize(Tensor qweight, Tensor scales, Tensor? qzeros, Tensor? gIdx, int groupSize)
    {
        (int inFeatures, int outFeatures) = Validate(qweight, scales, qzeros, gIdx, groupSize, 1);
        Tensor f32Scales = scales.DType == DType.F32 ? scales : scales.CastTo(DType.F32);
        Tensor output = new(new TensorShape(outFeatures, inFeatures), DType.F32);
        try
        {
            nint words = (nint)qweight.DataPointer, scale = (nint)f32Scales.DataPointer, dst = (nint)output.DataPointer;
            CpuParallel.For(outFeatures, (long)outFeatures * inFeatures, o =>
            {
                uint* w = (uint*)words;
                float* s = (float*)scale;
                float* row = (float*)dst + (long)o * inFeatures;
                for (int i = 0; i < inFeatures; i++)
                {
                    row[i] = s[(long)(i / groupSize) * outFeatures + o] * ((int)Nibble(w, outFeatures, o, i) - 8);
                }
            });
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
        finally
        {
            if (!ReferenceEquals(f32Scales, scales)) f32Scales.Dispose();
        }
    }

    /// <summary>The <c>[out, in]</c> shape a GPTQ linear dequantizes to.</summary>
    public static TensorShape LogicalShape(Tensor qweight) =>
        new(qweight.Shape[1], qweight.Shape[0] * NibblesPerWord);

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static uint Nibble(uint* words, int outFeatures, int o, int i) =>
        (words[(long)(i / NibblesPerWord) * outFeatures + o] >> (4 * (i % NibblesPerWord))) & 0xFu;

    private static (int In, int Out) Validate(Tensor qweight, Tensor scales, Tensor? qzeros, Tensor? gIdx, int groupSize, int blockAlign)
    {
        ArgumentNullException.ThrowIfNull(qweight);
        ArgumentNullException.ThrowIfNull(scales);
        if ((qweight.DType != DType.I32 && qweight.DType != DType.U32) || qweight.Shape.Rank != 2)
        {
            throw new ArgumentException($"A GPTQ qweight is a rank-2 I32 tensor, not {qweight.Shape} {qweight.DType.Name}.", nameof(qweight));
        }
        int inFeatures = checked((int)qweight.Shape[0] * NibblesPerWord), outFeatures = checked((int)qweight.Shape[1]);
        if (groupSize <= 0 || inFeatures % groupSize != 0 || groupSize % blockAlign != 0)
        {
            throw new NotSupportedException(
                $"A GPTQ group of {groupSize} does not tile {inFeatures} inputs in whole {blockAlign}-element blocks.");
        }
        int groups = inFeatures / groupSize;
        if (scales.Shape.Rank != 2 || scales.Shape[0] != groups || scales.Shape[1] != outFeatures)
        {
            throw new ArgumentException($"A [{outFeatures}, {inFeatures}] GPTQ linear with group {groupSize} needs [{groups}, {outFeatures}] scales, not {scales.Shape}.", nameof(scales));
        }
        if (scales.DType != DType.F16 && scales.DType != DType.BF16 && scales.DType != DType.F32)
        {
            throw new ArgumentException($"GPTQ scales are F16, BF16 or F32, not {scales.DType.Name}.", nameof(scales));
        }
        if (qzeros is not null) CheckSymmetricZeros(qzeros, groups, outFeatures);
        if (gIdx is not null) CheckGroupOrder(gIdx, inFeatures, groupSize);
        return (inFeatures, outFeatures);
    }

    // A symmetric checkpoint stores one zero for every group and column: 7 under AutoGPTQ's offset-by-one, 8 under GPTQModel v2.
    private static void CheckSymmetricZeros(Tensor qzeros, int groups, int outFeatures)
    {
        if (qzeros.Shape.Rank != 2 || qzeros.Shape[0] != groups || qzeros.Shape[1] * NibblesPerWord != outFeatures)
        {
            throw new ArgumentException($"GPTQ qzeros of {qzeros.Shape} do not match {groups} groups of {outFeatures} outputs.", nameof(qzeros));
        }
        ReadOnlySpan<uint> words = new((void*)qzeros.DataPointer, checked((int)qzeros.ElementCount));
        uint first = words.IsEmpty ? 0x77777777u : words[0];
        if (first != 0x77777777u && first != 0x88888888u)
        {
            throw new NotSupportedException("The GPTQ zero points are not the symmetric 8; asymmetric GPTQ weights are not supported.");
        }
        if (words.IndexOfAnyExcept(first) >= 0)
        {
            throw new NotSupportedException("The GPTQ zero points vary across groups; asymmetric GPTQ weights are not supported.");
        }
    }

    // desc_act (activation order) permutes the inputs across groups; only the identity order maps onto contiguous blocks.
    private static void CheckGroupOrder(Tensor gIdx, int inFeatures, int groupSize)
    {
        if (gIdx.ElementCount != inFeatures)
        {
            throw new ArgumentException($"A GPTQ g_idx of {gIdx.ElementCount} entries does not cover {inFeatures} inputs.", nameof(gIdx));
        }
        ReadOnlySpan<int> order = new((void*)gIdx.DataPointer, inFeatures);
        for (int i = 0; i < inFeatures; i++)
        {
            if (order[i] != i / groupSize)
            {
                throw new NotSupportedException("The GPTQ weight uses activation order (desc_act); it is not supported.");
            }
        }
    }
}
