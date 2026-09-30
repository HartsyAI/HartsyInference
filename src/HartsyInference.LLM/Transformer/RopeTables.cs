using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Split-half (rotate-half) RoPE cos/sin tables for the LLM transformer family: rows are <c>headDim</c>-strided with both halves of the rotated span duplicated, the layout <see cref="Core.Backends.IBackend.ApplyRopeSingle"/> consumes.</summary>
internal static unsafe class RopeTables
{
    /// <summary>Builds duplicated-half cos/sin for <paramref name="t"/> consecutive positions from <paramref name="posStart"/>: <c>cos[s,i] = cos[s,i+half] = cos((posStart+s)·freq_i)</c>; partial rotary fills only the first <paramref name="rotaryDim"/> dims of each row.</summary>
    public static void BuildRope(Tensor cos, Tensor sin, int t, int posStart, int headDim, int rotaryDim, float theta, RopeScaling scaling) =>
        Fill(cos, sin, t, posStart, default, headDim, rotaryDim, dimOffset: 0, inverse: false, theta, scaling);

    /// <summary>Per-row table for a ragged decode batch: row b uses absolute position <paramref name="positions"/>[b].</summary>
    public static void BuildRopeBatched(Tensor cos, Tensor sin, ReadOnlySpan<int> positions, int headDim, int rotaryDim,
        float theta, RopeScaling scaling) =>
        Fill(cos, sin, positions.Length, 0, positions, headDim, rotaryDim, dimOffset: 0, inverse: false, theta, scaling);

    /// <summary>Like <see cref="BuildRope"/> but the rotated span starts at <paramref name="dimOffset"/> inside each row (rope applied to the tail of the head), with <paramref name="inverse"/> negating sin to undo a rotation.</summary>
    public static void BuildRopeTail(Tensor cos, Tensor sin, int t, int posStart, int headDim, int rotaryDim, int dimOffset,
        bool inverse, float theta, RopeScaling scaling) =>
        Fill(cos, sin, t, posStart, default, headDim, rotaryDim, dimOffset, inverse, theta, scaling);

    /// <summary>Batched variant of <see cref="BuildRopeTail"/>: row b uses <paramref name="positions"/>[b].</summary>
    public static void BuildRopeBatchedTail(Tensor cos, Tensor sin, ReadOnlySpan<int> positions, int headDim, int rotaryDim,
        int dimOffset, bool inverse, float theta, RopeScaling scaling) =>
        Fill(cos, sin, positions.Length, 0, positions, headDim, rotaryDim, dimOffset, inverse, theta, scaling);

    /// <summary>Inverse of <see cref="BuildRope"/>: same cos, negated sin, so applying it after the forward table restores the input.</summary>
    public static void BuildRopeInverse(Tensor cos, Tensor sin, int t, int posStart, int headDim, int rotaryDim, float theta, RopeScaling scaling) =>
        Fill(cos, sin, t, posStart, default, headDim, rotaryDim, dimOffset: 0, inverse: true, theta, scaling);

    private static void Fill(Tensor cos, Tensor sin, int rows, int posStart, ReadOnlySpan<int> positions, int headDim,
        int rotaryDim, int dimOffset, bool inverse, float theta, RopeScaling scaling)
    {
        // 0/full rotaryDim → the whole head.
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        if (dimOffset < 0 || dimOffset + rdim > headDim)
            throw new ArgumentOutOfRangeException(nameof(dimOffset), $"rotary span [{dimOffset}, {dimOffset + rdim}) exceeds headDim {headDim}.");
        int half = rdim / 2;
        bool batched = !positions.IsEmpty;
        int maxPos = batched ? 0 : posStart + rows - 1;
        if (batched)
        {
            for (int s = 0; s < positions.Length; s++) maxPos = Math.Max(maxPos, positions[s]);
        }
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rdim, theta, scaling, maxPos + 1);
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < rows; s++)
        {
            int pos = batched ? positions[s] : posStart + s;
            long baseOff = (long)s * headDim + dimOffset;
            for (int i = 0; i < half; i++)
            {
                double angle = pos * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale);
                float si = (float)(Math.Sin(angle) * mscale);
                if (inverse) si = -si;
                pc[baseOff + i] = c; pc[baseOff + i + half] = c;
                ps[baseOff + i] = si; ps[baseOff + i + half] = si;
            }
        }
    }
}
