using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>Builds rotate-half cos/sin rows for multimodal RoPE: frequency pair <c>j</c> rotates by
/// <c>position[axis(j)] · invFreq[j]</c>, one shared frequency ladder over the rotary width with each pair reading its
/// axis (<see cref="MRopePairAxes"/>), expanded by <see cref="RopeTableExpander"/> into <c>headDim</c>-wide rows whose
/// channels past the rotary width are zero. With equal positions on every axis this is the plain partial-rotary table,
/// value for value. (<see cref="RopeSpec"/> cannot express it: there each axis owns its own frequency ladder.)</summary>
public static class MRopeTable
{
    /// <summary>Writes rows for tokens <c>[first, first + count)</c> of axis-major <paramref name="positions"/>
    /// (<c>[3 · tokens]</c>, see <see cref="MRopePositions"/>) into the first <c>count · headDim</c> floats of
    /// <paramref name="cos"/>/<paramref name="sin"/>.</summary>
    public static void BuildRows(ReadOnlySpan<int> positions, int tokens, int first, int count, ReadOnlySpan<double> invFreq,
        double attentionScaling, ReadOnlySpan<byte> pairAxes, int headDim, int rotaryDim, Span<float> cos, Span<float> sin)
    {
        int half = RopeTableExpander.EffectiveRotaryDim(headDim, rotaryDim, RopeTableLayout.RotateHalf) / 2;
        if (invFreq.Length < half || pairAxes.Length < half)
            throw new ArgumentException($"{half} frequency pairs need as many frequencies and axis entries.", nameof(invFreq));
        if (positions.Length != 3 * tokens || first < 0 || count < 0 || first + count > tokens)
            throw new ArgumentOutOfRangeException(nameof(first), $"Rows [{first}, {first + count}) of {tokens} positions.");
        long rows = (long)count * headDim;
        if (cos.Length < rows || sin.Length < rows)
            throw new ArgumentException($"{count} rows of {headDim} need larger cos/sin buffers.", nameof(cos));
        float[] pairCos = new float[count * half];
        float[] pairSin = new float[count * half];
        for (int r = 0; r < count; r++)
        {
            int token = first + r;
            for (int j = 0; j < half; j++)
            {
                double angle = positions[pairAxes[j] * tokens + token] * invFreq[j];
                pairCos[r * half + j] = (float)(Math.Cos(angle) * attentionScaling);
                pairSin[r * half + j] = (float)(Math.Sin(angle) * attentionScaling);
            }
        }
        RopeTableExpander.Expand(pairCos, pairSin, count, headDim, RopeTableLayout.RotateHalf, rotaryDim, RopeTablePadding.Zero,
            cos[..(int)rows], sin[..(int)rows]);
    }

    /// <summary>The rows of <see cref="BuildRows"/> as host F32 <c>[1, count, headDim]</c> tensors (caller owns).</summary>
    public static unsafe (Tensor Cos, Tensor Sin) BuildTensors(ReadOnlySpan<int> positions, int tokens, int first, int count,
        ReadOnlySpan<double> invFreq, double attentionScaling, ReadOnlySpan<byte> pairAxes, int headDim, int rotaryDim)
    {
        Tensor cos = new(new TensorShape(1, count, headDim), DType.F32);
        Tensor sin = new(new TensorShape(1, count, headDim), DType.F32);
        try
        {
            BuildRows(positions, tokens, first, count, invFreq, attentionScaling, pairAxes, headDim, rotaryDim,
                new Span<float>((void*)cos.DataPointer, count * headDim), new Span<float>((void*)sin.DataPointer, count * headDim));
            return (cos, sin);
        }
        catch
        {
            cos.Dispose();
            sin.Dispose();
            throw;
        }
    }
}
