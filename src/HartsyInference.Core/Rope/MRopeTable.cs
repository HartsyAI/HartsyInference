using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>Builds rotate-half cos/sin rows for multimodal RoPE: frequency pair <c>j</c> rotates by
/// <c>position[axis(j)] · invFreq[j]</c>, written to channels <c>j</c> and <c>j + rotaryDim/2</c> of a
/// <c>headDim</c>-wide row (channels past the rotary width are left as they are). With equal positions on every axis
/// this is the plain partial-rotary table, value for value.</summary>
public static class MRopeTable
{
    /// <summary>Writes rows for tokens <c>[first, first + count)</c> of axis-major <paramref name="positions"/>
    /// (<c>[3 · tokens]</c>, see <see cref="MRopePositions"/>) into <paramref name="cos"/>/<paramref name="sin"/>
    /// (<c>count · headDim</c> floats).</summary>
    public static void BuildRows(ReadOnlySpan<int> positions, int tokens, int first, int count, ReadOnlySpan<double> invFreq,
        double attentionScaling, ReadOnlySpan<byte> pairAxes, int headDim, int rotaryDim, Span<float> cos, Span<float> sin)
    {
        int half = rotaryDim / 2;
        if (rotaryDim <= 0 || rotaryDim > headDim || rotaryDim % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(rotaryDim), rotaryDim, $"The rotary width must be even and at most the head width {headDim}.");
        if (invFreq.Length < half || pairAxes.Length < half)
            throw new ArgumentException($"{half} frequency pairs need as many frequencies and axis entries.", nameof(invFreq));
        if (positions.Length != 3 * tokens || first < 0 || count < 0 || first + count > tokens)
            throw new ArgumentOutOfRangeException(nameof(first), $"Rows [{first}, {first + count}) of {tokens} positions.");
        if (cos.Length < (long)count * headDim || sin.Length < (long)count * headDim)
            throw new ArgumentException($"{count} rows of {headDim} need larger cos/sin buffers.", nameof(cos));
        for (int r = 0; r < count; r++)
        {
            int token = first + r;
            int rowBase = r * headDim;
            for (int j = 0; j < half; j++)
            {
                double angle = positions[pairAxes[j] * tokens + token] * invFreq[j];
                float c = (float)(Math.Cos(angle) * attentionScaling);
                float s = (float)(Math.Sin(angle) * attentionScaling);
                cos[rowBase + j] = c;
                cos[rowBase + j + half] = c;
                sin[rowBase + j] = s;
                sin[rowBase + j + half] = s;
            }
        }
    }

    /// <summary>The rows of <see cref="BuildRows"/> as host F32 <c>[1, count, headDim]</c> tensors (caller owns).</summary>
    public static unsafe (Tensor Cos, Tensor Sin) BuildTensors(ReadOnlySpan<int> positions, int tokens, int first, int count,
        ReadOnlySpan<double> invFreq, double attentionScaling, ReadOnlySpan<byte> pairAxes, int headDim, int rotaryDim)
    {
        Tensor cos = new(new TensorShape(1, count, headDim), DType.F32);
        Tensor sin = new(new TensorShape(1, count, headDim), DType.F32);
        try
        {
            Span<float> c = new((void*)cos.DataPointer, count * headDim);
            Span<float> s = new((void*)sin.DataPointer, count * headDim);
            c.Clear();
            s.Clear();
            BuildRows(positions, tokens, first, count, invFreq, attentionScaling, pairAxes, headDim, rotaryDim, c, s);
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
