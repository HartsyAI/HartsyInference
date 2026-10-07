namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Softmax pooling of <c>ratio</c> consecutive tokens into one KV latent, plus the partial group carried between calls.</summary>
/// <remarks>Works per absolute position, so prefill from 0, chunked prefill and single-token decode all share one path and
/// agree with upstream <c>Compressor</c>. Inputs are the already projected <c>wkv</c> and <c>wgate</c> rows; the caller
/// applies the output RMS norm. Ratio 1 has no pooling and no state, so it is not handled here. Upstream's compressor has
/// no positional bias and no overlapping windows, so plain per-group softmax is the whole operation.</remarks>
public sealed class DeepSeekV41CompressorState
{
    private readonly float[] _kv;
    private readonly float[] _score;

    /// <summary>Tokens pooled into one latent; at least 2.</summary>
    public int Ratio { get; }

    /// <summary>Width of one latent row.</summary>
    public int HeadDim { get; }

    /// <summary>Creates an empty state for one sequence.</summary>
    public DeepSeekV41CompressorState(int ratio, int headDim)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(ratio, 2);
        ArgumentOutOfRangeException.ThrowIfLessThan(headDim, 1);
        Ratio = ratio;
        HeadDim = headDim;
        _kv = new float[ratio * headDim];
        _score = new float[ratio * headDim];
        Reset();
    }

    /// <summary>Forgets any partial group.</summary>
    public void Reset()
    {
        Array.Clear(_kv);
        Array.Fill(_score, float.NegativeInfinity);
    }

    /// <summary>Upper bound on the latents <see cref="Pool"/> can emit for <paramref name="count"/> tokens starting at <paramref name="startPos"/>.</summary>
    public int MaxRows(int startPos, int count) => (startPos + count) / Ratio - startPos / Ratio;

    /// <summary>Consumes <paramref name="count"/> rows at absolute positions <c>[startPos, startPos + count)</c> and writes each completed group's latent to <paramref name="dest"/>.</summary>
    /// <param name="kv">Projected <c>wkv</c> rows, <c>[count, HeadDim]</c>.</param>
    /// <param name="score">Projected <c>wgate</c> rows, <c>[count, HeadDim]</c>.</param>
    /// <param name="startPos">Absolute position of the first row; groups are aligned to multiples of <see cref="Ratio"/>.</param>
    /// <param name="dest">Receives <c>[rows, HeadDim]</c>, pre-norm.</param>
    /// <returns>The number of latent rows written; zero while a group is still filling.</returns>
    public int Pool(ReadOnlySpan<float> kv, ReadOnlySpan<float> score, int count, int startPos, Span<float> dest)
    {
        int d = HeadDim;
        if (kv.Length != count * d || score.Length != count * d) throw new ArgumentException("kv and score must each hold count x HeadDim values.");
        if (startPos < 0) throw new ArgumentOutOfRangeException(nameof(startPos));
        int maxRows = MaxRows(startPos, count);
        if (dest.Length < maxRows * d) throw new ArgumentException("dest is too small for the completed groups.", nameof(dest));
        int rows = 0;
        for (int i = 0; i < count; i++)
        {
            int pos = startPos + i;
            int slot = pos % Ratio;
            kv.Slice(i * d, d).CopyTo(_kv.AsSpan(slot * d, d));
            score.Slice(i * d, d).CopyTo(_score.AsSpan(slot * d, d));
            if (slot == Ratio - 1) PoolGroup(dest.Slice(rows++ * d, d));
        }
        return rows;
    }

    private void PoolGroup(Span<float> dest)
    {
        int d = HeadDim;
        for (int c = 0; c < d; c++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < Ratio; j++) max = MathF.Max(max, _score[j * d + c]);
            float sum = 0f, acc = 0f;
            for (int j = 0; j < Ratio; j++)
            {
                float e = MathF.Exp(_score[j * d + c] - max);
                sum += e;
                acc += _kv[j * d + c] * e;
            }
            dest[c] = acc / sum;
        }
    }
}
