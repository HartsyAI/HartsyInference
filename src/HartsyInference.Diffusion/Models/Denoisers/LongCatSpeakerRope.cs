using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The 1D L-RoPE of LongCat-Video-Avatar's multi-speaker audio cross-attention (<c>SingleStreamAttention._process_cross_attn</c> and
/// <c>RotaryPositionalEmbedding1D</c>, meituan-longcat/LongCat-Video @ 58eb336). Each noisy video token takes a position in the range of the region
/// it attends to most, and each audio token takes its speaker's position, so a token's query is rotated close to the keys of its own
/// speaker.</summary>
/// <remarks>Positions and angles are F32, as the reference computes them: at the far background position an F64 angle would already differ by more
/// than the parity tolerance.</remarks>
public static class LongCatSpeakerRope
{
    /// <summary>Width of one speaker's query range (<c>class_interval</c>).</summary>
    public const int SpeakerInterval = 4;

    /// <summary>Upper end of the second speaker's query range (<c>class_range</c>); the first speaker's range starts at 0.</summary>
    public const int PositionRange = 24;

    /// <summary>The background position when there are three regions (<c>rope_bak</c>).</summary>
    public const int Background = PositionRange / 2;

    /// <summary>The background position when silent people add further regions.</summary>
    public const int FarBackground = 100;

    private const float Theta = 10000f;
    private const float Epsilon = 1e-8f;

    /// <summary>The audio position of the first speaker's tokens: the middle of its query range.</summary>
    public const float FirstSpeakerKey = SpeakerInterval / 2f;

    /// <summary>The audio position of the second speaker's tokens: the middle of its query range.</summary>
    public const float SecondSpeakerKey = PositionRange - SpeakerInterval / 2f;

    /// <summary>The background position for <paramref name="regions"/> mask regions.</summary>
    public static float BackgroundPosition(int regions) => regions <= LongCatSpeakers.MinRegions ? Background : FarBackground;

    /// <summary>One position per noisy token from the reference attention map <c>[regions, tokens]</c>. A speaker's own map values are scaled to its
    /// range by their minimum and maximum over every token; the token takes the first speaker's, the second speaker's or the background position,
    /// whichever region has its largest value (the first on a tie; regions past the third count as background).</summary>
    public static float[] QueryPositions(ReadOnlySpan<float> map, int regions, int tokens)
    {
        if (regions < LongCatSpeakers.MinRegions || tokens < 1 || map.Length != (long)regions * tokens)
            throw new ArgumentException($"The map must be [{regions}, {tokens}] with at least {LongCatSpeakers.MinRegions} regions.", nameof(map));
        (float firstMin, float firstMax) = Range(map[..tokens]);
        (float secondMin, float secondMax) = Range(map.Slice(tokens, tokens));
        float background = BackgroundPosition(regions);
        float[] positions = new float[tokens];
        for (int t = 0; t < tokens; t++)
        {
            int best = 0;
            float bestValue = map[t];
            for (int r = 1; r < regions; r++)
            {
                float value = map[r * tokens + t];
                if (value > bestValue)
                {
                    best = r;
                    bestValue = value;
                }
            }
            positions[t] = best switch
            {
                0 => Scaled(map[t], firstMin, firstMax, 0f, SpeakerInterval),
                1 => Scaled(map[tokens + t], secondMin, secondMax, PositionRange - SpeakerInterval, PositionRange),
                _ => background,
            };
        }
        return positions;
    }

    /// <summary>The positions of one latent frame's audio tokens, <c>tracks·contextTokens</c> of them: the first speaker's tokens, then the
    /// second speaker's, then the silent track's at the background position.</summary>
    public static float[] KeyPositions(int tracks, int contextTokens, int regions)
    {
        if (tracks < LongCatSpeakers.MinTracks || contextTokens < 1)
            throw new ArgumentOutOfRangeException(nameof(tracks), "The audio has at least two tracks of at least one token.");
        float[] positions = new float[tracks * contextTokens];
        positions.AsSpan(0, contextTokens).Fill(FirstSpeakerKey);
        positions.AsSpan(contextTokens, contextTokens).Fill(SecondSpeakerKey);
        positions.AsSpan(2 * contextTokens).Fill(BackgroundPosition(regions));
        return positions;
    }

    /// <summary>The cosine and sine tables <c>[positions, headDim]</c> of the positions, one angle per pair of adjacent elements, written to both.
    /// The caller owns both.</summary>
    public static (Tensor Cos, Tensor Sin) Tables(float[] positions, int headDim)
    {
        ArgumentNullException.ThrowIfNull(positions);
        if (positions.Length < 1 || headDim < 2 || headDim % 2 != 0)
            throw new ArgumentException("The tables need at least one position and an even head width.", nameof(headDim));
        float[] frequencies = new float[headDim / 2];
        for (int i = 0; i < frequencies.Length; i++)
            frequencies[i] = 1f / MathF.Pow(Theta, (2 * i) / (float)headDim);
        Tensor cos = new(new TensorShape(positions.Length, headDim), DType.F32);
        Tensor sin = new(new TensorShape(positions.Length, headDim), DType.F32);
        unsafe
        {
            float* cosRows = (float*)cos.DataPointer, sinRows = (float*)sin.DataPointer;
            nint cosBase = (nint)cosRows, sinBase = (nint)sinRows;
            Parallel.For(0, positions.Length, row =>
            {
                float* c = (float*)cosBase + (long)row * headDim;
                float* s = (float*)sinBase + (long)row * headDim;
                for (int i = 0; i < frequencies.Length; i++)
                {
                    (float sine, float cosine) = MathF.SinCos(positions[row] * frequencies[i]);
                    c[2 * i] = c[2 * i + 1] = cosine;
                    s[2 * i] = s[2 * i + 1] = sine;
                }
            });
        }
        return (cos, sin);
    }

    private static float Scaled(float value, float min, float max, float low, float high) =>
        (value - min) / (max - min + Epsilon) * (high - low) + low;

    private static (float Min, float Max) Range(ReadOnlySpan<float> values)
    {
        float min = values[0], max = values[0];
        foreach (float value in values)
        {
            min = MathF.Min(min, value);
            max = MathF.Max(max, value);
        }
        return (min, max);
    }
}
