namespace HartsyInference.Core.Numerics;

/// <summary>Host 1-D linear resampling of a time-major <c>[frames, channels]</c> feature sequence, with the sampling
/// rule of <c>torch.nn.functional.interpolate(mode="linear", align_corners=True)</c>.</summary>
public static class LinearResample
{
    /// <summary>Resamples <paramref name="source"/> (<c>[sourceFrames, channels]</c>, row-major) to
    /// <paramref name="destination"/> (<c>[destinationFrames, channels]</c>). Output frame <c>i</c> reads source
    /// position <c>i * (sourceFrames - 1) / (destinationFrames - 1)</c>, so both end frames are kept exactly. The scale
    /// and the position are F32, as in ATen's CPU kernel, so the taps and weights match torch bit for bit.</summary>
    public static void AlignCorners(ReadOnlySpan<float> source, int sourceFrames, int channels, Span<float> destination,
        int destinationFrames)
    {
        if (sourceFrames < 1) throw new ArgumentOutOfRangeException(nameof(sourceFrames), sourceFrames, "Needs at least one source frame.");
        if (destinationFrames < 1) throw new ArgumentOutOfRangeException(nameof(destinationFrames), destinationFrames, "Needs at least one output frame.");
        if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels), channels, "Needs at least one channel.");
        if (source.Length != (long)sourceFrames * channels)
            throw new ArgumentException($"source has {source.Length} values, expected {(long)sourceFrames * channels}.", nameof(source));
        if (destination.Length != (long)destinationFrames * channels)
            throw new ArgumentException($"destination has {destination.Length} values, expected {(long)destinationFrames * channels}.", nameof(destination));

        float scale = destinationFrames > 1 ? (float)(sourceFrames - 1) / (destinationFrames - 1) : 0f;
        for (int i = 0; i < destinationFrames; i++)
        {
            float position = scale * i;
            int lower = (int)position;
            int upper = lower + (lower < sourceFrames - 1 ? 1 : 0);
            float upperWeight = Math.Min(Math.Max(position - lower, 0f), 1f);
            float lowerWeight = 1f - upperWeight;
            ReadOnlySpan<float> a = source.Slice(lower * channels, channels);
            ReadOnlySpan<float> b = source.Slice(upper * channels, channels);
            Span<float> o = destination.Slice(i * channels, channels);
            for (int c = 0; c < channels; c++) o[c] = lowerWeight * a[c] + upperWeight * b[c];
        }
    }

    /// <summary>The frame count of a clip of <paramref name="sourceFrames"/> frames at <paramref name="sourceRate"/> after
    /// resampling to <paramref name="destinationRate"/>: <c>int(frames / sourceRate * destinationRate)</c> in F64, the
    /// reference pipelines' rounding (truncation toward zero).</summary>
    public static int FramesAtRate(int sourceFrames, double sourceRate, double destinationRate)
        => (int)(sourceFrames / sourceRate * destinationRate);
}
