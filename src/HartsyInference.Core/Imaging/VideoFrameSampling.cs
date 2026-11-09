using System.Globalization;

namespace HartsyInference.Core.Imaging;

/// <summary>How a video is sampled into frames for a vision-language model.</summary>
/// <param name="Fps">Frames sampled per second of video.</param>
/// <param name="MinFrames">The fewest frames kept.</param>
/// <param name="MaxFrames">The most frames kept.</param>
/// <param name="VideoMaxPixels">The pixel budget shared by every sampled frame.</param>
/// <param name="MinFramePixels">The smallest area one frame may be resized to (the processor's <c>shortest_edge</c>).</param>
/// <param name="MaxFramePixels">The largest area one frame may be resized to (the processor's <c>longest_edge</c>).</param>
public sealed record VideoFrameSamplingOptions(
    double Fps = 1.0,
    int MinFrames = 1,
    int MaxFrames = 256,
    long VideoMaxPixels = 65_536_000,
    long MinFramePixels = 4096,
    long MaxFramePixels = 16_777_216);

/// <summary>One sampled frame: its index in the decoded stream and its presentation time.</summary>
public readonly record struct SampledFrame(int Index, double TimestampSeconds);

/// <summary>One decoded frame with its presentation time.</summary>
public readonly record struct TimedRgbFrame(RawRgbFrame Frame, double TimestampSeconds);

/// <summary>A source of timed RGB frames: a decoded file, a camera, a directory of images.</summary>
public interface IVideoFrameSource
{
    /// <summary>The frames in presentation order.</summary>
    IAsyncEnumerable<TimedRgbFrame> ReadAsync(CancellationToken cancel = default);
}

/// <summary>Frame selection and sizing for video input to Qwen-VL-style towers (the rule MOSS-VL's video processor uses).</summary>
public static class VideoFrameSampling
{
    /// <summary>Picks frames from a decoded stream of <paramref name="totalFrames"/> frames at <paramref name="sourceFps"/>:
    /// <c>n = ceil(duration · fps − 1e-6)</c> clamped to [<c>MinFrames</c>, min(<c>MaxFrames</c>, <c>VideoMaxPixels / MinFramePixels</c>,
    /// <paramref name="totalFrames"/>)], indices <c>int(linspace(0, total − 1, n))</c> with duplicates removed, and each timestamp
    /// its index over <paramref name="sourceFps"/>.</summary>
    public static SampledFrame[] Plan(int totalFrames, double sourceFps, VideoFrameSamplingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (totalFrames <= 0) return [];
        if (sourceFps <= 0) throw new ArgumentOutOfRangeException(nameof(sourceFps), sourceFps, "The source frame rate must be positive.");
        double duration = totalFrames / sourceFps;
        long budgetFrames = options.MinFramePixels > 0 ? options.VideoMaxPixels / options.MinFramePixels : long.MaxValue;
        long upper = Math.Min(Math.Min(options.MaxFrames, budgetFrames), totalFrames);
        long n = (long)Math.Ceiling(duration * options.Fps - 1e-6);
        n = Math.Clamp(n, Math.Min(options.MinFrames, upper), Math.Max(upper, 1));

        List<SampledFrame> frames = new((int)n);
        int last = -1;
        // numpy linspace(0, total-1, n): i·step with the last value pinned to the end, then astype(int32) truncates.
        double step = n == 1 ? 0 : (double)(totalFrames - 1) / (n - 1);
        for (long i = 0; i < n; i++)
        {
            double at = i == n - 1 && n > 1 ? totalFrames - 1 : i * step;
            int index = (int)at;
            if (index == last) continue;
            frames.Add(new SampledFrame(index, index / sourceFps));
            last = index;
        }
        return [.. frames];
    }

    /// <summary>One size for every sampled frame (<see cref="SmartResize.Qwen2VlVideo"/>): the per-frame cap <c>MaxFramePixels</c>, then the
    /// whole clip (<paramref name="frameCount"/> frames) within <c>VideoMaxPixels</c>, never below <c>MinFramePixels</c> per frame.</summary>
    public static (int Height, int Width) FrameSize(int height, int width, int frameCount, int factor, VideoFrameSamplingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount), frameCount, "At least one frame is needed.");
        return SmartResize.Qwen2VlVideo(frameCount, height, width, temporalFactor: 1, factor, options.MinFramePixels, options.VideoMaxPixels,
            options.MinFramePixels, options.MaxFramePixels);
    }
}

/// <summary>Timestamp text for vision prompts, formatted as Python's <c>f"{t:.1f}"</c> (half-to-even on the decimal digit).</summary>
public static class VideoTimestampText
{
    /// <summary>Seconds with one decimal, e.g. <c>"12.5"</c>, rounded as Python rounds the exact binary value: 0.25 → <c>"0.2"</c>
    /// (a true tie, to even) and 0.35 → <c>"0.3"</c> (stored as 0.34999…).</summary>
    public static string Seconds1(double seconds)
    {
        if (!double.IsFinite(seconds)) throw new ArgumentOutOfRangeException(nameof(seconds), seconds, "A timestamp must be finite.");
        bool negative = seconds < 0;
        long bits = BitConverter.DoubleToInt64Bits(Math.Abs(seconds));
        int exponent = (int)((bits >> 52) & 0x7FF);
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0) exponent = 1; else mantissa |= 1L << 52;
        exponent -= 1075;   // value = mantissa · 2^exponent, exactly

        // tenths = round_half_even(value · 10), computed exactly with big integers.
        System.Numerics.BigInteger scaled = (System.Numerics.BigInteger)mantissa * 10;
        System.Numerics.BigInteger tenths;
        if (exponent >= 0)
        {
            tenths = scaled << exponent;
        }
        else
        {
            System.Numerics.BigInteger denominator = System.Numerics.BigInteger.One << -exponent;
            tenths = System.Numerics.BigInteger.DivRem(scaled, denominator, out System.Numerics.BigInteger remainder);
            int half = (remainder * 2).CompareTo(denominator);
            if (half > 0 || (half == 0 && !tenths.IsEven)) tenths += 1;
        }
        System.Numerics.BigInteger whole = System.Numerics.BigInteger.DivRem(tenths, 10, out System.Numerics.BigInteger digit);
        string sign = negative && !tenths.IsZero ? "-" : string.Empty;
        return string.Create(CultureInfo.InvariantCulture, $"{sign}{whole}.{digit}");
    }
}
