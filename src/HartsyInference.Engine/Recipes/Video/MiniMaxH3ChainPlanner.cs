using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Splits a requested duration longer than one MiniMax-H3 generation into a chain of generations, each
/// re-denoising only its own new frames while holding a copy of the previous segment's tail fixed. Pure geometry:
/// every length it emits is on H3's <c>17k+5</c> grid, so a segment's protected head covers whole latent tokens and
/// the seam lands on a token boundary rather than inside one.</summary>
public static class MiniMaxH3ChainPlanner
{
    /// <summary>Default frames of the previous segment carried into the next as fixed context. The smallest grid run
    /// that is more than a still, and the length this family's guide/mask canary was verified at.</summary>
    public const int DefaultContextFrames = 39;

    /// <summary>One generation in the chain.</summary>
    /// <param name="Index">Zero-based position in the chain.</param>
    /// <param name="FrameCount">Aligned frames this generation produces, protected head included.</param>
    /// <param name="ContextFrames">Leading frames copied from the previous segment and held fixed; zero for the first.</param>
    /// <param name="ContextLatentFrames">Video latent tokens those context frames occupy.</param>
    /// <param name="ContextAudioLatentFrames">Audio latent rows per channel those context frames occupy.</param>
    public readonly record struct Segment(
        int Index, int FrameCount, int ContextFrames, int ContextLatentFrames, int ContextAudioLatentFrames)
    {
        /// <summary>Frames this segment contributes to the assembled output; the protected head is context, not output.</summary>
        public int NewFrames => FrameCount - ContextFrames;
    }

    /// <summary>Splits <paramref name="targetFrames"/> into chained segments. A target that already fits one
    /// generation returns a single segment and no chaining.</summary>
    /// <param name="targetFrames">Frames the assembled output should reach, before grid alignment.</param>
    /// <param name="contextFrames">Frames carried between segments; must be on the <c>17k+5</c> grid.</param>
    /// <param name="maxSegmentFrames">Longest single generation to emit; snapped down onto the grid.</param>
    public static IReadOnlyList<Segment> Plan(int targetFrames,
        int contextFrames = DefaultContextFrames,
        int maxSegmentFrames = MiniMaxH3Geometry.TrainedFrameEnvelope)
    {
        if (targetFrames < 5)
        {
            throw new ArgumentOutOfRangeException(nameof(targetFrames), targetFrames,
                "A MiniMax-H3 chain needs at least 5 frames.");
        }
        if (maxSegmentFrames < 5 + 17)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSegmentFrames), maxSegmentFrames,
                "A MiniMax-H3 chain segment must be able to hold at least two grid steps.");
        }
        int capped = MiniMaxH3Geometry.SnapFrameCountDown(maxSegmentFrames);
        if (contextFrames < 5 || contextFrames % 17 != 5)
        {
            throw new ArgumentOutOfRangeException(nameof(contextFrames), contextFrames,
                "MiniMax-H3 chain context frames must be on the 17k+5 grid so the protected head covers whole latent tokens.");
        }
        if (contextFrames >= capped)
        {
            throw new ArgumentOutOfRangeException(nameof(contextFrames), contextFrames,
                $"MiniMax-H3 chain context of {contextFrames} frames leaves no new frames inside a {capped}-frame segment.");
        }

        List<Segment> segments = [];
        int first = Math.Min(MiniMaxH3Geometry.AlignFrameCount(targetFrames), capped);
        segments.Add(new Segment(0, first, 0, 0, 0));
        int produced = first;

        while (produced < targetFrames)
        {
            int wanted = MiniMaxH3Geometry.AlignFrameCount(targetFrames - produced + contextFrames);
            int frameCount = Math.Min(wanted, capped);
            segments.Add(new Segment(segments.Count, frameCount, contextFrames,
                MiniMaxH3Geometry.VideoLatentFrames(contextFrames),
                MiniMaxH3Geometry.AudioLatentFrames(contextFrames)));
            produced += frameCount - contextFrames;
        }
        return segments;
    }

    /// <summary>Frames the assembled output carries once every segment's protected head is dropped.</summary>
    public static int TotalFrames(IReadOnlyList<Segment> segments)
    {
        ArgumentNullException.ThrowIfNull(segments);
        int total = 0;
        foreach (Segment segment in segments)
        {
            total += segment.NewFrames;
        }
        return total;
    }

    /// <summary>One spatially-uniform mask value per video latent token: zero over the protected head, one after.</summary>
    public static float[] VideoMaskFrameValues(in Segment segment)
    {
        int latentT = MiniMaxH3Geometry.VideoLatentFrames(segment.FrameCount);
        if (segment.ContextLatentFrames > latentT)
        {
            throw new ArgumentOutOfRangeException(nameof(segment), segment.ContextLatentFrames,
                $"Protected head of {segment.ContextLatentFrames} latent frames exceeds the segment's {latentT}.");
        }
        float[] values = new float[latentT];
        values.AsSpan(segment.ContextLatentFrames).Fill(1f);
        return values;
    }

    /// <summary>One mask value per 40 Hz audio latent row: zero over the protected head, one after.</summary>
    public static float[] AudioMaskValues(in Segment segment)
    {
        int audioT = MiniMaxH3Geometry.AudioLatentFrames(segment.FrameCount);
        if (segment.ContextAudioLatentFrames > audioT)
        {
            throw new ArgumentOutOfRangeException(nameof(segment), segment.ContextAudioLatentFrames,
                $"Protected head of {segment.ContextAudioLatentFrames} audio rows exceeds the segment's {audioT}.");
        }
        float[] values = new float[audioT];
        values.AsSpan(segment.ContextAudioLatentFrames).Fill(1f);
        return values;
    }
}
