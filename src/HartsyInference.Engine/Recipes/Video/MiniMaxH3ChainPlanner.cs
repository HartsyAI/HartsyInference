using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Splits a duration longer than one MiniMax-H3 generation into chained generations that each re-denoise
/// only their own new frames. Every length lands on H3's <c>17k+5</c> grid, so a protected head always covers whole
/// latent tokens rather than ending inside one.</summary>
public static class MiniMaxH3ChainPlanner
{
    /// <summary>Default frames carried into the next segment; the smallest grid run that is more than a still.</summary>
    public const int DefaultContextFrames = 39;

    /// <summary>One generation in the chain.</summary>
    /// <param name="FrameCount">Aligned frames this generation produces, protected head included.</param>
    /// <param name="ContextFrames">Leading frames copied from the previous segment and held fixed; zero for the first.</param>
    public readonly record struct Segment(
        int Index, int FrameCount, int ContextFrames, int ContextLatentFrames, int ContextAudioLatentFrames)
    {
        /// <summary>Frames this segment contributes to the assembled output; the protected head is context, not output.</summary>
        public int NewFrames => FrameCount - ContextFrames;
    }

    /// <summary>Splits <paramref name="targetFrames"/> into chained segments; a target that already fits one
    /// generation returns a single segment.</summary>
    /// <param name="contextFrames">Frames carried between segments; must be on the <c>17k+5</c> grid.</param>
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
        // 40 Hz audio rows against 24 fps video: only a context that is a whole number of BOTH lands the audio join
        // on a row boundary. Off it, the protected rows span a different stretch of time than the carried frames do
        // and the two streams hand over out of alignment.
        if (MiniMaxH3Geometry.AudioLatentFrames(contextFrames) * MiniMaxH3Geometry.Fps
            != contextFrames * MiniMaxH3Geometry.AudioLatentFps)
        {
            throw new ArgumentOutOfRangeException(nameof(contextFrames), contextFrames,
                $"MiniMax-H3 chain context of {contextFrames} frames does not cover whole audio latent rows; "
                + "use a grid length that is also a whole number of 40 Hz rows (39, 90, 141, …).");
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

    /// <summary>One mask value per video latent token: zero over the protected head, one after.</summary>
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

    /// <summary>Audio latent rows the mask ramps over instead of stepping, at the end of the protected head. A hard
    /// edge leaves the latent itself discontinuous where preserved rows meet freshly denoised ones, and the audio VAE
    /// renders that step as an impulse — the pop ComfyUI's <c>audio_feather_ticks</c> exists to remove. Video takes no
    /// feather: a frame boundary carries no phase, so only the soundtrack hears the edge.</summary>
    public const int AudioFeatherRows = 3;

    /// <summary>One mask value per 40 Hz audio latent row: zero over the protected head, one after, raised-cosine
    /// across <see cref="AudioFeatherRows"/> rows at the join.</summary>
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
        if (segment.ContextAudioLatentFrames == 0)
        {
            return values;
        }
        int feather = Math.Min(AudioFeatherRows, segment.ContextAudioLatentFrames);
        int hard = segment.ContextAudioLatentFrames - feather;
        for (int i = 1; i <= feather; i++)
        {
            values[hard + i - 1] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * i / feather));
        }
        return values;
    }
}
