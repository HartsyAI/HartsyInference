namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>The segment plan of a long-form LongCat-Video-Avatar chain. Segment 0 is a whole generation of <see cref="SegmentLatents"/> latent frames.
/// Each later segment runs one reference slot (latent 0, fixed), the last <see cref="ContextLatents"/> latents of the chain so far, and
/// <see cref="NewLatentsPerSegment"/> new latents. Latent <c>g</c> covers pixel frames <c>4g−3…4g</c> (latent 0 covers pixel 0): the VAE's temporal
/// compression of 4, with the first frame on its own.</summary>
public sealed record LongCatAvatarChainPlan
{
    /// <summary>Latent frames of the chain's context, carried between segments. Fixed at four (13 pixel frames) for LongCat-Video-Avatar.</summary>
    public const int DefaultContextLatents = 4;

    private LongCatAvatarChainPlan(int segmentLatents, int contextLatents, int segments)
    {
        SegmentLatents = segmentLatents;
        ContextLatents = contextLatents;
        Segments = segments;
    }

    /// <summary>Latent frames of one generation (segment 0 and the new frames of a later segment's run, before the reference is dropped).</summary>
    public int SegmentLatents { get; }

    /// <summary>Latent frames of context carried between segments.</summary>
    public int ContextLatents { get; }

    /// <summary>How many segments the chain runs, segment 0 included.</summary>
    public int Segments { get; }

    /// <summary>New latent frames a later segment contributes.</summary>
    public int NewLatentsPerSegment => SegmentLatents - ContextLatents;

    /// <summary>Latent frames of the whole chain.</summary>
    public int TotalLatents => SegmentLatents + (Segments - 1) * NewLatentsPerSegment;

    /// <summary>Pixel frames of the whole chain: <c>4·(TotalLatents − 1) + 1</c>. The decoded clip is trimmed to the requested count.</summary>
    public int TotalPixelFrames => 4 * (TotalLatents - 1) + 1;

    /// <summary>Latent frames of a later segment's run: its reference slot, its context and its new frames.</summary>
    public int RunLatents => 1 + ContextLatents + NewLatentsPerSegment;

    /// <summary>Audio windows (pixel frames) of a later segment's run, the reference's <c>num_frames</c>: <c>4·(SegmentLatents − 1) + 1</c>.</summary>
    public int RunWindows => 4 * (SegmentLatents - 1) + 1;

    /// <summary>The plan for a chain of at least <paramref name="totalPixelFrames"/> pixel frames, from segments of
    /// <paramref name="segmentLatents"/> latent frames.</summary>
    public static LongCatAvatarChainPlan For(int segmentLatents, int totalPixelFrames, int contextLatents = DefaultContextLatents)
    {
        if (contextLatents < 1 || contextLatents >= segmentLatents)
            throw new ArgumentOutOfRangeException(nameof(contextLatents), contextLatents, "The context must be shorter than a segment.");
        if (totalPixelFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(totalPixelFrames), totalPixelFrames, "The chain needs at least one frame.");
        int newPerSegment = segmentLatents - contextLatents;
        int neededLatents = (totalPixelFrames - 1 + 3) / 4 + 1;
        int segments = 1;
        if (neededLatents > segmentLatents)
            segments += (neededLatents - segmentLatents + newPerSegment - 1) / newPerSegment;
        return new LongCatAvatarChainPlan(segmentLatents, contextLatents, segments);
    }

    /// <summary>The pixel frames of latent <paramref name="latent"/>: pixel 0 for latent 0, and four pixel frames for every other latent.</summary>
    public static IEnumerable<int> PixelsOfLatent(int latent)
    {
        if (latent == 0)
        {
            yield return 0;
            yield break;
        }
        for (int k = 0; k < 4; k++)
            yield return 4 * latent - 3 + k;
    }

    /// <summary>The global latent ids of later segment <paramref name="segment"/>'s run, in run order: the reference (latent 0), the context, then the new
    /// frames.</summary>
    public int[] RunLatentIds(int segment)
    {
        if (segment < 1 || segment >= Segments)
            throw new ArgumentOutOfRangeException(nameof(segment), segment, "A later segment index is from 1 to Segments − 1.");
        int firstNew = SegmentLatents + (segment - 1) * NewLatentsPerSegment;
        int[] ids = new int[RunLatents];
        ids[0] = 0;
        for (int k = 0; k < ContextLatents; k++)
            ids[1 + k] = firstNew - ContextLatents + k;
        for (int k = 0; k < NewLatentsPerSegment; k++)
            ids[1 + ContextLatents + k] = firstNew + k;
        return ids;
    }

    /// <summary>The audio windows (pixel frames) of a later segment's run, in order: <c>4·(SegmentLatents − 1) + 1</c> contiguous frames from
    /// <c>segment · 4 · NewLatentsPerSegment</c>. This is the reference's layout (meituan-longcat/LongCat-Video @ 58eb336, the demo's
    /// <c>audio_start_idx += stride · (num_frames − num_cond_frames)</c>): the first <c>4·(ContextLatents − 1) + 1</c> windows are the context,
    /// the rest the new frames. The reference slot takes no window of its own; the audio projection gives it a copy of the first frame's tokens
    /// (see <see cref="ReferenceSlotTokens"/>). Equals <c>LongCatAudioFeatures.SegmentStartFrame(segment)</c> for the Whisper layout.</summary>
    public int[] RunPixelIds(int segment)
    {
        if (segment < 1 || segment >= Segments)
            throw new ArgumentOutOfRangeException(nameof(segment), segment, "A later segment index is from 1 to Segments − 1.");
        int start = segment * 4 * NewLatentsPerSegment;
        int count = RunWindows;
        int[] pixels = new int[count];
        for (int i = 0; i < count; i++)
            pixels[i] = start + i;
        return pixels;
    }

    /// <summary>The audio tokens of a later segment's run, <c>[RunLatents · valuesPerFrame]</c> values, from the tokens of its audio windows
    /// <c>[(RunLatents − 1) · valuesPerFrame]</c> values: the reference slot takes a copy of the first frame's tokens, placed in front, as the
    /// reference's audio projection does when a reference latent is present.</summary>
    public static float[] ReferenceSlotTokens(float[] tokens, int valuesPerFrame)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (valuesPerFrame < 1 || tokens.Length == 0 || tokens.Length % valuesPerFrame != 0)
            throw new ArgumentException("The tokens must be whole frames of valuesPerFrame values.", nameof(tokens));
        float[] result = new float[tokens.Length + valuesPerFrame];
        Array.Copy(tokens, 0, result, 0, valuesPerFrame);
        Array.Copy(tokens, 0, result, valuesPerFrame, tokens.Length);
        return result;
    }

    /// <summary>Joins the chain's segments into one <c>[channels, TotalLatents, hw]</c> latent: segment 0 whole, then each later segment's new frames
    /// (the run's reference and context dropped). <paramref name="segments"/> is what <c>LongCatAvatarPipeline.Chain</c> returns.</summary>
    public float[] Join(IReadOnlyList<float[]> segments, int channels, int hw)
    {
        ArgumentNullException.ThrowIfNull(segments);
        if (segments.Count != Segments)
            throw new ArgumentException($"The chain has {Segments} segments; got {segments.Count}.", nameof(segments));
        int total = TotalLatents;
        float[] result = new float[channels * total * hw];
        int at = 0;
        for (int s = 0; s < segments.Count; s++)
        {
            int frames = s == 0 ? SegmentLatents : NewLatentsPerSegment;
            float[] part = segments[s];
            if (part.Length != channels * frames * hw)
                throw new ArgumentException($"Segment {s} holds {part.Length} values; expected {channels * frames * hw}.", nameof(segments));
            for (int c = 0; c < channels; c++)
                Array.Copy(part, c * frames * hw, result, (c * total + at) * hw, frames * hw);
            at += frames;
        }
        return result;
    }
}
