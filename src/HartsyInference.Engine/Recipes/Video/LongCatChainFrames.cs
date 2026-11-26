namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The output frames of a LongCat-Video-Avatar chain, assembled as each segment is decoded. Segment 0 keeps all its decoded frames; a later
/// segment keeps the frames after its context, as the reference does (<c>appended_frames = new_video_array[num_cond_frames:]</c>, the demo's segment
/// loop). Each segment's last frames become the next context, except after the last segment, which encodes nothing.</summary>
internal sealed class LongCatChainFrames
{
    private readonly List<byte[]> _frames = [];
    private readonly int _contextPixels;

    /// <param name="contextPixels">Pixel frames of context that a later segment begins with: <c>1 + 4·(contextLatents − 1)</c> (13 for four latents).</param>
    public LongCatChainFrames(int contextPixels)
    {
        if (contextPixels < 1)
            throw new ArgumentOutOfRangeException(nameof(contextPixels), contextPixels, "The context has at least one pixel frame.");
        _contextPixels = contextPixels;
    }

    /// <summary>Segments kept so far.</summary>
    public int Segments { get; private set; }

    /// <summary>The frames kept so far, in clip order.</summary>
    public byte[][] Frames => [.. _frames];

    /// <summary>Keeps one decoded segment's output frames and returns the next context. <paramref name="decoded"/> is the segment's whole decode;
    /// the first segment keeps all of it, a later one keeps the frames after its context. When <paramref name="last"/> is set the segment is the
    /// chain's last, nothing is encoded, and an empty context is returned.</summary>
    public float[] Step(byte[][] decoded, bool last, Func<byte[][], float[]> encodeContext)
    {
        ArgumentNullException.ThrowIfNull(decoded);
        ArgumentNullException.ThrowIfNull(encodeContext);
        int from = Segments == 0 ? 0 : _contextPixels;
        if (decoded.Length < from)
            throw new InvalidOperationException($"Segment {Segments} decoded {decoded.Length} frames; it needs at least {from} to drop its context.");
        _frames.AddRange(decoded[from..]);
        Segments++;
        return last ? [] : encodeContext(decoded);
    }
}
