using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>The multi-speaker region masks of LongCat-Video-Avatar on the token grid: the demo's masks in the source portrait's pixels
/// (<c>run_demo_avatar_multi_audio_to_video.py</c>), the pipeline's cover resize and centre crop to the generation size
/// (<c>_resize_and_centercrop_tensor</c>: bilinear, half-pixel centres), then the DiT's nearest-neighbour step to one value per token, set where the
/// result is above zero (meituan-longcat/LongCat-Video @ 58eb336).</summary>
/// <remarks>The masks are only ever read at the token samples and tested for being above zero, so the resized masks are not built: each sample
/// looks at the source pixels its bilinear value would weigh.</remarks>
public static class LongCatSpeakerMasks
{
    /// <summary>The margin of the default regions, as a fraction of the image height and of half its width.</summary>
    private const double DefaultMargin = 0.1;

    /// <summary>The reference's regions for a portrait with no boxes: the left and the right half of the image, each inset by a tenth of the height
    /// and of the half width.</summary>
    public static (LongCatSpeakerBox First, LongCatSpeakerBox Second) DefaultBoxes(int height, int width)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        int half = width / 2;
        int top = (int)(height * DefaultMargin), bottom = (int)(height * (1 - DefaultMargin));
        return (new LongCatSpeakerBox(top, (int)(half * DefaultMargin), bottom, (int)(half * (1 - DefaultMargin))),
            new LongCatSpeakerBox(top, (int)(half * DefaultMargin + half), bottom, (int)(half * (1 - DefaultMargin) + half)));
    }

    /// <summary>The regions on the token grid: the first speaker, the second speaker, the background (outside both speakers' regions), then one per
    /// box in <paramref name="others"/>. Both speaker boxes are given, or neither (the defaults); silent people need the speakers' boxes, since the
    /// reference halves the mask width with the default boxes and then cannot stack their masks.</summary>
    /// <param name="sourceHeight">The portrait's height, the pixels the boxes are in.</param>
    /// <param name="sourceWidth">The portrait's width.</param>
    /// <param name="first">The first speaker's box.</param>
    /// <param name="second">The second speaker's box.</param>
    /// <param name="others">The silent people's boxes.</param>
    /// <param name="height">The generation height in pixels.</param>
    /// <param name="width">The generation width in pixels.</param>
    /// <param name="gridHeight">Token rows of a latent frame.</param>
    /// <param name="gridWidth">Token columns of a latent frame.</param>
    /// <returns>The regions' count and their masks <c>[regions, gridHeight·gridWidth]</c> of zeros and ones.</returns>
    public static (int Regions, float[] Masks) TokenMasks(int sourceHeight, int sourceWidth, LongCatSpeakerBox? first, LongCatSpeakerBox? second,
        IReadOnlyList<LongCatSpeakerBox> others, int height, int width, int gridHeight, int gridWidth)
    {
        ArgumentNullException.ThrowIfNull(others);
        if (sourceHeight < 1 || sourceWidth < 1 || height < 1 || width < 1 || gridHeight < 1 || gridWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(sourceHeight), "Every size must be positive.");
        if (first.HasValue != second.HasValue)
            throw new ArgumentException("Give both speakers' boxes, or neither.", nameof(second));
        if (others.Count > 0 && !first.HasValue)
            throw new ArgumentException("Silent people's boxes need both speakers' boxes.", nameof(others));
        (LongCatSpeakerBox one, LongCatSpeakerBox two) = first.HasValue
            ? (first.Value, second!.Value)
            : DefaultBoxes(sourceHeight, sourceWidth);
        // A default box of a very small image can be empty; a given box cannot.
        if (first.HasValue && (!one.IsValid || !two.IsValid))
            throw new ArgumentException("A box needs non-negative corners with YMax above YMin and XMax above XMin.", nameof(first));
        foreach (LongCatSpeakerBox other in others)
            if (!other.IsValid)
                throw new ArgumentException("A box needs non-negative corners with YMax above YMin and XMax above XMin.", nameof(others));

        // The cover resize: the side that fits is scaled to the target and the other is truncated, then centre cropped.
        int resizedHeight, resizedWidth;
        if (width / (double)height > sourceWidth / (double)sourceHeight)
        {
            resizedWidth = width;
            resizedHeight = (int)((long)sourceHeight * width / (double)sourceWidth);
        }
        else
        {
            resizedHeight = height;
            resizedWidth = (int)((long)sourceWidth * height / (double)sourceHeight);
        }
        int top = (resizedHeight - height) / 2, left = (resizedWidth - width) / 2;

        int regions = LongCatSpeakers.MinRegions + others.Count;
        int tokens = gridHeight * gridWidth;
        float[] masks = new float[regions * tokens];
        float nearestY = (float)height / gridHeight, nearestX = (float)width / gridWidth;
        for (int gy = 0; gy < gridHeight; gy++)
        {
            int y = Math.Min((int)MathF.Floor(gy * nearestY), height - 1) + top;
            (int y0, int y1, bool useY1) = Taps(y, sourceHeight, resizedHeight);
            for (int gx = 0; gx < gridWidth; gx++)
            {
                int x = Math.Min((int)MathF.Floor(gx * nearestX), width - 1) + left;
                (int x0, int x1, bool useX1) = Taps(x, sourceWidth, resizedWidth);
                int token = gy * gridWidth + gx;
                for (int region = 0; region < regions; region++)
                {
                    bool set = Inside(region, y0, x0) || (useX1 && Inside(region, y0, x1)) || (useY1 && Inside(region, y1, x0))
                        || (useY1 && useX1 && Inside(region, y1, x1));
                    masks[region * tokens + token] = set ? 1f : 0f;
                }
            }
        }
        return (regions, masks);

        bool Inside(int region, int y, int x) => region switch
        {
            0 => one.Contains(y, x),
            1 => two.Contains(y, x),
            2 => !one.Contains(y, x) && !two.Contains(y, x),
            _ => others[region - LongCatSpeakers.MinRegions].Contains(y, x),
        };
    }

    /// <summary>The two source indices a bilinear sample at <paramref name="destination"/> reads along one axis (half-pixel centres), and whether
    /// the second one carries any weight.</summary>
    private static (int Low, int High, bool UsesHigh) Taps(int destination, int source, int resized)
    {
        float scale = (float)source / resized;
        float position = MathF.Max(scale * (destination + 0.5f) - 0.5f, 0f);
        int low = Math.Min((int)position, source - 1);
        int high = Math.Min(low + 1, source - 1);
        return (low, high, position - low > 0f);
    }
}
