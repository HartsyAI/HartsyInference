namespace HartsyInference.Core.Imaging;

/// <summary>Qwen2-VL's <c>smart_resize</c> and the pixel-budget rules built on it, shared by every Qwen-VL-style vision tower
/// (Qwen2.5-VL, Qwen3-VL, MOSS-VL, MiniMax-H3, …).</summary>
public static class SmartResize
{
    /// <summary>The largest long/short side ratio Qwen2-VL's processor accepts.</summary>
    public const double MaxAspectRatio = 200.0;

    /// <summary>An exact port of <c>transformers</c>' Qwen2-VL <c>smart_resize</c>: both sides become multiples of
    /// <paramref name="factor"/> (Python's half-to-even <c>round</c>), the area is brought inside
    /// [<paramref name="minPixels"/>, <paramref name="maxPixels"/>] with the aspect kept as closely as possible.</summary>
    /// <exception cref="ArgumentException">A side is not positive, or the aspect ratio exceeds <see cref="MaxAspectRatio"/>.</exception>
    public static (int Height, int Width) Qwen2Vl(int height, int width, int factor, long minPixels, long maxPixels)
    {
        if (height <= 0 || width <= 0 || factor <= 0)
            throw new ArgumentException($"smart_resize needs positive sizes, got {height}x{width} with factor {factor}.");
        double aspect = (double)Math.Max(height, width) / Math.Min(height, width);
        if (aspect > MaxAspectRatio)
            throw new ArgumentException($"The image aspect ratio {aspect:G4} exceeds {MaxAspectRatio}; Qwen-VL processors refuse it.");
        long hBar = (long)Math.Round((double)height / factor, MidpointRounding.ToEven) * factor;
        long wBar = (long)Math.Round((double)width / factor, MidpointRounding.ToEven) * factor;
        if (hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / maxPixels);
            hBar = Math.Max(factor, (long)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (long)Math.Floor(width / beta / factor) * factor);
        }
        else if (hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt((double)minPixels / ((double)height * width));
            hBar = (long)Math.Ceiling(height * beta / factor) * factor;
            wBar = (long)Math.Ceiling(width * beta / factor) * factor;
        }
        return (checked((int)hBar), checked((int)wBar));
    }

    /// <summary>The video form of <see cref="Qwen2Vl"/> (MOSS-VL's video processor): one size for every frame of a clip of
    /// <paramref name="frames"/> frames. The per-frame cap applies first, then the area of the whole clip is brought inside
    /// [<paramref name="minPixels"/>, <paramref name="maxPixels"/>], then the per-frame floor.</summary>
    /// <exception cref="ArgumentException">Fewer frames than the temporal factor, a side below <paramref name="factor"/>, or an aspect
    /// ratio past <see cref="MaxAspectRatio"/>.</exception>
    public static (int Height, int Width) Qwen2VlVideo(int frames, int height, int width, int temporalFactor, int factor, long minPixels,
        long maxPixels, long? perFrameMinPixels, long? perFrameMaxPixels)
    {
        if (frames < temporalFactor)
            throw new ArgumentException($"A clip of {frames} frames is shorter than the temporal patch of {temporalFactor}.");
        if (height < factor || width < factor)
            throw new ArgumentException($"Video frames of {height}x{width} are smaller than the {factor}-pixel patch block.");
        double aspect = (double)Math.Max(height, width) / Math.Min(height, width);
        if (aspect > MaxAspectRatio)
            throw new ArgumentException($"The video aspect ratio {aspect:G4} exceeds {MaxAspectRatio}; Qwen-VL processors refuse it.");
        long hBar = (long)Math.Round((double)height / factor, MidpointRounding.ToEven) * factor;
        long wBar = (long)Math.Round((double)width / factor, MidpointRounding.ToEven) * factor;
        long tBar = (long)Math.Round((double)frames / temporalFactor, MidpointRounding.ToEven) * temporalFactor;

        if (perFrameMaxPixels is { } perMax && hBar * wBar > perMax)
        {
            double beta = Math.Sqrt((double)height * width / perMax);
            hBar = Math.Max(factor, (long)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (long)Math.Floor(width / beta / factor) * factor);
        }
        if (tBar * hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt((double)frames * height * width / maxPixels);
            hBar = Math.Max(factor, (long)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (long)Math.Floor(width / beta / factor) * factor);
        }
        else if (tBar * hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt(minPixels / ((double)frames * height * width));
            hBar = (long)Math.Ceiling(height * beta / factor) * factor;
            wBar = (long)Math.Ceiling(width * beta / factor) * factor;
        }
        if (perFrameMinPixels is { } perMin && hBar * wBar < perMin)
        {
            double beta = Math.Sqrt((double)perMin / ((double)height * width));
            hBar = (long)Math.Ceiling(height * beta / factor) * factor;
            wBar = (long)Math.Ceiling(width * beta / factor) * factor;
        }
        return (checked((int)hBar), checked((int)wBar));
    }

    /// <summary>The engine's long-standing Qwen-VL variant: sides below <paramref name="factor"/> are raised to it first, and the
    /// longer side is optionally capped at <paramref name="maxSideLength"/> before snapping. Kept for the callers whose parity was
    /// measured against it (Qwen3-VL text encoders, Boogu-Image).</summary>
    public static (int Height, int Width) Compute(int height, int width, int factor, long minPixels, long maxPixels, int maxSideLength = 0)
    {
        if (height < factor) height = factor;
        if (width < factor) width = factor;

        if (maxSideLength > 0 && Math.Max(height, width) > maxSideLength)
        {
            double beta = (double)Math.Max(height, width) / maxSideLength;
            height = Math.Max(factor, (int)Math.Floor(height / beta));
            width = Math.Max(factor, (int)Math.Floor(width / beta));
        }

        int hBar = Math.Max(factor, (int)Math.Round((double)height / factor) * factor);
        int wBar = Math.Max(factor, (int)Math.Round((double)width / factor) * factor);

        if ((long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt((double)minPixels / (height * width));
            hBar = (int)Math.Ceiling(height * beta / factor) * factor;
            wBar = (int)Math.Ceiling(width * beta / factor) * factor;
        }
        return (hBar, wBar);
    }

    /// <summary>Splits a batch pixel budget across images in proportion to their original areas, then clamps each share to
    /// [<paramref name="floorPixels"/>, <paramref name="ceilingPixels"/>]. This is MOSS-VL's <c>multi_image_max_pixels</c> and
    /// <c>video_max_pixels</c> rule: each result is that image's <c>max_pixels</c> for <see cref="Qwen2Vl"/>.</summary>
    /// <param name="sizes">Each image's original (height, width).</param>
    /// <param name="totalBudget">The batch budget. The reference (<c>processing_moss_vl.py</c>, <c>MossVLImageProcessorFast._preprocess</c>):
    /// <c>proportional_max_pixels = int(multi_image_max_pixels * (original_pixels / total_original_pixels))</c>, then
    /// <c>max(…, min_pixels)</c>, then <c>min(…, max_pixels)</c>; the same truncation and clamps as here.</param>
    public static long[] ShareBudget(ReadOnlySpan<(int Height, int Width)> sizes, long totalBudget, long floorPixels, long ceilingPixels)
    {
        double total = 0;
        foreach ((int h, int w) in sizes) total += (double)h * w;
        long[] shares = new long[sizes.Length];
        for (int i = 0; i < sizes.Length; i++)
        {
            long share = total > 0 ? (long)(totalBudget * ((double)sizes[i].Height * sizes[i].Width / total)) : totalBudget;
            shares[i] = Math.Min(Math.Max(share, floorPixels), ceilingPixels);
        }
        return shares;
    }
}
