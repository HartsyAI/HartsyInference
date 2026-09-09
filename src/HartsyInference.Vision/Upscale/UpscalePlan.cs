namespace HartsyInference.Vision.Upscale;

/// <summary>How an upscale request maps onto an integer-factor model: the number of model passes to run and the
/// size the result ends at. A Real-ESRGAN generator only multiplies by its fixed factor, so a target is reached by
/// running enough passes to cover it and then resizing <em>down</em>; the plan never asks for a stretch past what
/// the final pass produced, because that would just be interpolation dressed up as super-resolution.</summary>
/// <param name="Passes">Model passes to run, at least one.</param>
/// <param name="Width">Final output width after the optional downsize.</param>
/// <param name="Height">Final output height after the optional downsize.</param>
public readonly record struct UpscalePlan(int Passes, int Width, int Height)
{
    /// <summary>Default cap on chained passes: two x4 passes already turn 1024 px into 16384 px.</summary>
    public const int DefaultMaxPasses = 2;

    /// <summary>Plans an upscale of a <paramref name="srcWidth"/> × <paramref name="srcHeight"/> image with a model of
    /// integer <paramref name="scale"/>. Without a target the plan is one pass at the model factor. With a target
    /// (either side may be null; the other is derived from the source aspect) the plan runs the fewest passes that
    /// cover the target on both axes, capped at <paramref name="maxPasses"/>, and reports the target clamped to
    /// what those passes produce.</summary>
    public static UpscalePlan Create(int srcWidth, int srcHeight, int scale, int? targetWidth, int? targetHeight,
        int maxPasses = DefaultMaxPasses)
    {
        if (srcWidth <= 0 || srcHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(srcWidth), $"Invalid source size {srcWidth}x{srcHeight}.");
        if (scale < 2)
            throw new ArgumentOutOfRangeException(nameof(scale), scale, "An upscale model's factor must be at least 2.");
        if (maxPasses < 1)
            throw new ArgumentOutOfRangeException(nameof(maxPasses), maxPasses, "At least one pass is required.");
        if (targetWidth is null && targetHeight is null)
        {
            return new UpscalePlan(1, srcWidth * scale, srcHeight * scale);
        }
        (int wantW, int wantH) = ResolveTarget(srcWidth, srcHeight, targetWidth, targetHeight);
        int passes = 1;
        long w = (long)srcWidth * scale, h = (long)srcHeight * scale;
        while ((w < wantW || h < wantH) && passes < maxPasses)
        {
            passes++;
            w *= scale;
            h *= scale;
        }
        return new UpscalePlan(passes, (int)Math.Min(wantW, w), (int)Math.Min(wantH, h));
    }

    private static (int Width, int Height) ResolveTarget(int srcWidth, int srcHeight, int? targetWidth, int? targetHeight)
    {
        if (targetWidth is <= 0 || targetHeight is <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetWidth), $"Invalid target size {targetWidth}x{targetHeight}.");
        int w = targetWidth ?? (int)Math.Max(1, Math.Round((double)targetHeight!.Value * srcWidth / srcHeight));
        int h = targetHeight ?? (int)Math.Max(1, Math.Round((double)targetWidth!.Value * srcHeight / srcWidth));
        return (w, h);
    }
}
