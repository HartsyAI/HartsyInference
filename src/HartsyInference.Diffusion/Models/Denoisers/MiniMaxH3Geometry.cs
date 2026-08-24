namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>MiniMax-H3's geometry contract, transcribed from the reference nodes. Every field of a
/// <see cref="MiniMaxH3GenerationRequest"/> has to come from here: the frame count lives on a coarse grid, the video
/// latent frame count is NOT <c>frames / 4</c>, and the audio length is derived from the aligned frame count rather
/// than from whatever the caller asked for.</summary>
public static class MiniMaxH3Geometry
{
    /// <summary>Pixel dimensions round to this. The DiT patch is 2x2 in LATENT space over a 16x-compressed VAE, so a
    /// multiple of 16 that is not a multiple of 32 gives an odd latent axis and the patchifier drops the last row or
    /// column.</summary>
    public const int CanvasMultiple = 32;

    /// <summary>Short edge of the nominal canvas.</summary>
    public const int BaseShortEdge = 768;

    /// <summary>Area cap; a larger request is scaled down to it before rounding.</summary>
    public const int MaxPixels = 768 * 1344;

    public const int Fps = 24;

    public const int AudioLatentFps = 40;

    /// <summary>Longest clip the model was trained on (~15 s at <see cref="Fps"/>), on the 17k+5 grid. Not a limit —
    /// past it is allowed and only warned about, since whether a length FITS is the pre-flight VRAM estimate's
    /// question (<see cref="MiniMaxH3ActivationEstimate"/>) and this constant is purely a quality signal.</summary>
    public const int TrainedFrameEnvelope = 362;

    /// <summary>Frame counts live on the <c>17k + 5</c> grid; anything else is rounded up onto it.</summary>
    public static int AlignFrameCount(int frames)
    {
        int n = Math.Max(5, frames);
        while (n % 17 != 5)
        {
            n++;
        }
        return n;
    }

    /// <summary>Video latent frames for an aligned frame count. Each latent token spans a different number of pixel
    /// frames (the <c>{1,4,4,4,4}</c> cycle), so 5 tokens cover 17 frames — hence 5 per 17, plus the 2-token base.</summary>
    public static int VideoLatentFrames(int alignedFrameCount) =>
        alignedFrameCount <= 5 ? 2 : (alignedFrameCount - 5) / 17 * 5 + 2;

    /// <summary>Audio latent frames covering an aligned frame count at 24 fps.</summary>
    public static int AudioLatentFrames(int alignedFrameCount) =>
        Math.Max(1, (int)Math.Round((double)alignedFrameCount / Fps * AudioLatentFps));

    /// <summary>The canvas H3 actually generates for a requested aspect: a 768 short edge, capped at
    /// <see cref="MaxPixels"/>, each axis rounded to <see cref="CanvasMultiple"/>.</summary>
    public static (int Width, int Height) AdaptCanvas(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "MiniMax-H3 needs a positive canvas.");
        }
        double ratio = (double)width / height;
        (double w, double h) = ratio >= 1.0 ? (BaseShortEdge * ratio, (double)BaseShortEdge)
            : ((double)BaseShortEdge, BaseShortEdge / ratio);
        if (w * h > MaxPixels)
        {
            double s = Math.Sqrt(MaxPixels / (w * h));
            w *= s;
            h *= s;
        }
        return (Round(w), Round(h));
    }

    /// <summary>The requested canvas honoured as asked, except that an area above <see cref="MaxPixels"/> is scaled
    /// down to it preserving aspect, each axis rounded to <see cref="CanvasMultiple"/>. Unlike
    /// <see cref="AdaptCanvas"/> this never renormalises the short edge to <see cref="BaseShortEdge"/> — a generation
    /// keeps the size the caller chose, so this is what the main path uses and <see cref="AdaptCanvas"/> stays the
    /// reference-clip rule.</summary>
    public static (int Width, int Height) ClampToMaxArea(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "MiniMax-H3 needs a positive canvas.");
        }
        double s = (long)width * height > MaxPixels ? Math.Sqrt((double)MaxPixels / ((double)width * height)) : 1.0;
        int w = Round(width * s), h = Round(height * s);
        if ((long)w * h <= MaxPixels)
        {
            return (w, h);
        }
        // Rounding to nearest keeps the aspect closest but can round BOTH axes up and land back above the cap — a
        // request just under it can too (1300x790 rounds to 1312x800). Re-round downward, which cannot exceed the
        // cap because each axis is then at most its exact aspect-preserving value.
        w = Floor(width * s);
        h = Floor(height * s);
        // ...unless an axis bottomed out at the grid minimum, which no longer tracks the aspect at all. Only the
        // other axis can move, so there is no "which one" choice to make here.
        while ((long)w * h > MaxPixels && (w > CanvasMultiple || h > CanvasMultiple))
        {
            if (w > CanvasMultiple) { w -= CanvasMultiple; }
            else { h -= CanvasMultiple; }
        }
        return (w, h);
    }

    /// <summary>Rounds a pixel axis DOWN onto <see cref="CanvasMultiple"/> without letting it collapse to zero.</summary>
    public static int Floor(double pixels) =>
        Math.Max(CanvasMultiple, (int)(pixels / CanvasMultiple) * CanvasMultiple);

    /// <summary>Rounds a pixel axis onto <see cref="CanvasMultiple"/> without letting it collapse to zero.</summary>
    public static int Round(double pixels) =>
        Math.Max(CanvasMultiple, (int)Math.Round(pixels / CanvasMultiple) * CanvasMultiple);

    /// <summary>Canvas a reference clip is resized onto: <see cref="AdaptCanvas"/>, unless that would enlarge the clip
    /// — then each axis rounds to its own size instead, so a small reference is never upscaled into a bigger canvas.</summary>
    public static (int Width, int Height) RefVideoCanvas(int width, int height)
    {
        (int canvasWidth, int canvasHeight) = AdaptCanvas(width, height);
        if ((long)width * height < (long)canvasWidth * canvasHeight)
        {
            return (Round(width), Round(height));
        }
        return (canvasWidth, canvasHeight);
    }

    /// <summary>Largest <c>17k + 5</c> frame count not exceeding <paramref name="frames"/>. A reference clip snaps
    /// <b>down</b> onto the grid, unlike a generation's length, which <see cref="AlignFrameCount"/> snaps up.</summary>
    public static int SnapFrameCountDown(int frames)
    {
        if (frames < 5)
        {
            throw new ArgumentOutOfRangeException(nameof(frames), frames,
                "MiniMax-H3 reference videos need at least 5 frames (~0.2 s at 24 fps).");
        }
        int n = frames;
        while (n % 17 != 5)
        {
            n--;
        }
        return n;
    }

    /// <summary>Frame indices the vision tower sees, sampled from the resized clip at 2 fps.</summary>
    public static IReadOnlyList<int> RefVideoSampleIndices(int frameCount)
    {
        if (frameCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frameCount), frameCount, "Must be positive.");
        }
        List<int> indices = new List<int>();
        for (int i = 0; i < frameCount; i += Fps / 2)
        {
            indices.Add(i);
        }
        return indices;
    }
}
