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
        (double w, double h) = ratio >= 1.0
            ? (BaseShortEdge * ratio, (double)BaseShortEdge)
            : ((double)BaseShortEdge, BaseShortEdge / ratio);
        if (w * h > MaxPixels)
        {
            double s = Math.Sqrt(MaxPixels / (w * h));
            w *= s;
            h *= s;
        }
        return (Round(w), Round(h));
    }

    /// <summary>Rounds a pixel axis onto <see cref="CanvasMultiple"/> without letting it collapse to zero.</summary>
    public static int Round(double pixels) =>
        Math.Max(CanvasMultiple, (int)Math.Round(pixels / CanvasMultiple) * CanvasMultiple);
}
