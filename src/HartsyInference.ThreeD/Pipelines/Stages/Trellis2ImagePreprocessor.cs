using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Imaging;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// Upstream <c>Trellis2ImageTo3DPipeline.preprocess_image</c>: shrink to at most 1024 px on the long edge, take the foreground matte
/// (the image's own alpha when it has any, otherwise a matter), crop the square around the pixels with alpha above 0.8, and
/// premultiply RGB by alpha. The result is the RGB square the conditioner resizes.
/// </summary>
public static class Trellis2ImagePreprocessor
{
    /// <summary>Longest edge kept before matting.</summary>
    public const int MaxInputEdge = 1024;

    /// <summary>Alpha a pixel needs to count as foreground when the crop is fitted: <c>alpha &gt; 0.8 · 255</c>.</summary>
    public const int ForegroundThreshold = 204;

    /// <summary>Prepares <paramref name="image"/>; <paramref name="matter"/> is needed only when the image has no usable alpha.</summary>
    /// <exception cref="HartsyInferenceException">The image has no alpha and no matter is available, or no foreground pixel is found.</exception>
    public static Trellis2PreparedImage Prepare(IBackend backend, Trellis2InputImage image, IForegroundMatter? matter)
        => Prepare(backend, image, matter, 1.0);

    /// <summary>As <see cref="Prepare(IBackend, Trellis2InputImage, IForegroundMatter?)"/> with the crop square widened by
    /// <paramref name="cropScale"/> (upstream's <c>int(size * scale)</c>): 1.0 for TRELLIS.2, 1.1 for Pixal3D.</summary>
    public static Trellis2PreparedImage Prepare(IBackend backend, Trellis2InputImage image, IForegroundMatter? matter, double cropScale)
        => PrepareWithCutout(backend, image, matter, cropScale).Crop;

    /// <summary>As <see cref="Prepare(IBackend, Trellis2InputImage, IForegroundMatter?, double)"/>, also returning the cutout the crop was taken from.</summary>
    public static Trellis2PreparedCutout PrepareWithCutout(IBackend backend, Trellis2InputImage image, IForegroundMatter? matter, double cropScale)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(image);
        if (!(cropScale >= 1.0))
            throw new ArgumentOutOfRangeException(nameof(cropScale), cropScale, "The crop scale must be at least 1.");
        image.Validate();
        bool hasAlpha = image.HasForegroundAlpha;
        if (!hasAlpha && matter is not { IsAvailable: true })
        {
            throw new HartsyInferenceException(
                "TRELLIS.2 needs the foreground cut out: supply an image with an alpha channel, or place a BiRefNet matting "
                + "checkpoint (Models/Vision/BiRefNet-matting/model.safetensors) so the engine can matte it.");
        }

        int width = image.Width, height = image.Height;
        byte[] rgb = image.Rgb;
        byte[]? alpha = hasAlpha ? image.Alpha : null;
        int longEdge = Math.Max(width, height);
        if (longEdge > MaxInputEdge)
        {
            double scale = (double)MaxInputEdge / longEdge;
            int newWidth = Math.Max(1, (int)(width * scale)), newHeight = Math.Max(1, (int)(height * scale));
            (rgb, alpha) = Shrink(rgb, alpha, width, height, newWidth, newHeight);
            width = newWidth;
            height = newHeight;
        }

        bool matted = alpha is null;
        if (alpha is null)
            alpha = ToBytes(matter!.Alpha(backend, rgb, width, height), width * height);
        return new Trellis2PreparedCutout(Crop(rgb, alpha, width, height, cropScale), rgb, alpha, width, height, matted);
    }

    /// <summary>Crops the square around the foreground and premultiplies RGB by alpha, as upstream does after matting.</summary>
    internal static Trellis2PreparedImage Crop(byte[] rgb, byte[] alpha, int width, int height, double cropScale = 1.0)
    {
        int minX = int.MaxValue, minY = int.MaxValue, maxX = -1, maxY = -1;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (alpha[y * width + x] <= ForegroundThreshold)
                    continue;
                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
            }
        }

        if (maxX < 0)
            throw new HartsyInferenceException("No foreground pixel has alpha above 0.8; the image has nothing to reconstruct.");

        // PIL rounds a fractional crop box half to even, and pads outside the image with zeros.
        double centerX = (minX + maxX) / 2.0, centerY = (minY + maxY) / 2.0;
        int size = (int)(Math.Max(maxX - minX, maxY - minY) * cropScale);
        int left = (int)Math.Round(centerX - size / 2), top = (int)Math.Round(centerY - size / 2);
        int right = (int)Math.Round(centerX + size / 2), bottom = (int)Math.Round(centerY + size / 2);
        int cropW = right - left, cropH = bottom - top;
        if (cropW <= 0 || cropH <= 0)
            throw new HartsyInferenceException("The foreground is a single pixel wide; the image has nothing to reconstruct.");

        byte[] output = new byte[cropW * cropH * 3];
        for (int y = 0; y < cropH; y++)
        {
            int sy = top + y;
            for (int x = 0; x < cropW; x++)
            {
                int sx = left + x;
                if (sx < 0 || sx >= width || sy < 0 || sy >= height)
                    continue;
                int src = sy * width + sx;
                float a = alpha[src] / 255f;
                for (int c = 0; c < 3; c++)
                    output[(y * cropW + x) * 3 + c] = (byte)(rgb[src * 3 + c] / 255f * a * 255f);
            }
        }

        return new Trellis2PreparedImage(output, cropW, cropH);
    }

    // PIL resizes RGBA through premultiplied RGBa, so a translucent edge does not bleed its hidden colour.
    private static (byte[] Rgb, byte[]? Alpha) Shrink(byte[] rgb, byte[]? alpha, int width, int height, int newWidth, int newHeight)
    {
        int pixels = width * height, newPixels = newWidth * newHeight;
        int channels = alpha is null ? 3 : 4;
        byte[] source = new byte[pixels * channels];
        for (int i = 0; i < pixels; i++)
        {
            if (alpha is null)
            {
                source[i * 3] = rgb[i * 3];
                source[i * 3 + 1] = rgb[i * 3 + 1];
                source[i * 3 + 2] = rgb[i * 3 + 2];
                continue;
            }

            for (int c = 0; c < 3; c++)
                source[i * 4 + c] = (byte)((rgb[i * 3 + c] * alpha[i] + 127) / 255);
            source[i * 4 + 3] = alpha[i];
        }

        float[] resized = new float[newPixels * channels];
        Resample.LanczosHwc8(source, width, height, channels, resized, newWidth, newHeight);
        byte[] newRgb = new byte[newPixels * 3];
        byte[]? newAlpha = alpha is null ? null : new byte[newPixels];
        for (int i = 0; i < newPixels; i++)
        {
            int a = alpha is null ? 255 : Round8(resized[i * 4 + 3]);
            if (newAlpha is not null)
                newAlpha[i] = (byte)a;
            for (int c = 0; c < 3; c++)
            {
                int v = Round8(resized[i * channels + c]);
                newRgb[i * 3 + c] = (byte)(alpha is null ? v : a == 0 ? 0 : Math.Min(255, (v * 255 + a / 2) / a));
            }
        }

        return (newRgb, newAlpha);
    }

    /// <summary>Quantizes a matte to 8 bits by truncation, as <c>ToPILImage</c> does for a float tensor.</summary>
    private static byte[] ToBytes(float[] matte, int expected)
    {
        if (matte.Length != expected)
            throw new HartsyInferenceException($"The matter returned {matte.Length} values for {expected} pixels.");
        byte[] bytes = new byte[expected];
        for (int i = 0; i < expected; i++)
            bytes[i] = (byte)Math.Clamp((int)(matte[i] * 255f), 0, 255);
        return bytes;
    }

    /// <summary>Rounds a resampled value to the nearest 8-bit level, half up, as PIL's resampler does.</summary>
    internal static int Round8(float value) => Math.Clamp((int)(value + 0.5f), 0, 255);
}
