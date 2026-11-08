using HartsyInference.Vision.Codec;

namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>The image as the models see it: the stored colours, the colours the models read (composited on black when the
/// input is transparent), and the input alpha that limits the outputs (null when the image counts as opaque).</summary>
internal sealed record HairDeckLoadedImage(byte[] StoredRgb, byte[] ModelRgb, byte[]? InputAlpha, int Width, int Height);

/// <summary>Input rules of the HairDeck reference (hairdeck.prep.preprocess.load_image and hairdeck.infer.image.load_image):
/// a side above 4096 px is downscaled, and an image with at least 0.1% of its pixels below alpha 250 is transparent.</summary>
internal static class HairDeckImageRules
{
    /// <summary>Larger sides are downscaled first, as the reference does before any model sees the image.</summary>
    public const int MaxSide = 4096;

    private const int OpaqueThreshold = 250;
    private const double TransparentFraction = 0.001;

    /// <summary>Applies the size and alpha rules to a decoded image.</summary>
    public static HairDeckLoadedImage Load(HairDeckSourceImage source)
    {
        ArgumentNullException.ThrowIfNull(source);
        byte[] rgb = source.Rgb;
        byte[]? alpha = source.Alpha;
        int width = source.Width, height = source.Height;
        if (Math.Max(width, height) > MaxSide)
        {
            double scale = (double)MaxSide / Math.Max(width, height);
            int newWidth = (int)Math.Round(width * scale, MidpointRounding.ToEven);
            int newHeight = (int)Math.Round(height * scale, MidpointRounding.ToEven);
            rgb = Downscale(rgb, width, height, 3, newWidth, newHeight);
            alpha = alpha is null ? null : Downscale(alpha, width, height, 1, newWidth, newHeight);
            width = newWidth;
            height = newHeight;
        }

        if (alpha is null || !IsTransparent(alpha))
        {
            return new HairDeckLoadedImage(rgb, rgb, null, width, height);
        }

        byte[] composited = new byte[rgb.Length];
        for (int i = 0; i < alpha.Length; i++)
        {
            int a = alpha[i];
            for (int c = 0; c < 3; c++)
            {
                composited[i * 3 + c] = (byte)((rgb[i * 3 + c] * a + 127) / 255);
            }
        }

        return new HairDeckLoadedImage(rgb, composited, alpha, width, height);
    }

    /// <summary>True when more than 0.1% of the pixels are not (nearly) opaque.</summary>
    public static bool IsTransparent(byte[] alpha)
    {
        long below = 0;
        foreach (byte value in alpha)
        {
            if (value < OpaqueThreshold)
            {
                below++;
            }
        }

        return (double)below / alpha.Length > TransparentFraction;
    }

    private static byte[] Downscale(byte[] pixels, int width, int height, int channels, int newWidth, int newHeight)
    {
        float[] resized = new float[(long)newWidth * newHeight * channels];
        Resample.LanczosHwc8(pixels, width, height, channels, resized, newWidth, newHeight);
        byte[] result = new byte[resized.Length];
        for (int i = 0; i < resized.Length; i++)
        {
            result[i] = (byte)Math.Clamp(MathF.Round(resized[i]), 0f, 255f);
        }

        return result;
    }
}
