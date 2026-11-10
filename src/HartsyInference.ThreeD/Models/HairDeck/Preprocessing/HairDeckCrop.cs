using HartsyInference.Core.Imaging;
using HartsyInference.Vision.Codec;

namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>Crop geometry and compositing of the HairDeck reference (hairdeck.data.crop, hairdeck.data.illus.crop_white and
/// hairdeck.infer.image.Pipeline.prep): the head box, the person matte on white, the 448 px model crop, and the matte crop.</summary>
internal static class HairDeckCrop
{
    /// <summary>Margin of the head box around the hair and face union, as a fraction of its side.</summary>
    public const double BoxMargin = 0.08;

    /// <summary>Bounding box (x0, y0, x1, y1), exclusive, of a mask, or null when empty.</summary>
    public static (double X0, double Y0, double X1, double Y1)? BoundingBox(bool[] mask, int width, int height)
    {
        int minX = int.MaxValue, minY = int.MaxValue, maxX = -1, maxY = -1;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (!mask[y * width + x])
                {
                    continue;
                }

                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
            }
        }

        return maxX < 0 ? null : (minX, minY, maxX + 1.0, maxY + 1.0);
    }

    /// <summary>The square head box of hairdeck.data.crop.crop_box for the union of the hair and face boxes (inference rule), with
    /// its corners rounded to 0.1 px as the reference stores them.</summary>
    public static HairDeckBox HeadBox((double X0, double Y0, double X1, double Y1) union)
    {
        double cx = (union.X0 + union.X1) / 2, cy = (union.Y0 + union.Y1) / 2;
        double side = Math.Max(union.X1 - union.X0, union.Y1 - union.Y0) * (1 + 2 * BoxMargin);
        return new HairDeckBox(
            Round1(cx - side / 2), Round1(cy - side / 2), Round1(cx + side / 2), Round1(cy + side / 2));
    }

    /// <summary>The union of two optional boxes; null when both are null.</summary>
    public static (double X0, double Y0, double X1, double Y1)? Union(
        (double X0, double Y0, double X1, double Y1)? a, (double X0, double Y0, double X1, double Y1)? b)
    {
        if (a is null)
        {
            return b;
        }

        if (b is null)
        {
            return a;
        }

        return (Math.Min(a.Value.X0, b.Value.X0), Math.Min(a.Value.Y0, b.Value.Y0),
            Math.Max(a.Value.X1, b.Value.X1), Math.Max(a.Value.Y1, b.Value.Y1));
    }

    /// <summary>The RGB image with the person matte composited on white (PIL alpha_composite over an opaque white base).</summary>
    public static byte[] ComposeOnWhite(byte[] rgb, byte[] matte, int pixelCount)
    {
        byte[] result = new byte[pixelCount * 3];
        for (int i = 0; i < pixelCount; i++)
        {
            int a = matte[i];
            for (int c = 0; c < 3; c++)
            {
                result[i * 3 + c] = (byte)((rgb[i * 3 + c] * a + 255 * (255 - a) + 127) / 255);
            }
        }

        return result;
    }

    /// <summary>Crops an RGB image to <paramref name="box"/> with white outside the image and resizes it to <paramref name="side"/>²,
    /// as hairdeck.data.illus.crop_white does (bicubic, antialiased, 8-bit, the premultiplied outside treated as transparent).</summary>
    /// <returns>Row-major RGB, <c>side * side * 3</c> bytes.</returns>
    public static byte[] CropOnWhite(byte[] rgb, int width, int height, HairDeckBox box, int side)
    {
        (int x0, int y0, int canvasWidth, int canvasHeight) = CanvasOf(box);
        byte[] canvas = new byte[(long)canvasWidth * canvasHeight * 4];
        for (int y = 0; y < height; y++)
        {
            int cy = y - y0;
            if (cy < 0 || cy >= canvasHeight)
            {
                continue;
            }

            for (int x = 0; x < width; x++)
            {
                int cx = x - x0;
                if (cx < 0 || cx >= canvasWidth)
                {
                    continue;
                }

                int dst = (cy * canvasWidth + cx) * 4;
                int src = (y * width + x) * 3;
                canvas[dst] = rgb[src];
                canvas[dst + 1] = rgb[src + 1];
                canvas[dst + 2] = rgb[src + 2];
                canvas[dst + 3] = 255;
            }
        }

        float[] resized = new float[(long)side * side * 4];
        Resample.BicubicHwc8(canvas, canvasWidth, canvasHeight, 4, resized, side, side, a: -0.5f, antialias: true, quantizeU8: true);
        byte[] result = new byte[(long)side * side * 3];
        for (int i = 0; i < side * side; i++)
        {
            float coverage = resized[i * 4 + 3];
            for (int c = 0; c < 3; c++)
            {
                result[i * 3 + c] = (byte)Math.Clamp(MathF.Round(resized[i * 4 + c] + 255f - coverage), 0f, 255f);
            }
        }

        return result;
    }

    /// <summary>The person matte cut to the box with nearest sampling, thresholded at 127 (hairdeck matte_crop); all true when it is empty.</summary>
    /// <returns>Row-major <c>side * side</c> bytes, 0 or 255.</returns>
    public static byte[] CropMatte(byte[] matte, int width, int height, HairDeckBox box, int side)
    {
        (int x0, int y0, int canvasWidth, int canvasHeight) = CanvasOf(box);
        byte[] canvas = new byte[(long)canvasWidth * canvasHeight];
        for (int y = 0; y < height; y++)
        {
            int cy = y - y0;
            if (cy < 0 || cy >= canvasHeight)
            {
                continue;
            }

            for (int x = 0; x < width; x++)
            {
                int cx = x - x0;
                if (cx >= 0 && cx < canvasWidth)
                {
                    canvas[cy * canvasWidth + cx] = matte[y * width + x];
                }
            }
        }

        byte[] result = new byte[side * side];
        bool any = false;
        for (int y = 0; y < side; y++)
        {
            int sy = (int)Math.Floor((y + 0.5) * canvasHeight / side);
            for (int x = 0; x < side; x++)
            {
                int sx = (int)Math.Floor((x + 0.5) * canvasWidth / side);
                bool inside = canvas[sy * canvasWidth + sx] > 127;
                result[y * side + x] = inside ? (byte)255 : (byte)0;
                any |= inside;
            }
        }

        if (!any)
        {
            Array.Fill(result, (byte)255);
        }

        return result;
    }

    /// <summary>Mean sRGB colour of the darker half of the hair pixels, in [0, 1] (hairdeck.infer.glb.hair_color); a fixed brown when there is no hair.</summary>
    public static double[] HairColor(byte[] rgb, bool[] hair, int pixelCount)
    {
        List<int> pixels = [];
        for (int i = 0; i < pixelCount; i++)
        {
            if (hair[i])
            {
                pixels.Add(i);
            }
        }

        if (pixels.Count == 0)
        {
            return [0.4, 0.3, 0.25];
        }

        double[] luminance = new double[pixels.Count];
        for (int k = 0; k < pixels.Count; k++)
        {
            int p = pixels[k];
            luminance[k] = rgb[p * 3] * 0.299 + rgb[p * 3 + 1] * 0.587 + rgb[p * 3 + 2] * 0.114;
        }

        double lumMedian = Median(luminance);
        List<int> dark = [];
        for (int k = 0; k < pixels.Count; k++)
        {
            if (luminance[k] <= lumMedian)
            {
                dark.Add(pixels[k]);
            }
        }

        double[] result = new double[3];
        double[] channel = new double[dark.Count];
        for (int c = 0; c < 3; c++)
        {
            for (int k = 0; k < dark.Count; k++)
            {
                channel[k] = rgb[dark[k] * 3 + c];
            }

            result[c] = Median(channel) / 255.0;
        }

        return result;
    }

    private static (int X0, int Y0, int Width, int Height) CanvasOf(HairDeckBox box)
    {
        int x0 = (int)Math.Round(box.X0, MidpointRounding.ToEven);
        int y0 = (int)Math.Round(box.Y0, MidpointRounding.ToEven);
        int x1 = (int)Math.Round(box.X1, MidpointRounding.ToEven);
        int y1 = (int)Math.Round(box.Y1, MidpointRounding.ToEven);
        int width = x1 - x0, height = y1 - y0;
        if (width <= 0 || height <= 0)
        {
            throw new HairDeckInputException(HairDeckInputFailure.InvalidImage, $"Head box {box} is empty after rounding.");
        }

        return (x0, y0, width, height);
    }

    /// <summary>numpy.median: the middle value, or the mean of the two middle values for an even count.</summary>
    private static double Median(double[] values)
    {
        double[] sorted = [.. values];
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static double Round1(double value) => Math.Round(value, 1, MidpointRounding.ToEven);
}
