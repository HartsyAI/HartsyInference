using HartsyInference.Core.Imaging;

namespace HartsyInference.ThreeD.Preprocessing;

/// <summary>TRELLIS (v1)'s input preparation (<c>preprocess_image</c> in <c>trellis/pipelines/trellis_image_to_3d.py</c>): a square crop around the foreground, 1.2 times its larger side, resized to the conditioner's input with a Lanczos filter and premultiplied by alpha, so the background is black. The matting that upstream does first (rembg) is the caller's.</summary>
public static class TrellisImageProcessor
{
    /// <summary>The conditioner's input edge in pixels.</summary>
    public const int DefaultSize = 518;

    /// <summary>The coverage a pixel needs to count towards the crop box: 0.8 of full.</summary>
    private const int BoxThreshold = 204;

    /// <summary>Prepares one image. A null <paramref name="alpha"/> makes the whole image opaque.</summary>
    /// <exception cref="ArgumentException">No pixel has coverage above 0.8.</exception>
    public static PreparedImage Process(byte[] rgb, byte[]? alpha, int width, int height, int size = DefaultSize)
    {
        byte[] coverage = ForegroundPixels.Validate(rgb, alpha, width, height);
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), size, "The size must be positive.");
        (int MinX, int MinY, int MaxX, int MaxY)? box = ForegroundPixels.Bounds(coverage, width, height, BoxThreshold);
        if (box is null)
            throw new ArgumentException("No pixel of the image is more than 80 % foreground; supply a cutout or let the engine matte it.", nameof(alpha));

        double centerX = (box.Value.MinX + box.Value.MaxX) / 2.0;
        double centerY = (box.Value.MinY + box.Value.MaxY) / 2.0;
        int side = (int)(Math.Max(box.Value.MaxX - box.Value.MinX, box.Value.MaxY - box.Value.MinY) * 1.2);
        // PIL rounds the float crop box half to even; Math.Round does the same.
        int x0 = Round(centerX - side / 2);
        int y0 = Round(centerY - side / 2);
        int x1 = Round(centerX + side / 2);
        int y1 = Round(centerY + side / 2);
        if (x1 <= x0 || y1 <= y0)
            throw new ArgumentException("The foreground is too small to crop.", nameof(alpha));

        byte[] crop = ForegroundPixels.CropRgba(ForegroundPixels.ToRgba(rgb, coverage), width, height, x0, y0, x1, y1);
        int cropWidth = x1 - x0;
        int cropHeight = y1 - y0;
        // PIL filters RGBA premultiplied (RGBa): multiply, resize, divide back, as Image.resize does for this mode.
        Premultiply(crop);
        byte[] resized = new byte[size * size * 4];
        Resample.LanczosPil8(crop, cropWidth, cropHeight, 4, resized, size, size);
        Unpremultiply(resized);

        byte[] output = new byte[size * size * 3];
        for (int i = 0; i < size * size; i++)
        {
            float a = resized[i * 4 + 3] / 255f;
            for (int c = 0; c < 3; c++)
                output[i * 3 + c] = (byte)Math.Clamp((int)(resized[i * 4 + c] / 255f * a * 255f), 0, 255);
        }
        return new PreparedImage(output, size, size);
    }

    private static int Round(double value) => (int)Math.Round(value, MidpointRounding.ToEven);

    private static void Premultiply(byte[] rgba)
    {
        for (int i = 0; i < rgba.Length; i += 4)
        {
            for (int c = 0; c < 3; c++)
            {
                int t = rgba[i + c] * rgba[i + 3] + 128;
                rgba[i + c] = (byte)((t + (t >> 8)) >> 8);
            }
        }
    }

    private static void Unpremultiply(byte[] rgba)
    {
        for (int i = 0; i < rgba.Length; i += 4)
        {
            int a = rgba[i + 3];
            for (int c = 0; c < 3; c++)
                rgba[i + c] = a == 0 ? (byte)0 : (byte)Math.Min(255 * rgba[i + c] / a, 255);
        }
    }
}
