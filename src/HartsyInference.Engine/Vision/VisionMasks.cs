using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Vision;

/// <summary>Mask buffer helpers: bounding-box rasterization (the fallback when SAM 2 isn't installed) and packing an
/// L8 mask into the contract's RGB24 <see cref="ImageData"/>.</summary>
public static class VisionMasks
{
    /// <summary>Rasterizes an axis-aligned pixel-space box into an <c>W*H</c> L8 buffer (255 inside).</summary>
    public static byte[] RasterizeBox(float boxX1, float boxY1, float boxX2, float boxY2, int width, int height)
    {
        byte[] bytes = new byte[(long)width * height];
        int x1 = Math.Clamp((int)MathF.Floor(boxX1), 0, width);
        int x2 = Math.Clamp((int)MathF.Ceiling(boxX2), 0, width);
        int y1 = Math.Clamp((int)MathF.Floor(boxY1), 0, height);
        int y2 = Math.Clamp((int)MathF.Ceiling(boxY2), 0, height);
        for (int y = y1; y < y2; y++)
        {
            int rowOff = y * width;
            for (int x = x1; x < x2; x++)
            {
                bytes[rowOff + x] = 255;
            }
        }
        return bytes;
    }

    /// <summary>Wraps an L8 mask as an <see cref="ImageData"/> by replicating the single channel into R, G and B —
    /// the contract carries masks as RGB24, so a mask reads as a grayscale image in any consumer.</summary>
    public static ImageData ToImageData(byte[] gray, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(gray);
        byte[] rgb = new byte[(long)width * height * 3];
        for (int i = 0; i < gray.Length; i++)
        {
            byte v = gray[i];
            int o = i * 3;
            rgb[o] = v;
            rgb[o + 1] = v;
            rgb[o + 2] = v;
        }
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }
}
