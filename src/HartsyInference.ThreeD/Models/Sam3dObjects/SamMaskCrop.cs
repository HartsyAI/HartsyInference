namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// Crop geometry of the released preprocessor (<c>crop_around_mask_with_padding</c>, factor 1.2, padding 0, then
/// <c>pad_to_square_centered</c>). Images are <c>C × H × W</c> row-major floats. Resizing to 518 and the colour handling are separate steps.
/// </summary>
public static class SamMaskCrop
{
    /// <summary>
    /// The square box around the non-zero pixels of <paramref name="mask"/>: its centre, side <c>int(max(w, h, 2) · factor)</c>, as
    /// <c>x1, y1, x2, y2</c> with <c>x2 − x1 = 2 · (side / 2)</c>. May reach outside the image.
    /// </summary>
    public static (int X1, int Y1, int X2, int Y2) ComputeBox(ReadOnlySpan<float> mask, int height, int width, float boxSizeFactor)
    {
        if (height <= 0 || width <= 0 || mask.Length != height * width)
            throw new ArgumentException($"A mask of {height}x{width} holds {height * width} values; got {mask.Length}.", nameof(mask));
        int minX = int.MaxValue, minY = int.MaxValue, maxX = int.MinValue, maxY = int.MinValue;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (mask[y * width + x] == 0f)
                    continue;
                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
            }
        }

        if (maxX < minX)
            throw new InvalidOperationException("The mask is empty.");
        double centerX = (minX + maxX) / 2.0, centerY = (minY + maxY) / 2.0;
        int boxW = maxX - minX, boxH = maxY - minY;
        if (boxW < 2 || boxH < 2)
            throw new InvalidOperationException("Bounding box dimensions must be at least 2x2.");
        int side = (int)(Math.Max(Math.Max(boxW, boxH), 2) * boxSizeFactor);
        int half = side / 2;
        return ((int)(centerX - half), (int)(centerY - half), (int)(centerX + half), (int)(centerY + half));
    }

    /// <summary>Crops <c>[x1, x2) × [y1, y2)</c>; cells outside the image take <paramref name="fill"/> (torchvision pads a tensor crop with 0).</summary>
    public static float[] Crop(ReadOnlySpan<float> image, int channels, int height, int width, (int X1, int Y1, int X2, int Y2) box, float fill = 0f)
    {
        RequireImage(image, channels, height, width);
        int outH = box.Y2 - box.Y1, outW = box.X2 - box.X1;
        if (outH <= 0 || outW <= 0)
            throw new ArgumentException("The crop box is empty.", nameof(box));
        float[] result = new float[channels * outH * outW];
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < outH; y++)
            {
                int sy = box.Y1 + y;
                for (int x = 0; x < outW; x++)
                {
                    int sx = box.X1 + x;
                    bool inside = sy >= 0 && sy < height && sx >= 0 && sx < width;
                    result[(c * outH + y) * outW + x] = inside ? image[(c * height + sy) * width + sx] : fill;
                }
            }
        }

        return result;
    }

    /// <summary>Pads the shorter side so the image is square, the extra cell on the far side when the difference is odd.</summary>
    public static (float[] Image, int Size) PadToSquareCentered(ReadOnlySpan<float> image, int channels, int height, int width, float fill = 0f)
    {
        RequireImage(image, channels, height, width);
        int size = Math.Max(height, width);
        int padH = (size - height) / 2, padW = (size - width) / 2;
        return (Pad(image, channels, height, width, padH, size - height - padH, padW, size - width - padW, fill), size);
    }

    /// <summary>Pads each side by <paramref name="amount"/> cells, as the <c>padding_factor</c> step does (<c>int(size · factor)</c> per side).</summary>
    public static float[] PadEvenly(ReadOnlySpan<float> image, int channels, int height, int width, int amount, float fill = 0f)
    {
        RequireImage(image, channels, height, width);
        if (amount < 0)
            throw new ArgumentOutOfRangeException(nameof(amount), amount, "The padding must not be negative.");
        return Pad(image, channels, height, width, amount, amount, amount, amount, fill);
    }

    private static float[] Pad(
        ReadOnlySpan<float> image, int channels, int height, int width, int top, int bottom, int left, int right, float fill)
    {
        int outH = height + top + bottom, outW = width + left + right;
        float[] result = new float[channels * outH * outW];
        Array.Fill(result, fill);
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
                image.Slice((c * height + y) * width, width).CopyTo(result.AsSpan((c * outH + y + top) * outW + left, width));
        }

        return result;
    }

    private static void RequireImage(ReadOnlySpan<float> image, int channels, int height, int width)
    {
        if (channels <= 0 || height <= 0 || width <= 0 || image.Length != channels * height * width)
            throw new ArgumentException($"A {channels}x{height}x{width} image holds {channels * height * width} values; got {image.Length}.", nameof(image));
    }
}
