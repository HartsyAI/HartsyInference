namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>The condition inputs of both generators and the pointmap normalisation the pose decode needs.</summary>
/// <param name="Structure">Inputs of the structure stage (images, masks, normalised pointmaps).</param>
/// <param name="Slat">Inputs of the SLAT stage (images and masks only).</param>
/// <param name="PointmapScale">The scene scale, equal on all three axes; null without a pointmap.</param>
/// <param name="PointmapShift">The per-axis shift; null without a pointmap.</param>
public sealed record SamPrepared(SamConditionInputs Structure, SamConditionInputs Slat, float[]? PointmapScale, float[]? PointmapShift);

/// <summary>
/// The released preprocessing (<c>pipeline.yaml</c> <c>ss_preprocessor</c> and <c>slat_preprocessor</c>, <c>PreProcessor._process_image_mask_pointmap_mess</c>) on planar floats.
/// The pointmap is normalised first with the full mask (<c>ObjectCentricSSI</c>: shift of the masked points, one scene scale); the <c>cropped</c> branch crops image, mask
/// and normalised pointmap to the mask's bounding square scaled by 1.2 (cells outside the frame 0), pads to a square (pointmap NaN) and resizes; the <c>full</c> branch only pads
/// to a square and resizes (the pointmap with the same scale and shift). Images resize bilinearly with antialiasing, masks and pointmaps nearest. The RGB image is not multiplied by
/// the mask (upstream splits the RGBA input into an image and an alpha mask).
/// </summary>
public static class SamPreprocessor
{
    /// <summary><c>box_size_factor</c> of both crops.</summary>
    public const float BoxSizeFactor = 1.2f;

    /// <summary>
    /// Prepares one object. <paramref name="rgb"/> is planar <c>[3, height, width]</c> in 0 to 1, <paramref name="mask"/> <c>[height, width]</c> (the alpha channel in 0 to 1),
    /// <paramref name="pointmap"/> optional planar <c>[3, height, width]</c> in the PyTorch3D camera convention (NaN or infinity for invalid points; a different size is resized nearest to the image first, as upstream does).
    /// </summary>
    /// <param name="size">Output side; 518 for the released models (<see cref="SamDinoEmbedder.InputSize"/>).</param>
    /// <exception cref="InvalidOperationException">The mask is empty, or its bounding box smaller than 2 x 2.</exception>
    public static SamPrepared Prepare(
        ReadOnlySpan<float> rgb, ReadOnlySpan<float> mask, SamPointmapInput? pointmap, int height, int width, int size = SamDinoEmbedder.InputSize)
    {
        if (height <= 0 || width <= 0 || size <= 0)
            throw new ArgumentException("The image and output sizes must be positive.");
        if (rgb.Length != 3 * height * width || mask.Length != height * width)
            throw new ArgumentException($"Expected a 3 x {height} x {width} image and a {height} x {width} mask; got {rgb.Length} and {mask.Length} values.");

        float[]? normalized = null, scale = null, shift = null;
        if (pointmap is not null)
        {
            float[] aligned = pointmap.Height == height && pointmap.Width == width
                ? pointmap.Planar
                : SamImageResize.Nearest(pointmap.Planar, 3, pointmap.Height, pointmap.Width, height, width);
            (scale, shift) = SamPointmapNormalizer.ComputeScaleAndShift(aligned, mask, height, width);
            normalized = (float[])aligned.Clone();
            SamPointmapNormalizer.Normalize(normalized, height, width, scale, shift);
        }

        (int X1, int Y1, int X2, int Y2) box = SamMaskCrop.ComputeBox(mask, height, width, BoxSizeFactor);
        int side = box.X2 - box.X1;

        float[] croppedImage = Square(SamMaskCrop.Crop(rgb, 3, height, width, box), 3, box.Y2 - box.Y1, side, size, bilinear: true, fill: 0f);
        float[] croppedMask = Square(SamMaskCrop.Crop(mask, 1, height, width, box), 1, box.Y2 - box.Y1, side, size, bilinear: false, fill: 0f);
        float[] fullImage = Square(rgb, 3, height, width, size, bilinear: true, fill: 0f);
        float[] fullMask = Square(mask, 1, height, width, size, bilinear: false, fill: 0f);

        SamPointmapInput? croppedPoints = null, fullPoints = null;
        if (normalized is not null)
        {
            croppedPoints = new SamPointmapInput(
                Square(SamMaskCrop.Crop(normalized, 3, height, width, box, 0f), 3, box.Y2 - box.Y1, side, size, bilinear: false, fill: float.NaN), size, size);
            fullPoints = new SamPointmapInput(Square(normalized, 3, height, width, size, bilinear: false, fill: float.NaN), size, size);
        }

        return new SamPrepared(
            new SamConditionInputs(croppedImage, croppedMask, fullImage, fullMask, croppedPoints, fullPoints),
            new SamConditionInputs(croppedImage, croppedMask, fullImage, fullMask),
            scale,
            shift);
    }

    /// <summary>Pads to a square (centred, <paramref name="fill"/>) then resizes to <paramref name="size"/>.</summary>
    private static float[] Square(ReadOnlySpan<float> image, int channels, int height, int width, int size, bool bilinear, float fill)
    {
        (float[] padded, int side) = SamMaskCrop.PadToSquareCentered(image, channels, height, width, fill);
        return bilinear
            ? SamImageResize.BilinearAntialias(padded, channels, side, side, size, size)
            : SamImageResize.Nearest(padded, channels, side, side, size, size);
    }
}
