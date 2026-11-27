using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamMaskCropTests
{
    private static float[] MaskWithBox(int h, int w, int x1, int y1, int x2, int y2)
    {
        float[] mask = new float[h * w];
        for (int y = y1; y <= y2; y++)
        {
            for (int x = x1; x <= x2; x++)
                mask[y * w + x] = 1f;
        }

        return mask;
    }

    [Fact]
    public void ComputeBox_IsASquareAroundTheMaskCentreScaledByTheFactor()
    {
        // Mask x 10..30, y 20..30: w 20, h 10, centre (20, 25); side int(20 · 1.2) = 24, half 12.
        (int x1, int y1, int x2, int y2) = SamMaskCrop.ComputeBox(MaskWithBox(64, 64, 10, 20, 30, 30), 64, 64, 1.2f);

        Assert.Equal((8, 13, 32, 37), (x1, y1, x2, y2));
        Assert.Equal(x2 - x1, y2 - y1);
    }

    [Fact]
    public void ComputeBox_OddSideIsOneCellShortOfTheSide()
    {
        // Mask 0..9 square: w = h = 9, centre 4.5, side int(9 · 1) = 9, half 4: box (0, 0, 8, 8) since int(4.5 − 4) = 0.
        (int x1, _, int x2, _) = SamMaskCrop.ComputeBox(MaskWithBox(16, 16, 0, 0, 9, 9), 16, 16, 1f);

        Assert.Equal(0, x1);
        Assert.Equal(8, x2);
    }

    [Fact]
    public void ComputeBox_RejectsEmptyAndTinyMasks()
    {
        Assert.Throws<InvalidOperationException>(() => SamMaskCrop.ComputeBox(new float[16], 4, 4, 1.2f));
        Assert.Throws<InvalidOperationException>(() => SamMaskCrop.ComputeBox(MaskWithBox(8, 8, 3, 3, 4, 4), 8, 8, 1.2f));
    }

    [Fact]
    public void Crop_PadsOutsideCellsWithTheFillValue()
    {
        float[] image = [1f, 2f, 3f, 4f];

        float[] crop = SamMaskCrop.Crop(image, 1, 2, 2, (-1, -1, 2, 2), fill: float.NaN);

        Assert.Equal(9, crop.Length);
        Assert.True(float.IsNaN(crop[0]));
        Assert.Equal([1f, 2f], crop.Skip(4).Take(2));
        Assert.Equal([3f, 4f], crop.Skip(7).Take(2));
    }

    [Fact]
    public void PadToSquareCentered_PutsTheExtraCellOnTheFarSide()
    {
        float[] image = [1f, 2f, 3f, 4f, 5f, 6f, 7f];

        (float[] padded, int size) = SamMaskCrop.PadToSquareCentered(image, 1, 1, 7);

        // Height 1 -> 7: 3 rows above, 3 below.
        Assert.Equal(7, size);
        Assert.Equal(49, padded.Length);
        Assert.Equal(image, padded.Skip(21).Take(7));
        Assert.Equal(0f, padded[0]);

        (float[] odd, int oddSize) = SamMaskCrop.PadToSquareCentered(new float[2], 1, 1, 2);
        Assert.Equal(2, oddSize);
        Assert.Equal(4, odd.Length);
    }

    [Fact]
    public void PadToSquareCentered_RoundPadEvenOddSplitsFloorFirst()
    {
        (float[] padded, int size) = SamMaskCrop.PadToSquareCentered([1f, 2f, 3f, 4f, 5f, 6f], 1, 2, 3);

        // 3 − 2 = 1: floor(1 / 2) = 0 rows above, 1 below.
        Assert.Equal(3, size);
        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f, 0f, 0f, 0f], padded);
    }

    [Fact]
    public void PadEvenly_AddsTheAmountOnEverySide()
    {
        float[] padded = SamMaskCrop.PadEvenly([5f], 1, 1, 1, 2, fill: -1f);

        Assert.Equal(25, padded.Length);
        Assert.Equal(5f, padded[12]);
        Assert.Equal(24, padded.Count(v => v == -1f));
    }
}
