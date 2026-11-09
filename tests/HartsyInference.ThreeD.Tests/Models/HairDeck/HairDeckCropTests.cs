using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

namespace HartsyInference.ThreeD.Tests.Models.HairDeck;

public sealed class HairDeckCropTests
{
    [Fact]
    public void HeadBox_AddsEightPercentMarginAroundASquareOfTheUnion()
    {
        // union 100 x 100 at (100, 50): centre (150, 100), side 100 * 1.16
        HairDeckBox box = HairDeckCrop.HeadBox((100, 50, 200, 150));

        Assert.Equal(new HairDeckBox(92, 42, 208, 158), box);
    }

    [Fact]
    public void HeadBox_RoundsCornersToTenthOfAPixel()
    {
        HairDeckBox box = HairDeckCrop.HeadBox((0, 0, 3, 1));

        // side 3 * 1.16 = 3.48, centre (1.5, 0.5): x 1.5 -/+ 1.74 and y 0.5 -/+ 1.74, rounded to 0.1
        Assert.Equal(new HairDeckBox(-0.2, -1.2, 3.2, 2.2), box);
    }

    [Fact]
    public void BoundingBox_IsExclusiveAndNullWhenEmpty()
    {
        bool[] mask = new bool[4 * 3];
        mask[1 * 4 + 2] = true;
        mask[2 * 4 + 3] = true;

        Assert.Equal((2.0, 1.0, 4.0, 3.0), HairDeckCrop.BoundingBox(mask, 4, 3));
        Assert.Null(HairDeckCrop.BoundingBox(new bool[12], 4, 3));
    }

    [Fact]
    public void ComposeOnWhite_BlendsTheMatteOverWhite()
    {
        byte[] rgb = [10, 20, 30, 10, 20, 30, 10, 20, 30];
        byte[] matte = [0, 255, 128];

        byte[] composed = HairDeckCrop.ComposeOnWhite(rgb, matte, 3);

        Assert.Equal([255, 255, 255, 10, 20, 30], composed[..6]);
        // 10 * 128 + 255 * 127 = 1280 + 32385 = 33665; / 255 = 132.02 -> 132
        Assert.Equal(132, composed[6]);
    }

    [Fact]
    public void CropOnWhite_KeepsAUniformImageUniformInside()
    {
        byte[] rgb = new byte[16 * 16 * 3];
        Array.Fill(rgb, (byte)90);
        HairDeckBox box = new(0, 0, 16, 16);

        byte[] crop = HairDeckCrop.CropOnWhite(rgb, 16, 16, box, 8);

        Assert.All(crop, value => Assert.Equal(90, value));
    }

    [Fact]
    public void CropOnWhite_PadsTheOutsideOfTheImageWithWhite()
    {
        byte[] rgb = new byte[4 * 4 * 3];
        Array.Fill(rgb, (byte)40);
        HairDeckBox box = new(-4, -4, 8, 8);

        // scale 1: the image occupies crop pixels 4..7; the corners lie outside it
        byte[] crop = HairDeckCrop.CropOnWhite(rgb, 4, 4, box, 12);

        int last = (11 * 12 + 11) * 3;
        Assert.Equal([255, 255, 255], crop[..3]);
        Assert.Equal([255, 255, 255], crop[last..(last + 3)]);
        Assert.Equal(40, crop[(5 * 12 + 5) * 3]);
    }

    [Fact]
    public void CropMatte_IsAllTrueWhenEmpty()
    {
        byte[] matte = new byte[4 * 4];

        byte[] cut = HairDeckCrop.CropMatte(matte, 4, 4, new HairDeckBox(0, 0, 4, 4), 4);

        Assert.All(cut, value => Assert.Equal(255, value));
    }

    [Fact]
    public void CropMatte_SamplesNearestAndThresholdsAtOneTwentySeven()
    {
        byte[] matte = new byte[4 * 4];
        matte[0] = 127;
        matte[1] = 128;

        byte[] cut = HairDeckCrop.CropMatte(matte, 4, 4, new HairDeckBox(0, 0, 4, 4), 4);

        Assert.Equal(0, cut[0]);
        Assert.Equal(255, cut[1]);
    }

    [Fact]
    public void HairColor_IsTheFixedBrownWithoutHair()
    {
        double[] color = HairDeckCrop.HairColor(new byte[12], new bool[4], 4);

        Assert.Equal([0.4, 0.3, 0.25], color);
    }

    [Fact]
    public void HairColor_UsesTheDarkerHalfWithNumpyMedians()
    {
        // four hair pixels: luminance 10, 20, 30, 200; the darker half is the first two
        byte[] rgb = [10, 10, 10, 20, 20, 20, 30, 30, 30, 200, 200, 200];
        bool[] hair = [true, true, true, true];

        double[] color = HairDeckCrop.HairColor(rgb, hair, 4);

        // median of 10 and 20 is 15
        Assert.Equal(15.0 / 255.0, color[0], precision: 12);
    }
}
