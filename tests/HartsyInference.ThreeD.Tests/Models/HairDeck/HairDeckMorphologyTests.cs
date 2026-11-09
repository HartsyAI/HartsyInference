using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

namespace HartsyInference.ThreeD.Tests.Models.HairDeck;

public sealed class HairDeckMorphologyTests
{
    [Theory]
    [InlineData(0.2, 1)]
    [InlineData(2.5, 2)]
    [InlineData(3.5, 4)]
    [InlineData(10.24, 10)]
    public void Radius_RoundsHalfToEvenAndIsAtLeastOne(double value, int expected)
    {
        Assert.Equal(expected, HairDeckMorphology.Radius(value));
    }

    [Fact]
    public void Dilate_GrowsASinglePixelToTheSquareAndClipsAtTheBorder()
    {
        bool[] mask = new bool[5 * 5];
        mask[0] = true;

        bool[] grown = HairDeckMorphology.Dilate(mask, 5, 5, 1);

        Assert.True(grown[0]);
        Assert.True(grown[1]);
        Assert.True(grown[5]);
        Assert.True(grown[6]);
        Assert.False(grown[2]);
        Assert.Equal(4, Count(grown));
    }

    [Fact]
    public void Label_UsesFourConnectivity()
    {
        // two pixels touching only at a corner are two components
        bool[] mask = [true, false, false, true];
        int[] labels = new int[4];

        int count = HairDeckMorphology.Label(mask, 2, 2, labels);

        Assert.Equal(2, count);
        Assert.NotEqual(labels[0], labels[3]);
    }

    [Fact]
    public void BoxMean_CountsOutsideTheImageAsZero()
    {
        bool[] mask = [true, true, true, true];

        double[] means = HairDeckMorphology.BoxMean(mask, 2, 2, 3);

        // each pixel's 3 x 3 window holds all four pixels at most: 4 / 9
        Assert.Equal(4.0 / 9.0, means[0], precision: 12);
    }

    [Fact]
    public void GaussianReflect_KeepsAConstantPlaneConstant()
    {
        float[] plane = new float[6 * 4];
        Array.Fill(plane, 0.75f);

        float[] blurred = HairDeckMorphology.GaussianReflect(plane, 4, 6, 0.8);

        Assert.All(blurred, value => Assert.Equal(0.75f, value, precision: 5));
    }

    [Fact]
    public void LogitsToMask_ThresholdsTheUpsampledLogitsAtZero()
    {
        // a 2 x 2 logit plane with the top row positive; upsampled to 4 x 4 (align_corners=False) the rows are
        // 5, 2.5, -2.5, -5 at the left edge, so the top two rows are foreground
        float[] logits = [5f, 5f, -5f, -5f];

        bool[] mask = HairDeckMorphology.LogitsToMask(logits, 2, 4, 4);

        Assert.True(mask[0]);
        Assert.True(mask[1 * 4]);
        Assert.False(mask[2 * 4]);
        Assert.False(mask[3 * 4 + 3]);
    }

    [Theory]
    [InlineData(1000, 1167, 1024, 877)]
    [InlineData(1500, 500, 341, 1024)]
    [InlineData(600, 400, 400, 600)]
    public void WorkingShape_FitsTheLongSideToTheWorkingResolution(int width, int height, int expectedHeight, int expectedWidth)
    {
        (int shapeHeight, int shapeWidth) = HairDeckPreprocessor.WorkingShape(width, height);

        Assert.Equal(expectedHeight, shapeHeight);
        Assert.Equal(expectedWidth, shapeWidth);
    }

    private static int Count(bool[] mask)
    {
        int count = 0;
        foreach (bool value in mask)
        {
            if (value)
            {
                count++;
            }
        }

        return count;
    }
}
