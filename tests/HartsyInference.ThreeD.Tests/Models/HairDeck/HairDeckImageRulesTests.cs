using HartsyInference.ThreeD.Models.HairDeck;
using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

namespace HartsyInference.ThreeD.Tests.Models.HairDeck;

public sealed class HairDeckImageRulesTests
{
    [Fact]
    public void Load_TreatsAFullyOpaqueAlphaPlaneAsOpaque()
    {
        byte[] rgb = [1, 2, 3, 4, 5, 6];
        byte[] alpha = [255, 254];
        HairDeckSourceImage source = HairDeckSourceImage.Create(rgb, alpha, 2, 1);

        HairDeckLoadedImage loaded = HairDeckImageRules.Load(source);

        Assert.Null(loaded.InputAlpha);
        Assert.Same(rgb, loaded.ModelRgb);
    }

    [Fact]
    public void IsTransparent_NeedsMoreThanOneTenthOfAPercentBelowAlpha250()
    {
        byte[] tenThousand = new byte[10000];
        Array.Fill(tenThousand, (byte)255);

        for (int i = 0; i < 10; i++)
        {
            tenThousand[i] = 0;
        }

        // exactly 0.1% is not above the threshold
        Assert.False(HairDeckImageRules.IsTransparent(tenThousand));

        tenThousand[10] = 249;
        Assert.True(HairDeckImageRules.IsTransparent(tenThousand));
    }

    [Fact]
    public void Load_CompositesATransparentImageOnBlackForTheModels()
    {
        // one fully transparent pixel and one half-transparent pixel make the image transparent (2 of 2 pixels)
        byte[] rgb = [200, 100, 50, 200, 100, 50];
        byte[] alpha = [0, 128];
        HairDeckSourceImage source = HairDeckSourceImage.Create(rgb, alpha, 2, 1);

        HairDeckLoadedImage loaded = HairDeckImageRules.Load(source);

        Assert.Equal(alpha, loaded.InputAlpha);
        Assert.Equal([0, 0, 0], loaded.ModelRgb[..3]);
        // (200 * 128 + 127) / 255 = 100
        Assert.Equal([100, 50, 25], loaded.ModelRgb[3..6]);
        Assert.Equal(rgb, loaded.StoredRgb);
    }

    [Fact]
    public void Load_DownscalesASideAboveFourThousandNinetySixPixels()
    {
        byte[] rgb = new byte[5000 * 100 * 3];
        HairDeckSourceImage source = HairDeckSourceImage.Create(rgb, null, 5000, 100);

        HairDeckLoadedImage loaded = HairDeckImageRules.Load(source);

        Assert.Equal(4096, loaded.Width);
        Assert.Equal(82, loaded.Height);
        Assert.Equal(4096 * 82 * 3, loaded.ModelRgb.Length);
    }

    [Fact]
    public void Create_RejectsBuffersThatDoNotMatchTheSize()
    {
        HairDeckInputException rgbError = Assert.Throws<HairDeckInputException>(
            () => HairDeckSourceImage.Create(new byte[5], null, 2, 1));
        HairDeckInputException alphaError = Assert.Throws<HairDeckInputException>(
            () => HairDeckSourceImage.Create(new byte[6], new byte[3], 2, 1));
        HairDeckInputException sizeError = Assert.Throws<HairDeckInputException>(
            () => HairDeckSourceImage.Create([], null, 0, 1));

        Assert.Equal(HairDeckInputFailure.InvalidImage, rgbError.Failure);
        Assert.Equal(HairDeckInputFailure.InvalidImage, alphaError.Failure);
        Assert.Equal(HairDeckInputFailure.InvalidImage, sizeError.Failure);
    }

    [Fact]
    public void HairDeckInput_CreateChecksThePlaneLengths()
    {
        int side = HairDeckInput.Resolution;
        byte[] rgb = new byte[3 * side * side];
        byte[] matte = new byte[side * side];

        double[] color = [0.4, 0.3, 0.25];

        HairDeckInput input = HairDeckInput.Create(rgb, matte, color);

        Assert.Same(rgb, input.Rgb448);
        Assert.Equal(color, input.HairBaseColorSrgb);
        Assert.Throws<ArgumentException>(() => HairDeckInput.Create(new byte[10], matte, color));
        Assert.Throws<ArgumentException>(() => HairDeckInput.Create(rgb, new byte[10], color));
        Assert.Throws<ArgumentException>(() => HairDeckInput.Create(rgb, matte, [1.5, 0.0, 0.0]));
    }
}
