using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Tests;

public sealed class HairDeckColorSpaceTests
{
    [Fact]
    public void LabToLinearRgb_MapsWhiteBlackAndMidGrey()
    {
        float[] lab = [100f, 0f, 0f, 0f, 0f, 0f, 50f, 0f, 0f];
        float[] rgb = new float[9];

        HairDeckColorSpace.LabToLinearRgb(lab, rgb);

        for (int k = 0; k < 3; k++) Assert.Equal(1f, rgb[k], 3);
        for (int k = 3; k < 6; k++) Assert.Equal(0f, rgb[k], 6);
        for (int k = 6; k < 9; k++) Assert.Equal(0.18419f, rgb[k], 3);
    }

    [Fact]
    public void LabToLinearRgb_ClipsOutOfGamutValues()
    {
        float[] lab = [50f, 120f, -120f];
        float[] rgb = new float[3];

        HairDeckColorSpace.LabToLinearRgb(lab, rgb);

        foreach (float v in rgb) Assert.InRange(v, 0f, 1f);
    }
}
