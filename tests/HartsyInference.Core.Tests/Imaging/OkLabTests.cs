using HartsyInference.Core.Imaging;
using Xunit;

namespace HartsyInference.Core.Tests.Imaging;

/// <summary>OKLab and its planar helpers. The primaries' values are the ones in Ottosson's published table, written in
/// from it; nothing here was compared with another implementation run on this machine.</summary>
public sealed class OkLabTests
{
    private const float Tolerance = 2e-4f;

    [Theory]
    [InlineData(1f, 0f, 0f, 0.62796f, 0.22486f, 0.12585f)]
    [InlineData(0f, 1f, 0f, 0.86644f, -0.23389f, 0.17950f)]
    [InlineData(0f, 0f, 1f, 0.45201f, -0.03246f, -0.31153f)]
    [InlineData(1f, 1f, 1f, 1f, 0f, 0f)]
    [InlineData(0f, 0f, 0f, 0f, 0f, 0f)]
    public void LinearPrimariesMatchThePublishedValues(float r, float g, float b, float l, float a, float bb)
    {
        (float outL, float outA, float outB) = OkLab.FromLinearSrgb(r, g, b);

        Assert.InRange(outL, l - Tolerance, l + Tolerance);
        Assert.InRange(outA, a - Tolerance, a + Tolerance);
        Assert.InRange(outB, bb - Tolerance, bb + Tolerance);
    }

    [Fact]
    public void TransferFunctionRoundTripsAndKeepsItsFixedPoints()
    {
        Assert.Equal(0f, Srgb.ToLinear(0f));
        Assert.Equal(1f, Srgb.ToLinear(1f), 6);
        Assert.Equal(1f, Srgb.FromLinear(1f), 6);
        Assert.Equal(0.21404f, Srgb.ToLinear(0.5f), 4);
        for (int step = -20; step <= 120; step++)
        {
            float value = step / 100f;
            Assert.Equal(value, Srgb.FromLinear(Srgb.ToLinear(value)), 5);
        }
        Assert.True(float.IsFinite(Srgb.ToLinear(-0.3f)) && Srgb.ToLinear(-0.3f) < 0f);
    }

    [Fact]
    public void LinearSrgbRoundTripsThroughOkLab()
    {
        Random rng = new(3);
        for (int i = 0; i < 2000; i++)
        {
            float r = (float)rng.NextDouble(), g = (float)rng.NextDouble(), b = (float)rng.NextDouble();
            (float l, float a, float bb) = OkLab.FromLinearSrgb(r, g, b);
            (float outR, float outG, float outB) = OkLab.ToLinearSrgb(l, a, bb);

            Assert.InRange(outR, r - 2e-5f, r + 2e-5f);
            Assert.InRange(outG, g - 2e-5f, g + 2e-5f);
            Assert.InRange(outB, b - 2e-5f, b + 2e-5f);
        }
    }

    [Fact]
    public void PlanarConversionMatchesThePerPixelFunctionsAndRoundTrips()
    {
        const int pixels = 257;
        Random rng = new(4);
        float[] srgb = new float[3 * pixels];
        for (int i = 0; i < srgb.Length; i++) srgb[i] = (float)rng.NextDouble();
        float[] okLab = new float[3 * pixels];
        float[] back = new float[3 * pixels];

        OkLab.FromSrgbPlanar(srgb, okLab, pixels);
        OkLab.ToSrgbPlanar(okLab, back, pixels);

        for (int i = 0; i < pixels; i++)
        {
            (float l, float a, float b) = OkLab.FromLinearSrgb(Srgb.ToLinear(srgb[i]),
                Srgb.ToLinear(srgb[pixels + i]), Srgb.ToLinear(srgb[2 * pixels + i]));
            Assert.Equal(l, okLab[i]);
            Assert.Equal(a, okLab[pixels + i]);
            Assert.Equal(b, okLab[2 * pixels + i]);
        }
        for (int i = 0; i < srgb.Length; i++) Assert.InRange(back[i], srgb[i] - 5e-5f, srgb[i] + 5e-5f);
    }

    [Fact]
    public void PlanarConversionCanRunInPlaceOnOneBuffer()
    {
        const int pixels = 100;
        Random rng = new(8);
        float[] original = new float[3 * pixels];
        for (int i = 0; i < original.Length; i++) original[i] = (float)rng.NextDouble();
        float[] separate = new float[3 * pixels];
        OkLab.FromSrgbPlanar(original, separate, pixels);
        float[] inPlace = (float[])original.Clone();

        OkLab.FromSrgbPlanar(inPlace, inPlace, pixels);

        Assert.True(separate.AsSpan().SequenceEqual(inPlace));
        float[] backSeparate = new float[3 * pixels];
        OkLab.ToSrgbPlanar(separate, backSeparate, pixels);
        OkLab.ToSrgbPlanar(inPlace, inPlace, pixels);
        Assert.True(backSeparate.AsSpan().SequenceEqual(inPlace));
    }

    [Fact]
    public void PlanarConversionRejectsAWrongLength()
    {
        Assert.Throws<ArgumentException>(() => OkLab.FromSrgbPlanar(new float[6], new float[9], 3));
        Assert.Throws<ArgumentException>(() => OkLab.ToSrgbPlanar(new float[9], new float[6], 3));
    }

    [Fact]
    public void ClipMovesAnOutOfGamutColourIntoGamutAndLeavesAnInGamutOneAlone()
    {
        // (0.5, 0.3, 0.3) is far outside sRGB: its linear red is above 1 and its green below 0.
        float[] outside = [0.5f, 0.3f, 0.3f];
        (float rawR, float rawG, _) = OkLab.ToLinearSrgb(0.5f, 0.3f, 0.3f);
        Assert.True(rawR > 1f || rawG < 0f, "the test colour must start outside the gamut");
        float[] inside = [0.6f, 0.05f, -0.04f];
        (float insideR, float insideG, float insideB) = OkLab.ToLinearSrgb(0.6f, 0.05f, -0.04f);
        foreach (float component in new[] { insideR, insideG, insideB }) Assert.InRange(component, 0f, 1f);
        float[] insideBefore = (float[])inside.Clone();

        OkLab.ClipToSrgbGamut(outside, 1);
        OkLab.ClipToSrgbGamut(inside, 1);

        (float r, float g, float b) = OkLab.ToLinearSrgb(outside[0], outside[1], outside[2]);
        foreach (float component in new[] { r, g, b }) Assert.InRange(component, -1e-5f, 1f + 1e-5f);
        for (int i = 0; i < 3; i++) Assert.InRange(inside[i], insideBefore[i] - 1e-5f, insideBefore[i] + 1e-5f);
    }

    [Fact]
    public void ToSrgbClipsOnlyWhenAsked()
    {
        float[] okLab = [0.5f, 0.3f, 0.3f];
        float[] clipped = new float[3];
        float[] raw = new float[3];

        OkLab.ToSrgbPlanar(okLab, clipped, 1, clip: true);
        OkLab.ToSrgbPlanar(okLab, raw, 1, clip: false);

        foreach (float value in clipped) Assert.InRange(value, 0f, 1f);
        Assert.Contains(raw, value => value < 0f || value > 1f);
    }

    [Fact]
    public void NormalizationRoundTripsAndNeedsExplicitPositiveValues()
    {
        OkLabNormalization normalization = new([0.5f, 0.01f, -0.02f], [0.2f, 0.05f, 0.08f]);
        float[] planar = [0.3f, 0.9f, 0.01f, 0.2f, -0.1f, 0.0f];
        float[] original = (float[])planar.Clone();

        normalization.Normalize(planar, 2);
        Assert.Equal((0.3f - 0.5f) / 0.2f, planar[0], 6);
        Assert.Equal((0.2f - 0.01f) / 0.05f, planar[3], 5);
        Assert.Equal((-0.1f - -0.02f) / 0.08f, planar[4], 5);
        normalization.Denormalize(planar, 2);

        for (int i = 0; i < planar.Length; i++) Assert.Equal(original[i], planar[i], 5);
        Assert.Throws<ArgumentException>(() => new OkLabNormalization([0f, 0f, 0f], [1f, 0f, 1f]));
        Assert.Throws<ArgumentException>(() => new OkLabNormalization([0f, 0f], [1f, 1f]));
        Assert.Throws<InvalidOperationException>(() => default(OkLabNormalization).Normalize(new float[3], 1));
        Assert.Throws<ArgumentException>(() => new OkLabNormalization([0f, 0f, 0f], [1f, float.PositiveInfinity, 1f]));
        // Equal numbers make equal values.
        OkLabNormalization same = new([0.5f, 0.01f, -0.02f], [0.2f, 0.05f, 0.08f]);
        Assert.Equal(normalization, same);
        Assert.Equal(normalization.GetHashCode(), same.GetHashCode());
    }
}
