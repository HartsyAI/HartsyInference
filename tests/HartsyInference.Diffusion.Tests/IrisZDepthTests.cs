using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Iris-Z depth math against values worked out from the reference's formulas (speridlabs/iris-z-sdk
/// <c>iris_z/depth.py</c>, <c>iris_z/rgbd.py</c> and <c>model/sampling.py</c>).</summary>
public sealed class IrisZDepthTests
{
    [Fact]
    public void EncodeSpansTheSecondToNinetyEighthPercentile()
    {
        float[] logDepth = new float[101];
        for (int i = 0; i < logDepth.Length; i++) logDepth[i] = i;

        float[] encoded = IrisZDepth.EncodeLogDepth(logDepth, [], skyCeiling: 1f);

        // Percentiles 2 and 98 of 0..100 are exactly 2 and 98.
        Assert.Equal(-1f, encoded[0]);
        Assert.Equal(-1f, encoded[2], 5);
        Assert.Equal(0f, encoded[50], 5);
        Assert.Equal(1f, encoded[98], 5);
        Assert.Equal(1f, encoded[100]);
    }

    [Fact]
    public void SkyCeilingSqueezesRealDepthAndKeepsSkyAtOne()
    {
        float[] logDepth = new float[101];
        bool[] sky = new bool[101];
        for (int i = 0; i < logDepth.Length; i++) logDepth[i] = i;
        sky[100] = true;

        float[] encoded = IrisZDepth.EncodeLogDepth(logDepth, sky, skyCeiling: 0.95f);

        Assert.Equal(1f, encoded[100]);
        Assert.Equal(-1f, encoded[0]);
        Assert.True(encoded[99] <= 0.95f + 1e-6f);
    }

    [Fact]
    public void UnsqueezeInvertsTheCeilingAndMarksSky()
    {
        float[] encoded = [-1f, -0.025f, 0.95f, 0.99f];

        IrisZDepth.UnsqueezeFromCeiling(encoded, skyCeiling: 0.95f);

        Assert.Equal(-1f, encoded[0], 5);
        Assert.Equal(0f, encoded[1], 5);
        Assert.Equal(1f, encoded[2], 5);
        Assert.Equal(1f, encoded[3]);
        Assert.Equal(new[] { false, false, true, true }, IrisZDepth.SkyMask(encoded, 0.95f));
        Assert.Equal(new[] { false, false, false, false }, IrisZDepth.SkyMask(encoded, 1f));
    }

    [Fact]
    public void UnsqueezeWithoutACeilingOnlyClamps()
    {
        float[] encoded = [-1.5f, 0.25f, 1.5f];

        IrisZDepth.UnsqueezeFromCeiling(encoded, skyCeiling: 1f);

        Assert.Equal(new[] { -1f, 0.25f, 1f }, encoded);
    }

    [Fact]
    public void AlignAffineRecoversScaleAndShift()
    {
        float[] reference = [0.1f, -0.4f, 0.7f, 0.2f, -0.9f];
        float[] member = new float[reference.Length];
        for (int i = 0; i < member.Length; i++) member[i] = (reference[i] - 0.3f) / 2f;

        float[] aligned = IrisZDepth.AlignAffine(member, reference);

        for (int i = 0; i < reference.Length; i++) Assert.Equal(reference[i], aligned[i], 4);
    }

    [Fact]
    public void EnsembleTakesTheMedianOfAlignedMembersAndKeepsMajoritySky()
    {
        float[] a = [0f, 0.2f, 0.4f, 1f];
        float[] b = [0f, 0.2f, 0.4f, 1f];
        float[] c = [0f, 0.2f, 0.4f, 0.5f];

        float[] combined = IrisZDepth.Ensemble([a, b, c], referenceIndex: 0);

        Assert.Equal(0f, combined[0], 4);
        Assert.Equal(0.2f, combined[1], 4);
        Assert.Equal(1f, combined[3]);
    }

    [Fact]
    public void EnsembleOfOneIsThatMember()
    {
        float[] only = [0.1f, 0.2f];

        Assert.Equal(only, IrisZDepth.Ensemble([only], 0));
    }

    [Theory]
    [InlineData(1024, 1024, 1024, 1024, 1024)]
    [InlineData(2048, 2048, 1024, 1024, 1024)]
    [InlineData(1920, 1080, 1024, 1280, 720)]
    [InlineData(640, 480, 512, 576, 432)]
    public void ChooseResolutionKeepsAspectOnSixteenPixelSides(int sourceWidth, int sourceHeight, int target, int width, int height)
    {
        Assert.Equal((width, height), IrisZDepth.ChooseResolution(sourceWidth, sourceHeight, target));
    }

    [Fact]
    public void ChooseResolutionNeverEnlargesButEnlargedSizeDoes()
    {
        (int width, int height) = IrisZDepth.ChooseResolution(320, 240, 1024);
        Assert.True(width <= 320 && height <= 240);

        (int enlargedWidth, int enlargedHeight) = IrisZDepth.EnlargedSize(320, 240, 1024);
        Assert.Equal(0, enlargedWidth % 16);
        Assert.Equal(0, enlargedHeight % 16);
        Assert.InRange((long)enlargedWidth * enlargedHeight, (long)(1024 * 1024 * 0.8), (long)(1024 * 1024 * 1.2));
    }

    [Fact]
    public void ColourizeRunsNearRedToFarBlueWithBlackSky()
    {
        float[] logDepth = new float[200];
        bool[] sky = new bool[200];
        for (int i = 0; i < logDepth.Length; i++) logDepth[i] = i / 199f;
        sky[199] = true;

        byte[] rgb = IrisZDepth.Colourize(logDepth, sky);

        Assert.True(rgb[0] > rgb[2], "the nearest pixel is red");
        Assert.True(rgb[198 * 3 + 2] > rgb[198 * 3], "the farthest pixel is blue");
        Assert.Equal(new byte[] { 0, 0, 0 }, rgb[(199 * 3)..]);
    }

    [Fact]
    public void GrayDepthAndDisparityAreInverted()
    {
        float[] logDepth = [-1f, 0f, 1f];

        byte[] depth = IrisZDepth.ToGray(logDepth, disparity: false);
        byte[] disparity = IrisZDepth.ToGray(logDepth, disparity: true);

        Assert.Equal(0, depth[0]);
        Assert.Equal(255, depth[6]);
        Assert.Equal(255, disparity[0]);
        Assert.Equal(0, disparity[6]);
    }

    [Fact]
    public void BilinearResizeIsExactOnALinearRamp()
    {
        float[] ramp = [0f, 1f, 2f, 3f];

        float[] same = IrisZDepth.ResizeBilinear(ramp, 4, 1, 4, 1);
        float[] half = IrisZDepth.ResizeBilinear(ramp, 4, 1, 2, 1);

        Assert.Equal(ramp, same);
        Assert.Equal(0.5f, half[0], 5);
        Assert.Equal(2.5f, half[1], 5);
    }

    [Theory]
    // mu = a2·seq + b2 above 4300 tokens.
    [InlineData(4301, 50, 0.00016927 * 4301 + 0.45666666)]
    // Below: the line through (10 steps, m10) and (200 steps, m200), evaluated at the step count.
    [InlineData(1024, 10, 8.73809524e-05 * 1024 + 1.89833333)]
    [InlineData(1024, 200, 0.00016927 * 1024 + 0.45666666)]
    public void EmpiricalMuMatchesTheReferenceLine(int tokens, int steps, double expected)
    {
        Assert.Equal(expected, FlowMatchEulerDiscreteScheduler.EmpiricalMu(tokens, steps), 4);
    }
}
