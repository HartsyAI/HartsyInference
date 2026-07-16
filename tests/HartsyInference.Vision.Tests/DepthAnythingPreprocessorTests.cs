using HartsyInference.Vision.DepthAnything;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Unit-tier checks for the Depth-Anything-V2 preprocessing math (no weights, no reference dumps):
/// lower-bound-518 / multiple-of-14 target sizing and min-max normalization.</summary>
public sealed class DepthAnythingPreprocessorTests
{
    [Theory]
    [InlineData(518, 518, 518, 518)]   // native — untouched
    [InlineData(640, 480, 686, 518)]   // landscape: short side hits 518, long side rounds to 686
    [InlineData(480, 640, 518, 686)]   // portrait mirror
    [InlineData(1920, 1080, 924, 518)] // 16:9 → 921.6 rounds to 924
    [InlineData(100, 100, 518, 518)]   // upscale
    [InlineData(451, 300, 784, 518)]   // the parity fixture (see dump_depth_anything.py)
    public void ComputeTargetSize_LowerBoundMultipleOf14(int srcW, int srcH, int expectedW, int expectedH)
    {
        DepthAnythingPreprocessor pre = new();
        (int w, int h) = pre.ComputeTargetSize(srcW, srcH);
        Assert.Equal((expectedW, expectedH), (w, h));
        Assert.Equal(0, w % 14);
        Assert.Equal(0, h % 14);
        Assert.True(w >= 518 && h >= 518);
    }

    [Fact]
    public void NormalizeToUnit_MapsMinMaxTo01()
    {
        float[] depth = [2f, 4f, 6f, 10f];
        DepthAnythingPreprocessor.NormalizeToUnit(depth);
        Assert.Equal(0f, depth[0]);
        Assert.Equal(0.25f, depth[1]);
        Assert.Equal(0.5f, depth[2]);
        Assert.Equal(1f, depth[3]);
    }

    [Fact]
    public void NormalizeToUnit_ConstantMapBecomesZeros()
    {
        float[] depth = [3f, 3f, 3f];
        DepthAnythingPreprocessor.NormalizeToUnit(depth);
        Assert.All(depth, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void ToGrayscaleRgb24_RendersUnitDepth()
    {
        byte[] rgb = DepthAnythingPreprocessor.ToGrayscaleRgb24([0f, 0.5f, 1f]);
        Assert.Equal([0, 0, 0, 128, 128, 128, 255, 255, 255], rgb);
    }
}
