using HartsyInference.Vision.DepthPro2;

namespace HartsyInference.Vision.Tests.DepthPro2;

/// <summary>The size of an unbuffered stream's attention cache and when a device is too small for it.</summary>
public sealed class DepthPro2VideoCacheTests
{
    [Fact]
    public void TheCacheIsEveryCrossFrameBlocksWindowAndOneReadOfIt()
    {
        // ViT-H+ (11 cross-frame blocks, 1280 wide) at 1024x576, padded to itself, 2304 patches a frame.
        long rows = 5 + 16L * 2304, kv = rows * 1280 * 2 * sizeof(float);
        Assert.Equal(12 * kv + 11L * 16 * 5 * 1280 * sizeof(float),
            DepthPro2VideoStream.UnbufferedCacheBytes(DepthPro2Preset.HugePlus, 1024, 576, 16));

        // At the token cap the window alone is about 29.5 GB.
        long capped = DepthPro2VideoStream.UnbufferedCacheBytes(DepthPro2Preset.HugePlus, 2048, 2048, 16);
        Assert.InRange(capped / 1e9, 32.2, 32.3);
    }

    [Fact]
    public void ACacheThatDoesNotFitBesideTheWeightsIsRefusedBeforeAnyFrame()
    {
        const long weights = 3_500_000_000, card = 24_000_000_000;
        DepthPro2VideoStream.CheckCacheFits(DepthPro2Preset.HugePlus, 1024, 576, 16, weights, card);
        DepthPro2VideoStream.CheckCacheFits(DepthPro2Preset.HugePlus, 2048, 2048, 16, weights, deviceBytes: 0);

        ArgumentException refused = Assert.Throws<ArgumentException>(
            () => DepthPro2VideoStream.CheckCacheFits(DepthPro2Preset.HugePlus, 2048, 2048, 16, weights, card));

        Assert.Equal("window", refused.ParamName);
        Assert.Contains("GB", refused.Message);
        long exact = DepthPro2VideoStream.UnbufferedCacheBytes(DepthPro2Preset.HugePlus, 1024, 576, 16) + weights;
        DepthPro2VideoStream.CheckCacheFits(DepthPro2Preset.HugePlus, 1024, 576, 16, weights, exact);
        Assert.Throws<ArgumentException>(() => DepthPro2VideoStream.CheckCacheFits(DepthPro2Preset.HugePlus, 1024, 576, 16, weights, exact - 1));
    }
}
