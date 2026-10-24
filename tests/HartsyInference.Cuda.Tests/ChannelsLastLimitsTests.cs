using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Size selection for the channels-last layout copies. CPU-only: the predicate touches no driver state.</summary>
public sealed class ChannelsLastLimitsTests
{
    // LTX-2.5 video VAE decode at 1024x896: one output frame is 917,504 elements, so the output transpose's d1 is
    // outT * 917,504 and gridDim.y is ceil(d1 / 32).
    private const long Frame1024x896 = 1024L * 896;

    [Fact]
    public void Ltx1024x896_TwoOutputFrames_Fits()
    {
        long outSpatial = 2 * Frame1024x896;
        Assert.True(ChannelsLastLimits.ConvFits(xBatch: 4, c: 128, xSpatial: Frame1024x896, k: 128, kernelSpatial: 27,
            outBatch: 1, outSpatial: outSpatial));
    }

    [Fact]
    public void Ltx1024x896_ThreeOutputFrames_DoesNotFit()
    {
        // gridDim.y = ceil(2,752,512 / 32) = 86,016 > 65,535: cuLaunchKernel returns CUDA_ERROR_INVALID_VALUE.
        long outSpatial = 3 * Frame1024x896;
        Assert.False(ChannelsLastLimits.ConvFits(xBatch: 5, c: 128, xSpatial: Frame1024x896, k: 128, kernelSpatial: 27,
            outBatch: 1, outSpatial: outSpatial));
    }

    [Fact]
    public void OutputDimAtTheCap_Fits_OneMoreElementDoesNot()
    {
        long cap = ChannelsLastLimits.MaxGridDimYZ * ChannelsLastLimits.TileDim;   // 2,097,120
        Assert.True(ChannelsLastLimits.TransposeFits(batch: 1, d1: cap, d2: 128));
        Assert.False(ChannelsLastLimits.TransposeFits(batch: 1, d1: cap + 1, d2: 128));
    }

    [Fact]
    public void SmallerFrames768x512_FiveOutputFrames_Fits()
    {
        long outSpatial = 5 * 768L * 512;
        Assert.True(ChannelsLastLimits.ConvFits(xBatch: 6, c: 128, xSpatial: 768L * 512, k: 128, kernelSpatial: 27,
            outBatch: 1, outSpatial: outSpatial));
    }

    [Fact]
    public void Batch_AboveGridZ_DoesNotFit()
    {
        Assert.True(ChannelsLastLimits.TransposeFits(batch: 65_535, d1: 128, d2: 64));
        Assert.False(ChannelsLastLimits.TransposeFits(batch: 65_536, d1: 128, d2: 64));
        Assert.False(ChannelsLastLimits.TransposeFits(batch: 0, d1: 128, d2: 64));
    }

    [Fact]
    public void TwoDimensionalImage_AtDocumentedBoundary_RoutesByOutputPixels()
    {
        // 1448x1448 (2,096,704 px) still fits; 2048x1024 (2,097,152 px) does not and falls back to NCHW.
        Assert.True(ChannelsLastLimits.TransposeFits(batch: 1, d1: 1448L * 1448, d2: 320));
        Assert.False(ChannelsLastLimits.TransposeFits(batch: 1, d1: 2048L * 1024, d2: 320));
    }
}
