using Xunit;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Tests.Imaging;

/// <summary>The shared VLM imaging rules. Expected values come from <c>transformers</c>' <c>smart_resize</c>, numpy's
/// <c>linspace</c> and Python's <c>f"{t:.1f}"</c>, run once and pasted here.</summary>
public sealed class VisionImagingTests
{
    [Theory]
    [InlineData(720, 1280, 4096, 16777216, 704, 1280)]
    [InlineData(75, 100, 4096, 16777216, 64, 96)]
    [InlineData(30, 40, 4096, 16777216, 64, 96)]
    [InlineData(180, 320, 4096, 8192, 64, 96)]
    [InlineData(1080, 1920, 4096, 1000000, 736, 1312)]
    [InlineData(48, 48, 4096, 16777216, 64, 64)]
    [InlineData(16, 3000, 4096, 16777216, 32, 896)]
    public void Qwen2VlSmartResizeMatchesTransformers(int h, int w, long min, long max, int eh, int ew) =>
        Assert.Equal((eh, ew), SmartResize.Qwen2Vl(h, w, 32, min, max));

    [Fact]
    public void Qwen2VlSmartResizeRefusesAnExtremeAspect() =>
        Assert.Throws<ArgumentException>(() => SmartResize.Qwen2Vl(10, 2010, 32, 4096, 16777216));

    [Fact]
    public void ShareBudgetSplitsByAreaAndClamps()
    {
        long[] shares = SmartResize.ShareBudget([(100, 100), (100, 300), (1, 1)], totalBudget: 40_000, floorPixels: 4096, ceilingPixels: 25_000);
        // Areas 10000, 30000, 1 of 40001: raw shares 9999, 29999 (capped to 25000) and 0 (raised to the floor).
        Assert.Equal([9999L, 25_000, 4096], shares);
    }

    [Fact]
    public void PatchifyWritesMergeBlockOrder()
    {
        // 64x64 image, patch 16, merge 2: a 4x4 patch grid in 2x2 merge blocks. Each pixel holds row*1000 + col.
        VisionPatchSpec spec = new(PatchSize: 16, MergeSize: 2, TemporalPatchSize: 1);
        float[] plane = new float[3 * 64 * 64];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++)
                    plane[c * 4096 + y * 64 + x] = c * 1_000_000 + y * 1000 + x;

        using Tensor rows = VisionPatchPreprocessor.Patchify([plane], 64, 64, spec);
        ReadOnlySpan<float> r = rows.AsSpan<float>();

        Assert.Equal(new TensorShape(16, 768), rows.Shape);
        // Row 0: block (0,0) patch (0,0) — top-left pixel. Row 1: same block, patch (0,1) — starts at column 16.
        Assert.Equal(0f, r[0]);
        Assert.Equal(16f, r[768]);
        // Row 2: patch (1,0) of the first block — starts at row 16. Row 4: the second block's first patch — column 32.
        Assert.Equal(16_000f, r[2 * 768]);
        Assert.Equal(32f, r[4 * 768]);
        // Within a row: channel-major, then patch rows: element 256 is channel 1's first pixel.
        Assert.Equal(1_000_000f, r[256]);
        Assert.Equal(1_000f, r[16]);
    }

    [Fact]
    public void PreprocessorRescalesAndNormalizes()
    {
        VisionPatchPreprocessor pre = new(new VisionPatchSpec(16, 2, 1), [0.5f, 0.5f, 0.5f], [0.5f, 0.5f, 0.5f], ResampleOptions.PilBicubic);
        byte[] rgb = new byte[32 * 32 * 3];
        for (int i = 0; i < rgb.Length; i += 3) { rgb[i] = 255; rgb[i + 1] = 0; rgb[i + 2] = 51; }

        using VisionPatchGrid grid = pre.Preprocess(new RawRgbFrame(rgb, 32, 32), 32, 32);

        Assert.Equal((1, 2, 2, 1, 1), (grid.GridT, grid.GridH, grid.GridW, grid.MergedHeight, grid.MergedWidth));
        ReadOnlySpan<float> r = grid.PixelValues.AsSpan<float>();
        Assert.Equal(1f, r[0], 5);          // 255 → (1 - 0.5) / 0.5
        Assert.Equal(-1f, r[256], 5);       // 0 → -1
        Assert.Equal(-0.6f, r[512], 5);     // 51 → (0.2 - 0.5) / 0.5
    }

    [Fact]
    public void PreprocessorRefusesATargetOffTheMergeGrid()
    {
        VisionPatchPreprocessor pre = new(new VisionPatchSpec(16, 2, 1), [0.5f, 0.5f, 0.5f], [0.5f, 0.5f, 0.5f], ResampleOptions.PilBicubic);
        Assert.Throws<ArgumentException>(() => pre.Preprocess(new RawRgbFrame(new byte[48 * 48 * 3], 48, 48), 48, 48));
        Assert.Throws<ArgumentException>(() => pre.Preprocess(new RawRgbFrame(new byte[10], 48, 48), 32, 32));
    }

    [Theory]
    [InlineData(300, 1.0, 30.0, new[] { 0, 33, 66, 99, 132, 166, 199, 232, 265, 299 })]
    [InlineData(10, 1.0, 2.0, new[] { 0, 2, 4, 6, 9 })]
    [InlineData(7, 1.0, 30.0, new[] { 0 })]
    public void FramePlanMatchesNumpyLinspace(int total, double fps, double sourceFps, int[] expected)
    {
        SampledFrame[] plan = VideoFrameSampling.Plan(total, sourceFps, new VideoFrameSamplingOptions(Fps: fps));
        Assert.Equal(expected, plan.Select(f => f.Index).ToArray());
        Assert.All(plan, f => Assert.Equal(f.Index / sourceFps, f.TimestampSeconds, 12));
    }

    [Theory]
    [InlineData(3000, 2.0, 25.0, 240, 359_761)]
    [InlineData(100, 4.0, 10.0, 40, 1962)]
    public void LongFramePlansMatchNumpyLinspace(int total, double fps, double sourceFps, int count, long indexSum)
    {
        SampledFrame[] plan = VideoFrameSampling.Plan(total, sourceFps, new VideoFrameSamplingOptions(Fps: fps));
        Assert.Equal(count, plan.Length);
        Assert.Equal(indexSum, plan.Sum(f => (long)f.Index));
        Assert.Equal(total - 1, plan[^1].Index);
    }

    [Fact]
    public void FramePlanHonoursMaxFrames()
    {
        SampledFrame[] plan = VideoFrameSampling.Plan(36_000, 30.0, new VideoFrameSamplingOptions(Fps: 1.0, MaxFrames: 256));
        Assert.Equal(256, plan.Length);
        Assert.Equal(0, plan[0].Index);
        Assert.Equal(35_999, plan[^1].Index);
    }

    [Theory]
    [InlineData(10, 720, 1280, 1_000_000, 224, 416)]
    [InlineData(100_000, 720, 1280, 1_000_000, 64, 96)]
    [InlineData(5, 120, 160, 65_536_000, 128, 160)]
    [InlineData(256, 1080, 1920, 65_536_000, 352, 672)]
    [InlineData(3, 64, 48, 65_536_000, 64, 64)]
    public void FrameSizeMatchesTheVideoProcessor(int frames, int h, int w, long videoMaxPixels, int eh, int ew) =>
        // Expected values from MOSS-VL's video smart_resize (per-frame cap, clip volume, per-frame floor).
        Assert.Equal((eh, ew), VideoFrameSampling.FrameSize(h, w, frames, 32, new VideoFrameSamplingOptions(VideoMaxPixels: videoMaxPixels)));

    [Fact]
    public void FrameSizeRefusesFramesSmallerThanThePatchBlock() =>
        Assert.Throws<ArgumentException>(() => VideoFrameSampling.FrameSize(16, 64, 4, 32, new VideoFrameSamplingOptions()));

    [Theory]
    [InlineData(0.0, "0.0")]
    [InlineData(0.25, "0.2")]
    [InlineData(0.35, "0.3")]
    [InlineData(2.5, "2.5")]
    [InlineData(12.45, "12.4")]
    [InlineData(1.0 / 3.0, "0.3")]
    [InlineData(0.05, "0.1")]
    [InlineData(0.15, "0.1")]
    [InlineData(100.0, "100.0")]
    public void TimestampTextMatchesPythonFormatting(double seconds, string expected) =>
        Assert.Equal(expected, VideoTimestampText.Seconds1(seconds));
}
