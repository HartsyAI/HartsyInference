using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using Xunit;
using static HartsyInference.Tests.Common.CpuSchedules;

namespace HartsyInference.Video.Tests;

/// <summary><see cref="VideoRgbFrames.ExtractAllFrames"/> fans the frames out through <see cref="CpuParallel"/>. The
/// bytes match extracting each frame on its own, whether the frames ran over every core, under a
/// <c>numerics.cpuThreads</c> cap of 1, or inside an <see cref="CpuParallel.InlineScope"/>.</summary>
[Collection(CpuThreadsCollection.Name)]
public sealed class VideoRgbFramesScheduleTests
{
    [Theory]
    [InlineData(3)]
    public void ExtractAllFrames_MatchesFrameByFrame_UnderEverySchedule(int channels)
    {
        const int frames = 12, height = 64, width = 96;
        using Tensor rgb = new(new TensorShape([1L, channels, frames, height, width]), DType.F32);
        Random rng = new(channels);
        Span<float> values = rgb.AsSpan<float>();
        // A little past [-1, 1], so the clamp to [0, 255] is exercised at both ends.
        for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() * 2.4 - 1.2);
        byte[][] expected = new byte[frames][];
        for (int f = 0; f < frames; f++) expected[f] = VideoRgbFrames.ExtractFrame(rgb, f);

        (byte[][] parallel, byte[][] capped, byte[][] inline) = UnderEverySchedule(() => VideoRgbFrames.ExtractAllFrames(rgb));

        for (int f = 0; f < frames; f++)
        {
            Assert.Equal(expected[f], parallel[f]);
            Assert.Equal(expected[f], capped[f]);
            Assert.Equal(expected[f], inline[f]);
        }
    }
}
