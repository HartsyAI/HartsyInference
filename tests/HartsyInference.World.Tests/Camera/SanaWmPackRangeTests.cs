using HartsyInference.Core.Tensors;
using HartsyInference.World.Camera;
using Xunit;

namespace HartsyInference.World.Tests.Camera;

/// <summary><see cref="SanaWmCameraConditioning.PackRange"/>, the camera inputs of one stream chunk: for every chunk of the
/// upstream plan, packing just that chunk from a path that ends at the chunk's last sampled raw frame gives exactly the slice of
/// the whole-clip <see cref="SanaWmCameraConditioning.Pack"/>, so a live stream needs no pose beyond the chunk it renders.</summary>
public sealed class SanaWmPackRangeTests
{
    private const int LatentHeight = 4, LatentWidth = 6;

    [Theory]
    [InlineData(0, 4)]
    [InlineData(4, 3)]
    [InlineData(7, 3)]
    [InlineData(3, 1)]
    public void ChunkPack_EqualsTheWholeClipSlice(int firstLatent, int latentFrames)
    {
        const int rawFrames = 73;
        List<float[]> poses = Path(rawFrames);
        PinholeIntrinsics[] intrinsics = Intrinsics(rawFrames);
        (float[] wholeConditions, Tensor wholePlucker) = SanaWmCameraConditioning.Pack(poses, intrinsics, LatentHeight, LatentWidth);
        using Tensor whole = wholePlucker;

        int lastRaw = (firstLatent + latentFrames - 1) * SanaWmCameraConditioning.TemporalStride;
        List<float[]> truncated = poses.GetRange(0, lastRaw + 1);
        (float[] conditions, Tensor chunkPlucker) = SanaWmCameraConditioning.PackRange(truncated, intrinsics, LatentHeight, LatentWidth,
            firstLatent, latentFrames);
        using Tensor chunk = chunkPlucker;

        int width = SanaWmCameraConditioning.ConditionWidth;
        Assert.Equal(wholeConditions.AsSpan(firstLatent * width, latentFrames * width).ToArray(), conditions);
        Assert.Equal(new TensorShape([1, SanaWmCameraConditioning.PluckerChannels, latentFrames, LatentHeight, LatentWidth]), chunk.Shape);
        int plane = LatentHeight * LatentWidth, totalLatent = (int)whole.Shape[2];
        ReadOnlySpan<float> w = whole.AsReadOnlySpan<float>(), c = chunk.AsReadOnlySpan<float>();
        for (int channel = 0; channel < SanaWmCameraConditioning.PluckerChannels; channel++)
        {
            ReadOnlySpan<float> expected = w.Slice((channel * totalLatent + firstLatent) * plane, latentFrames * plane);
            ReadOnlySpan<float> actual = c.Slice(channel * latentFrames * plane, latentFrames * plane);
            Assert.True(expected.SequenceEqual(actual), $"Plücker channel {channel} differs");
        }
    }

    [Fact]
    public void ChunkPack_RefusesAPathShortOfTheChunk()
    {
        List<float[]> poses = Path(16);
        Assert.Throws<ArgumentException>(() =>
            SanaWmCameraConditioning.PackRange(poses, Intrinsics(1), LatentHeight, LatentWidth, firstLatentFrame: 1, latentFrames: 2));
    }

    // A path that moves and turns every frame, so a wrong raw frame anywhere changes the numbers.
    private static List<float[]> Path(int frames)
    {
        List<float[]> poses = new(frames);
        for (int f = 0; f < frames; f++)
        {
            float yaw = 0.01f * f, c = MathF.Cos(yaw), s = MathF.Sin(yaw);
            poses.Add([c, 0, s, 0.01f * f, 0, 1, 0, -0.002f * f, -s, 0, c, 0.025f * f, 0, 0, 0, 1]);
        }

        return poses;
    }

    private static PinholeIntrinsics[] Intrinsics(int frames)
    {
        PinholeIntrinsics[] intrinsics = new PinholeIntrinsics[frames];
        for (int f = 0; f < frames; f++)
            intrinsics[f] = new PinholeIntrinsics(5f + 0.01f * f, 5f, 3f, 2f);
        return intrinsics;
    }
}
