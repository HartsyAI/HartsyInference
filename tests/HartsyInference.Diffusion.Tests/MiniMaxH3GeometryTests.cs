using Xunit;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MiniMax-H3's geometry grids, pinned against the reference nodes. Each of these was a live defect: frames
/// were derived as <c>frames / 4</c> (delivering 102 of a requested 121), pixel axes snapped to 16 (leaving an odd
/// latent axis that the 2x2 patchifier silently truncated), and audio was sized from the raw request rather than the
/// aligned frame count (generating audio past the end of the video, then trimming it away).</summary>
public class MiniMaxH3GeometryTests
{
    [Theory]
    [InlineData(1, 5)]
    [InlineData(5, 5)]
    [InlineData(6, 22)]
    [InlineData(22, 22)]
    [InlineData(49, 56)]
    [InlineData(121, 124)]
    [InlineData(124, 124)]
    public void AlignFrameCount_SnapsUpOntoThe17kPlus5Grid(int requested, int expected)
    {
        int aligned = MiniMaxH3Geometry.AlignFrameCount(requested);
        Assert.Equal(expected, aligned);
        Assert.Equal(5, aligned % 17);
        Assert.True(aligned >= Math.Max(5, requested));
    }

    [Theory]
    [InlineData(5, 2)]
    [InlineData(22, 7)]
    [InlineData(56, 17)]
    [InlineData(124, 37)]
    public void VideoLatentFrames_MatchesTheReference(int frames, int expected) =>
        Assert.Equal(expected, MiniMaxH3Geometry.VideoLatentFrames(frames));

    /// <summary>The decoder expands each latent token by the <c>{1,4,4,4,4}</c> cycle, so the latent count the
    /// geometry hands the pipeline must expand back to exactly the aligned frame count — otherwise the caller is
    /// quietly short-changed (121 requested delivered 102 before this was pinned).</summary>
    [Theory]
    [InlineData(5)]
    [InlineData(22)]
    [InlineData(56)]
    [InlineData(124)]
    [InlineData(362)]
    public void VideoLatentFrames_RoundTripsThroughTheFramePerTokenCycle(int frames)
    {
        int[] framePerToken = [1, 4, 4, 4, 4];
        int latentT = MiniMaxH3Geometry.VideoLatentFrames(frames);
        int decoded = 0;
        for (int k = 0; k < latentT; k++)
        {
            decoded += framePerToken[k % 5];
        }
        Assert.Equal(frames, decoded);
    }

    [Theory]
    [InlineData(56, 93)]
    [InlineData(124, 207)]
    public void AudioLatentFrames_FollowTheAlignedFrameCount(int frames, int expected) =>
        Assert.Equal(expected, MiniMaxH3Geometry.AudioLatentFrames(frames));

    /// <summary>Audio and video must cover the same wall-clock span; drift here is what silently trimmed ~0.4 s off
    /// every clip.</summary>
    [Theory]
    [InlineData(22)]
    [InlineData(56)]
    [InlineData(124)]
    public void AudioAndVideoDurationsAgree(int frames)
    {
        double video = (double)frames / MiniMaxH3Geometry.Fps;
        double audio = (double)MiniMaxH3Geometry.AudioLatentFrames(frames) / MiniMaxH3Geometry.AudioLatentFps;
        Assert.True(Math.Abs(video - audio) < 0.02, $"{frames}f: video {video:F4}s vs audio {audio:F4}s");
    }

    /// <summary>Every accepted pixel axis must give an EVEN latent axis — the DiT patch is 2x2 in latent space over a
    /// 16x VAE, so an odd axis loses its last patch row/column in <c>UnpackVideo</c> with no error.</summary>
    [Theory]
    [InlineData(1360, 768)]
    [InlineData(1920, 1080)]
    [InlineData(1024, 1024)]
    [InlineData(720, 1280)]
    [InlineData(1, 1)]
    public void AdaptCanvas_AlwaysYieldsAnEvenLatentGrid(int width, int height)
    {
        (int w, int h) = MiniMaxH3Geometry.AdaptCanvas(width, height);
        Assert.Equal(0, w % MiniMaxH3Geometry.CanvasMultiple);
        Assert.Equal(0, h % MiniMaxH3Geometry.CanvasMultiple);
        Assert.Equal(0, w / 16 % 2);
        Assert.Equal(0, h / 16 % 2);
        Assert.True((long)w * h <= MiniMaxH3Geometry.MaxPixels, $"{w}x{h} exceeds the area cap");
    }

    /// <summary>The 16:9 default is the recipe's declared canvas; 1360x768 (a multiple of 16 but not 32) is exactly
    /// the case that used to truncate to 1344 without saying so.</summary>
    [Fact]
    public void AdaptCanvas_SixteenByNine_IsTheDeclaredDefault()
    {
        Assert.Equal((1344, 768), MiniMaxH3Geometry.AdaptCanvas(1920, 1080));
        Assert.Equal(1344, MiniMaxH3Geometry.Round(1360));
    }
}
