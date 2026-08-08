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

    /// <summary>The main generation path must keep the size the caller asked for — only the area cap may move it.
    /// <see cref="MiniMaxH3Geometry.AdaptCanvas"/> would renormalise all of these onto a 768 short edge, which is
    /// right for a reference clip and wrong for a generation.</summary>
    [Theory]
    [InlineData(960, 960)]
    [InlineData(1344, 768)]
    [InlineData(704, 1280)]
    [InlineData(512, 512)]
    public void ClampToMaxArea_UnderTheCap_LeavesTheRequestedCanvasAlone(int width, int height)
    {
        Assert.Equal((width, height), MiniMaxH3Geometry.ClampToMaxArea(width, height));
    }

    /// <summary>Over the cap, the canvas comes back under it at roughly the requested aspect. 280x3904 is the worst
    /// case an exhaustive sweep found: barely over the cap at an extreme aspect, where rounding BOTH axes to the
    /// nearest grid step lands back above it — the reason the clamp walks the longer axis down afterwards.</summary>
    [Theory]
    [InlineData(1920, 1080)]
    [InlineData(3840, 2160)]
    [InlineData(2048, 2048)]
    [InlineData(1080, 1920)]
    [InlineData(280, 3904)]
    public void ClampToMaxArea_OverTheCap_ComesBackUnderItAtTheRequestedAspect(int width, int height)
    {
        (int w, int h) = MiniMaxH3Geometry.ClampToMaxArea(width, height);
        Assert.Equal(0, w % MiniMaxH3Geometry.CanvasMultiple);
        Assert.Equal(0, h % MiniMaxH3Geometry.CanvasMultiple);
        Assert.True((long)w * h <= MiniMaxH3Geometry.MaxPixels,
            $"{width}x{height} clamped to {w}x{h} = {(long)w * h}px, still past the {MiniMaxH3Geometry.MaxPixels}px cap");
        Assert.True((long)w * h < (long)width * height, $"{width}x{height} must shrink, got {w}x{h}");
        // Aspect is held per-axis rather than as a ratio: at 280x3904 the clamped width is 288, where a single 32-px
        // grid step is 11% of the axis, so a ratio tolerance loose enough to pass would be too loose to mean anything.
        double s = Math.Sqrt((double)MiniMaxH3Geometry.MaxPixels / ((double)width * height));
        Assert.True(Math.Abs(w - width * s) <= MiniMaxH3Geometry.CanvasMultiple,
            $"width {w} is more than one grid step off the aspect-preserving {width * s:F0}");
        Assert.True(Math.Abs(h - height * s) <= MiniMaxH3Geometry.CanvasMultiple,
            $"height {h} is more than one grid step off the aspect-preserving {height * s:F0}");
    }

    /// <summary>An exhaustive sweep of the clamp, because the failure mode it guards is a shape that silently costs
    /// more compute than the model was trained for — one aspect slipping through is one too many.</summary>
    [Fact]
    public void ClampToMaxArea_NeverExceedsTheCap_AcrossEveryAspect()
    {
        for (int w = 64; w <= 4096; w += 8)
        {
            for (int h = 64; h <= 4096; h += 8)
            {
                (int cw, int ch) = MiniMaxH3Geometry.ClampToMaxArea(w, h);
                Assert.True((long)cw * ch <= MiniMaxH3Geometry.MaxPixels,
                    $"{w}x{h} -> {cw}x{ch} = {(long)cw * ch}px exceeds the cap");
            }
        }
    }

    /// <summary>The envelope is a quality warning threshold, not a cap — it has to sit on the same 17k+5 grid every
    /// accepted frame count does, or a request at exactly the envelope would snap past its own threshold.</summary>
    [Fact]
    public void TrainedFrameEnvelope_IsOnTheFrameGrid()
    {
        Assert.Equal(MiniMaxH3Geometry.TrainedFrameEnvelope,
            MiniMaxH3Geometry.AlignFrameCount(MiniMaxH3Geometry.TrainedFrameEnvelope));
        Assert.Equal(5, MiniMaxH3Geometry.TrainedFrameEnvelope % 17);
        Assert.True(MiniMaxH3Geometry.TrainedFrameEnvelope > 124,
            "the envelope must sit above H3's own 124-frame default, or every default generation would warn");
    }
}
