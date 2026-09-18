using Xunit;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Driving audio has to be visible to planning, not only to the pipeline that consumes it. Converted into
/// a mask after construction it would be invisible to the profile check, so an unrelated family would accept
/// <c>--driving-audio</c> and silently ignore it, and a sparse profile would pass planning only to be refused at
/// the execution boundary.</summary>
public sealed class MiniMaxH3DrivingAudioRequestTests
{
    private static AudioClip Track() => new AudioClip { Data = [0x52, 0x49, 0x46, 0x46], Format = "wav" };

    [Fact]
    public void DrivingAudioIsARequestedFeature()
    {
        VideoRequest request = new VideoRequest { Prompt = "test", VideoAudioReference = Track() };
        Assert.True(VideoService.RequestedFeatures(request).HasFlag(VideoFeatures.DrivingAudio));
    }

    [Fact]
    public void AnOrdinaryRequestDoesNotAskForDrivingAudio()
    {
        VideoRequest request = new VideoRequest { Prompt = "test", Frames = 141 };
        Assert.False(VideoService.RequestedFeatures(request).HasFlag(VideoFeatures.DrivingAudio));
    }

    /// <summary>The output timing edits that would slide the picture against the track driving it.</summary>
    /// <remarks>Frame edits reach the frames only — the leading frames are dropped, the boomerang is built, and a
    /// differing fps is muxed rather than resampled — while the soundtrack is trimmed or padded at its end alone. Lip
    /// sync is the whole point of driving audio, so each of these has to refuse rather than quietly desync.</remarks>
    [Theory]
    [InlineData("trim", "start")]
    [InlineData("boomerang", "reverse")]
    [InlineData("fps", "speed")]
    public void TimingEditsThatWouldDesyncTheDrivenTrackAreRefused(string edit, string expectedInMessage)
    {
        VideoRequest request = edit switch
        {
            "trim" => new VideoRequest { Prompt = "test", VideoAudioReference = Track(), TrimVideoStartFrames = 4 },
            "boomerang" => new VideoRequest { Prompt = "test", VideoAudioReference = Track(), VideoBoomerang = true },
            _ => new VideoRequest { Prompt = "test", VideoAudioReference = Track(), Fps = 30 },
        };

        ArgumentException error = Assert.Throws<ArgumentException>(
            () => MiniMaxH3RecipePipeline.RejectTimingEditsThatBreakLipSync(request));
        Assert.Contains(expectedInMessage, error.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>An END trim stays allowed: the soundtrack is trimmed at its end too, so the start stays aligned.</summary>
    [Fact]
    public void AnEndTrimIsStillAllowedWithDrivingAudio()
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            VideoAudioReference = Track(),
            TrimVideoEndFrames = 4,
            Fps = 24,
        };

        MiniMaxH3RecipePipeline.RejectTimingEditsThatBreakLipSync(request);
    }

    /// <summary>Reference audio describes the sound to aim for; driving audio fixes it. A request carrying only the
    /// former must not be classified as the latter, or planning would demand support the caller never asked for.</summary>
    [Fact]
    public void ReferenceAudioIsNotDrivingAudio()
    {
        VideoRequest request = new VideoRequest { Prompt = "test", ReferenceAudios = [Track()] };
        VideoFeatures features = VideoService.RequestedFeatures(request);
        Assert.True(features.HasFlag(VideoFeatures.ReferenceAudios));
        Assert.False(features.HasFlag(VideoFeatures.DrivingAudio));
    }

    /// <summary>The recipe is the release authority the profile's features are intersected against.</summary>
    [Fact]
    public void TheH3RecipeDeclaresDrivingAudio()
        => Assert.True(new MiniMaxH3Recipe().Supports.HasFlag(VideoFeatures.DrivingAudio));
}
