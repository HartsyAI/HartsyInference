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
