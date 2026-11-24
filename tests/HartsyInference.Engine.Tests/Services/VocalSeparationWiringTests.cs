using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

public sealed class VocalSeparationWiringTests
{
    private static readonly AudioClip Clip = new() { Data = [1, 2, 3], Format = "wav" };

    [Fact]
    public void SeparatingWithoutDrivingAudioIsAPlanningError()
    {
        VideoPlanIssue? issue = VideoService.SeparateVocalsIssue(new VideoRequest { Prompt = "p", SeparateDrivingVocals = true });

        Assert.NotNull(issue);
        Assert.Equal("video.separate_vocals.no_driving_audio", issue.Code);
        Assert.Equal(VideoPlanIssueSeverity.Error, issue.Severity);
        Assert.Equal(nameof(VideoRequest.SeparateDrivingVocals), issue.Field);
    }

    [Fact]
    public void SeparatingWithDrivingAudioOrNotAskedIsFine()
    {
        Assert.Null(VideoService.SeparateVocalsIssue(new VideoRequest { Prompt = "p", SeparateDrivingVocals = true, VideoAudioReference = Clip }));
        Assert.Null(VideoService.SeparateVocalsIssue(new VideoRequest { Prompt = "p" }));
    }

    [Fact]
    public void TheFlagDoesNotAddAFeatureOfItsOwn()
    {
        // The driving audio is what a recipe must support, and that is already the DrivingAudio feature.
        VideoFeatures with = VideoService.RequestedFeatures(new VideoRequest { Prompt = "p", SeparateDrivingVocals = true, VideoAudioReference = Clip });
        VideoFeatures without = VideoService.RequestedFeatures(new VideoRequest { Prompt = "p", VideoAudioReference = Clip });
        Assert.Equal(without, with);
        Assert.True(with.HasFlag(VideoFeatures.DrivingAudio));
    }

    [Theory]
    [InlineData(93, 25, null, null, 4.72)]
    [InlineData(93, 25, 186, null, 8.44)]
    [InlineData(93, 25, null, 200, 9.0)]
    [InlineData(null, 25, null, null, null)]
    [InlineData(93, null, null, null, null)]
    public void OnlyTheClipLengthOfDrivingAudioIsSeparated(int? frames, int? fps, int? chain, int? animate, double? expected)
    {
        double? seconds = VideoService.DrivingAudioSeconds(new VideoRequest
        {
            Prompt = "p", Frames = frames, Fps = fps, ChainTotalFrames = chain, AnimateTotalFrames = animate,
        });
        Assert.Equal(expected, seconds is null ? null : Math.Round(seconds.Value, 2));
    }

    [Fact]
    public async Task DemucsOnlyOptionsAreRefusedForKimVocal2()
    {
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "kim-vocal-2", Modality = Modality.Fx };

        await Assert.ThrowsAsync<ArgumentException>(() => engine.Fx.SeparateAsync(spec, new FxSeparateRequest { Audio = Clip, Shifts = 2 }));
        await Assert.ThrowsAsync<ArgumentException>(() => engine.Fx.SeparateAsync(spec, new FxSeparateRequest { Audio = Clip, Segment = 7.8 }));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => engine.Fx.SeparateAsync(spec, new FxSeparateRequest { Audio = Clip, Overlap = 1.0 }));
    }
}
