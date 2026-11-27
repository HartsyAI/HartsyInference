using HartsyInference.Audio.Io;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
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
    [InlineData(121, 24, 97, null, 6.04)]
    [InlineData(121, 24, null, 97, 6.04)]
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
    public void ASpeechEncoderGetsTheSeparatedWaveformNotTheClip()
    {
        float[] separated = [0.25f, -0.5f, 0.125f];
        VideoRequest withStem = new() { Prompt = "p", VideoAudioReference = Clip, SeparatedSpeech16k = separated };

        // The clip here is not a decodable WAV, so returning at all proves it was never read.
        Assert.Same(separated, VideoRecipeUtils.DrivingSpeech16k(withStem, Clip));
    }

    [Fact]
    public void WithoutASeparatedWaveformTheClipIsDecoded()
    {
        float[] samples = [0f, 0.5f, -0.5f, 0.25f];
        AudioClip wav = new() { Data = AudioClipCodec.EncodeWav(samples, null, 16_000), Format = "wav" };

        float[] decoded = VideoRecipeUtils.DrivingSpeech16k(new VideoRequest { Prompt = "p", VideoAudioReference = wav }, wav);

        Assert.Equal(samples.Length, decoded.Length);
        Assert.InRange(decoded[1], 0.499f, 0.501f);
    }

    [Fact]
    public void SeparatorInputIsResampledWithTheSoxrHqStandIn()
    {
        float[] tone = new float[2400];
        for (int i = 0; i < tone.Length; i++) tone[i] = 0.4f * MathF.Sin(2f * MathF.PI * 1000f * i / 24_000f);
        AudioClip clip = new() { Data = AudioClipCodec.EncodeWav(tone, null, 24_000), Format = "wav" };
        float[] decoded = AudioClipCodec.DecodeNative(clip).ToStereo().Left;

        (float[] left, float[] right) = AudioClipCodec.DecodeStereoSoxrHq(clip, 44_100);

        Assert.Equal(SincResampler.ResampleSoxrHqStandIn(decoded, 24_000, 44_100), left);
        Assert.Equal(left, right);
    }

    [Fact]
    public void AnOddSourceRateStillDecodes()
    {
        // 44101 Hz shares no factor with 44100, so the stand-in's kernel table would be about 2e9 entries.
        Assert.False(SincResampler.SupportsSoxrHqStandIn(44_101, 44_100));
        Assert.True(SincResampler.SupportsSoxrHqStandIn(48_000, 44_100));
        AudioClip clip = new() { Data = AudioClipCodec.EncodeWav(new float[4410], null, 44_101), Format = "wav" };

        (float[] left, float[] right) = AudioClipCodec.DecodeStereoSoxrHq(clip, 44_100);

        Assert.InRange(left.Length, 4405, 4412);
        Assert.Equal(left.Length, right.Length);
    }

    [Fact]
    public void KimVocal2IsTheOneEntryRoutedToTheMdxSeparator()
    {
        // The route is by this flag and the weights loader is Kim_Vocal_2's, so a second flagged entry would load the wrong file.
        CatalogEntry flagged = Assert.Single(ModelCatalog.All, e => e.Features.HasFlag(CatalogFeatures.MdxNetSeparator));
        Assert.Equal("kim-vocal-2", flagged.Id);
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
