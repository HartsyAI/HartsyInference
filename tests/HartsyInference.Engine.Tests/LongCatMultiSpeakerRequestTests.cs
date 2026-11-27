using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The multi-speaker request: which families take it, what the avatar refuses at planning, the vocal pre-step, and how the request's
/// speakers become the encoder's tracks and the DiT's regions.</summary>
public sealed class LongCatMultiSpeakerRequestTests
{
    private static readonly AudioClip Speech = new() { Data = [] };
    private static readonly ImageData Picture = new() { Rgb = new byte[32 * 16 * 3], Width = 32, Height = 16 };

    private static VideoEffectiveSettings Settings(int frames = 93) => new()
    {
        Width = 832, Height = 480, Frames = frames, Fps = 25, Steps = 8, CfgScale = 1f, Seed = 3,
        ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
    };

    private static VideoSpeakerBox Box(int yMin, int xMin, int yMax, int xMax) => new() { YMin = yMin, XMin = xMin, YMax = yMax, XMax = xMax };

    private static VideoSpeakers Two() => new() { First = new VideoSpeaker { Audio = Speech }, Second = new VideoSpeaker { Audio = Speech } };

    private static VideoRequest Request(VideoSpeakers speakers) => new() { Prompt = "p", InitImage = Picture, Speakers = speakers };

    private static string? Refusal(VideoRequest request) => new LongCatAvatarRecipe().RequestRefusal(request, Settings());

    [Fact]
    public void OnlyTheAvatarRecipeTakesSpeakers()
    {
        Assert.Equal(VideoFeatures.MultiSpeaker | VideoFeatures.InitImage, VideoService.RequestedFeatures(Request(Two())));
        Assert.True(new LongCatAvatarRecipe().Supports.HasFlag(VideoFeatures.MultiSpeaker));
        Assert.False(new LongCatVideoRecipe().Supports.HasFlag(VideoFeatures.MultiSpeaker));
        Assert.False(new LongCatVideoDistilledRecipe().Supports.HasFlag(VideoFeatures.MultiSpeaker));
        Assert.False(new WanS2VRecipe().Supports.HasFlag(VideoFeatures.MultiSpeaker));
        Assert.False(new MiniMaxH3Recipe().Supports.HasFlag(VideoFeatures.MultiSpeaker));
    }

    [Fact]
    public void ValidRequestsPass()
    {
        Assert.Null(Refusal(Request(Two())));
        Assert.Null(Refusal(Request(Two() with { Second = new VideoSpeaker() })));
        Assert.Null(Refusal(Request(new VideoSpeakers
        {
            // A box past the far edge is clipped, as the reference's slicing clips it.
            First = new VideoSpeaker { Audio = Speech, Box = Box(0, 0, 16, 12) },
            Second = new VideoSpeaker { Box = Box(0, 20, 400, 400) },
            Timing = VideoSpeakerTiming.Sequential,
            SilentBoxes = [Box(0, 12, 8, 20)],
        })));
        Assert.Null(Refusal(Request(Two()) with { ChainTotalFrames = 150 }));
    }

    [Fact]
    public void WhatTheReferenceCannotRunIsRefused()
    {
        VideoSpeakers two = Two();
        void Refused(VideoRequest request, string because)
        {
            string? refusal = Refusal(request);
            Assert.NotNull(refusal);
            Assert.Contains(because, refusal, StringComparison.Ordinal);
        }

        Refused(Request(two) with { InitImage = null }, "init image");
        Refused(Request(two) with { VideoAudioReference = Speech }, "VideoAudioReference");
        Refused(Request(two with { First = new VideoSpeaker(), Second = new VideoSpeaker() }), "at least one speaker");
        Refused(Request(two with { Timing = (VideoSpeakerTiming)7 }), "timing");
        Refused(Request(two with { First = two.First with { Box = Box(0, 0, 8, 8) } }), "both speakers' boxes");
        Refused(Request(two with { First = two.First with { Box = Box(0, 0, 0, 8) }, Second = two.Second with { Box = Box(0, 8, 8, 16) } }),
            "first speaker's box");
        Refused(Request(two with { First = two.First with { Box = Box(0, 0, 8, 8) }, Second = two.Second with { Box = Box(0, -1, 8, 16) } }),
            "second speaker's box");
        Refused(Request(two with { SilentBoxes = [Box(0, 0, 8, 8)] }), "need both speakers' boxes");
        // A transport can send a null where a speaker belongs.
        Refused(Request(two with { First = null! }), "both speakers");
        VideoSpeakers boxed = two with { First = two.First with { Box = Box(0, 0, 16, 16) }, Second = two.Second with { Box = Box(0, 16, 16, 32) } };
        Refused(Request(boxed with { SilentBoxes = [Box(0, 40, 8, 48)] }), "silent person's box");
        Refused(Request(boxed with { SilentBoxes = [null!] }), "box is missing");
        Refused(Request(boxed with { SilentBoxes = [.. Enumerable.Repeat(Box(0, 0, 8, 8), LongCatAvatarSpeakers.MaxSilentBoxes + 1)] }),
            "at most");
        Refused(Request(two) with { ChainTotalFrames = 300 }, "budget");
    }

    [Fact]
    public void TheVocalPreStepNeedsASpeakerWithAudio()
    {
        Assert.Null(VideoService.SeparateVocalsIssue(Request(Two()) with { SeparateDrivingVocals = true }));
        Assert.Null(VideoService.SeparateVocalsIssue(Request(Two() with { First = new VideoSpeaker() }) with { SeparateDrivingVocals = true }));
        Assert.NotNull(VideoService.SeparateVocalsIssue(
            Request(Two() with { First = new VideoSpeaker(), Second = new VideoSpeaker() }) with { SeparateDrivingVocals = true }));
    }

    [Fact]
    public void ASeparatedSpeakerDrivesTheVideoWithTheVocalsAndIsHeardAsRecorded()
    {
        float[] recording = [.. Enumerable.Repeat(0.5f, 16_000)], other = [.. Enumerable.Repeat(0.25f, 16_000)];
        float[] vocals = [.. Enumerable.Repeat(0.125f, 16_000)];
        VideoSpeakers speakers = new()
        {
            First = new VideoSpeaker { Audio = Wav(recording), SeparatedSpeech16k = vocals },
            Second = new VideoSpeaker { Audio = Wav(other) },
        };

        LongCatSpeakerAudio audio = LongCatAvatarSpeakers.PrepareAudio(speakers, 25, 25, normalizeLoudness: false);

        Assert.Equal(2, audio.Tracks.Length);
        Assert.Equal(0.125f, audio.Tracks[0][100]);
        Assert.InRange(audio.Soundtrack[100], 0.749f, 0.751f);
    }

    [Fact]
    public void BoxesBecomeTokenRegionsOfTheInitImage()
    {
        VideoSpeakers speakers = Two() with
        {
            First = new VideoSpeaker { Audio = Speech, Box = Box(0, 16, 16, 32) },
            Second = new VideoSpeaker { Audio = Speech, Box = Box(0, 0, 16, 16) },
        };

        LongCatSpeakers regions = LongCatAvatarSpeakers.Conditioning(speakers, Picture, tracks: 2, width: 32, height: 16, gridHeight: 1,
            gridWidth: 2);

        Assert.Equal(3, regions.Regions);
        // The first speaker is the one named first, wherever the box is.
        Assert.Equal([0f, 1f, 1f, 0f, 0f, 0f], regions.Masks.ToArray());
    }

    private static AudioClip Wav(float[] samples) => new() { Data = AudioClipCodec.EncodeWav(samples, null, 16_000), Format = "wav" };
}
