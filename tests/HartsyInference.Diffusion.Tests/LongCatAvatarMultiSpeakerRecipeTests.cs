using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar recipe run end to end with two speakers, on a tiny checkpoint written to disk: a three-segment chain with each speaker's
/// audio read on its own, a silent track for marked silent people, and the mixed soundtrack. umT5, the VAE and the Whisper features are stand-ins
/// (the features follow the speech they are given), so the pixel values mean nothing.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarMultiSpeakerRecipeTests
{
    private const int Size = 16, SegmentFrames = 17, ChainFrames = 23, SampleRate = 16_000;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ATwoSpeakerChainReadsEachSpeakerAndReturnsTheMix(bool onePointSix)
    {
        Run run = Generate(onePointSix, Speakers());

        Assert.Equal(ChainFrames, run.Result.Frames.Count);
        Assert.Equal(3, run.Decoded.Count);
        // One feature read per speaker, each over the 25 planned frames plus one second.
        Assert.Equal(2, run.FeatureCalls.Count);
        Assert.All(run.FeatureCalls, speech => Assert.Equal(2 * SampleRate, speech.Length));
        AudioBuffer audio = run.Result.Audio!;
        Assert.Equal(SampleRate, audio.SampleRate);
        Assert.Single(audio.Channels);
        Assert.Equal(ChainFrames / (double)LongCatAvatarRecipe.OutputFps, audio.Seconds, 3);
        float[] first = VideoRecipeUtils.DecodeMono16k(LongCatTinyAvatarRelease.Speech(1.2));
        float[] second = VideoRecipeUtils.DecodeMono16k(LongCatTinyAvatarRelease.Speech(1.5));
        Assert.Equal(first[1000] + second[1000], audio.Channels[0][1000]);
    }

    [Fact]
    public void SilentPeopleAddAThirdTrack()
    {
        VideoSpeakers speakers = Speakers();
        Run run = Generate(onePointSix: false, speakers with
        {
            First = speakers.First with { Box = new VideoSpeakerBox { YMin = 0, XMin = 0, YMax = 16, XMax = 8 } },
            Second = speakers.Second with { Box = new VideoSpeakerBox { YMin = 0, XMin = 8, YMax = 16, XMax = 16 } },
            SilentBoxes = [new VideoSpeakerBox { YMin = 0, XMin = 0, YMax = 8, XMax = 8 }],
        });

        Assert.Equal(3, run.FeatureCalls.Count);
        Assert.All(run.FeatureCalls[2], sample => Assert.Equal(0f, sample));
        Assert.Equal(ChainFrames, run.Result.Frames.Count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EverySpeakerInputReachesEverySegmentAndTheSameInputsRepeat(bool onePointSix)
    {
        List<float[]> first = Generate(onePointSix, Speakers()).Decoded;
        List<float[]> again = Generate(onePointSix, Speakers()).Decoded;
        List<float[]> otherSecond = Generate(onePointSix, Speakers() with
        {
            Second = new VideoSpeaker { Audio = LongCatTinyAvatarRelease.Speech(0.7) },
        }).Decoded;
        List<float[]> swappedBoxes = Generate(onePointSix, Speakers() with
        {
            First = Speakers().First with { Box = new VideoSpeakerBox { YMin = 0, XMin = 8, YMax = 16, XMax = 16 } },
            Second = Speakers().Second with { Box = new VideoSpeakerBox { YMin = 0, XMin = 0, YMax = 16, XMax = 8 } },
        }).Decoded;

        for (int segment = 0; segment < first.Count; segment++)
        {
            Assert.Equal(first[segment], again[segment]);
            Assert.NotEqual(first[segment], otherSecond[segment]);
            Assert.NotEqual(first[segment], swappedBoxes[segment]);
        }
    }

    [Fact]
    public void SequentialSpeakersTakeTurnsInTheConditioningAndTheSoundtrack()
    {
        Run run = Generate(onePointSix: true, Speakers(firstSeconds: 0.4) with { Timing = VideoSpeakerTiming.Sequential });

        float[] first = VideoRecipeUtils.DecodeMono16k(LongCatTinyAvatarRelease.Speech(0.4));
        float[] second = VideoRecipeUtils.DecodeMono16k(LongCatTinyAvatarRelease.Speech(1.5));
        Assert.Equal(second[100], run.Result.Audio!.Channels[0][first.Length + 100]);
        // The first track falls silent where the second one starts.
        Assert.All(run.FeatureCalls[0].Skip(first.Length), sample => Assert.Equal(0f, sample));
        Assert.All(run.FeatureCalls[1].Take(first.Length), sample => Assert.Equal(0f, sample));
    }

    private static VideoSpeakers Speakers(double firstSeconds = 1.2) => new()
    {
        First = new VideoSpeaker { Audio = LongCatTinyAvatarRelease.Speech(firstSeconds) },
        Second = new VideoSpeaker { Audio = LongCatTinyAvatarRelease.Speech(1.5) },
    };

    private sealed record Run(VideoGenerationResult Result, List<float[]> Decoded, List<float[]> FeatureCalls);

    private static Run Generate(bool onePointSix, VideoSpeakers speakers)
    {
        LongCatConfig config = LongCatTinyAvatarRelease.Config;
        using LongCatTinyAvatarRelease release = new(onePointSix);
        List<float[]> calls = [];
        // Features that follow the speech: each video frame's values are seeded by the samples of that frame.
        LongCatDrivingFeatures features = (speech, fps) =>
        {
            calls.Add([.. speech]);
            int frames = (int)(speech.Length / (double)SampleRate * fps), width = config.AudioBlock * config.AudioChannel, hop = SampleRate / fps;
            float[] values = new float[frames * width];
            for (int f = 0; f < frames; f++)
            {
                float energy = 0f;
                for (int i = f * hop; i < (f + 1) * hop; i++)
                    energy += speech[i] * speech[i];
                Array.Fill(values, MathF.Sqrt(energy / hop) + 0.01f * (f % 7), f * width, width);
            }
            return values;
        };
        using LongCatRecipeStandIns standIns = new(config.CaptionChannels, config.InChannels);
        using CpuBackend backend = new();
        using IVideoRecipePipeline pipeline = new LongCatAvatarRecipe(standIns.Sides, features).Construct(new RecipeContext
        {
            CheckpointPath = release.Checkpoint, Backend = backend,
            VariantHints = onePointSix ? [LongCatAvatarVariants.Avatar16FamilyId] : [],
        });
        VideoRequest request = new()
        {
            Prompt = "two people speaking", NegativePrompt = "", Seed = 3, ChainTotalFrames = ChainFrames,
            InitImage = LongCatTinyAvatarRelease.Portrait(Size), Speakers = speakers,
        };
        VideoEffectiveSettings settings = new()
        {
            Width = Size, Height = Size, Frames = SegmentFrames, Fps = LongCatAvatarRecipe.OutputFps, Steps = 8, CfgScale = 1f, Seed = 3,
            ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
        };

        VideoGenerationResult result = pipeline.Generate(request, settings, progress: null, CancellationToken.None);

        return new Run(result, [.. standIns.Vae.Decoded.Select(static call => call.Values)], calls);
    }
}
