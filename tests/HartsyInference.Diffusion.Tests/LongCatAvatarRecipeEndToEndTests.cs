using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar recipe run end to end as a chain, on a tiny checkpoint and LoRA files written to disk: the recipe loads the DiT with its gated
/// LoRAs, encodes the prompt and the portrait, cuts the driving windows, runs three segments with the context re-encoded between them, and returns the
/// requested frames and soundtrack. umT5, the VAE and the Whisper features are stand-ins, so the pixel values mean nothing.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarRecipeEndToEndTests
{
    // 17 frames are 5 latents, one more than the chain's context, so each later segment adds one latent: 23 frames take three segments.
    private const int Size = 16, SegmentFrames = 17, ChainFrames = 23, SampleRate = 16_000;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AChainRequestRunsThreeSegmentsAndReturnsTheRequestedClip(bool onePointSix)
    {
        (VideoGenerationResult result, LongCatStandInVae vae, List<(int Samples, int Fps)> featureCalls) = Run(onePointSix, audioSeed: 1);

        Assert.Equal(ChainFrames, result.Frames.Count);
        Assert.All(result.Frames, frame => Assert.Equal(Size * Size * 3, frame.Rgb.Length));
        Assert.NotNull(result.Audio);
        Assert.Equal(ChainFrames / (double)LongCatAvatarRecipe.OutputFps, result.Audio.Seconds, 3);
        // The features are read once, for the whole clip.
        Assert.Equal((int)(1.2 * SampleRate), Assert.Single(featureCalls).Samples);
        Assert.Equal(LongCatAvatarRecipe.OutputFps, featureCalls[0].Fps);
        // Each of the three segments is decoded once, at a segment's latent length.
        Assert.Equal([5, 5, 5], vae.Decoded.Select(call => call.Frames));
        // The portrait, then the 13 context frames after each segment but the last.
        Assert.Equal([1, 13, 13], vae.EncodedClipFrames);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheDrivingAudioReachesEverySegmentAndTheSameInputsRepeat(bool onePointSix)
    {
        List<(int Frames, float[] Values)> first = Run(onePointSix, audioSeed: 1).Vae.Decoded;
        List<(int Frames, float[] Values)> again = Run(onePointSix, audioSeed: 1).Vae.Decoded;
        List<(int Frames, float[] Values)> otherAudio = Run(onePointSix, audioSeed: 2).Vae.Decoded;

        for (int segment = 0; segment < first.Count; segment++)
        {
            Assert.Equal(first[segment].Values, again[segment].Values);
            Assert.NotEqual(first[segment].Values, otherAudio[segment].Values);
        }
    }

    private static (VideoGenerationResult Result, LongCatStandInVae Vae, List<(int Samples, int Fps)> FeatureCalls) Run(bool onePointSix,
        int audioSeed)
    {
        LongCatConfig config = LongCatTinyAvatarRelease.Config;
        using LongCatTinyAvatarRelease release = new(onePointSix);
        List<(int Samples, int Fps)> featureCalls = [];

        using LongCatRecipeStandIns standIns = new(config.CaptionChannels, config.InChannels);
        using CpuBackend backend = new();
        using IVideoRecipePipeline pipeline = new LongCatAvatarRecipe(standIns.Sides, LongCatTinyAvatarRelease.Features(audioSeed, featureCalls))
            .Construct(new RecipeContext
            {
                CheckpointPath = release.Checkpoint, Backend = backend,
                VariantHints = onePointSix ? [LongCatAvatarVariants.Avatar16FamilyId] : [],
            });
        VideoRequest request = new()
        {
            Prompt = "a person speaking", NegativePrompt = "", Seed = 3, ChainTotalFrames = ChainFrames,
            InitImage = LongCatTinyAvatarRelease.Portrait(Size), VideoAudioReference = LongCatTinyAvatarRelease.Speech(1.2),
        };
        VideoEffectiveSettings settings = new()
        {
            Width = Size, Height = Size, Frames = SegmentFrames, Fps = LongCatAvatarRecipe.OutputFps, Steps = 8, CfgScale = 1f, Seed = 3,
            ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
        };

        VideoGenerationResult result = pipeline.Generate(request, settings, progress: null, CancellationToken.None);

        return (result, standIns.Vae, featureCalls);
    }
}
