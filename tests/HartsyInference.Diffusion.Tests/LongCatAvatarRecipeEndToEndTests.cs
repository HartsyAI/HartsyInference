using HartsyInference.Core.Tensors;
using HartsyInference.Audio.Dsp.Loudness;
using HartsyInference.Core.Backends;
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
        (VideoGenerationResult result, LongCatStandInVae vae, List<(float[] Speech, int Fps)> featureCalls) = Run(onePointSix, audioSeed: 1);

        Assert.Equal(ChainFrames, result.Frames.Count);
        Assert.All(result.Frames, frame => Assert.Equal(Size * Size * 3, frame.Rgb.Length));
        Assert.NotNull(result.Audio);
        Assert.Equal(ChainFrames / (double)LongCatAvatarRecipe.OutputFps, result.Audio.Seconds, 3);
        // The features are read once, for the whole clip: the 25 planned frames plus one second, the 1.2 s of speech padded with silence.
        (float[] speech, int fps) = Assert.Single(featureCalls);
        Assert.Equal(LongCatAvatarRecipe.OutputFps, fps);
        Assert.Equal(2 * SampleRate, speech.Length);
        Assert.All(speech.Skip((int)(1.2 * SampleRate)), sample => Assert.Equal(0f, sample));
        float[] decoded = VideoRecipeUtils.DrivingSpeech16k(new VideoRequest { Prompt = "p" }, LongCatTinyAvatarRelease.Speech(1.2));
        if (onePointSix)
        {
            // 1.6 hears the speech as it is.
            Assert.Equal(decoded, speech.Take(decoded.Length));
        }
        else
        {
            // 1.5 hears it at the reference's loudness.
            Assert.InRange(Bs1770Loudness.Integrated(speech, SampleRate), -23.001, -22.999);
            Assert.NotEqual(decoded, speech.Take(decoded.Length));
        }
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

    /// <summary>Runs the chain on the backend <paramref name="makeBackend"/> makes (CPU by default). Shared by the CPU tests above and the device parity tests.</summary>
    internal static (VideoGenerationResult Result, LongCatStandInVae Vae, List<(float[] Speech, int Fps)> FeatureCalls) Run(bool onePointSix,
        int audioSeed, Func<IBackend>? makeBackend = null, bool? int8 = null, Dictionary<string, Tensor>? dense = null)
    {
        LongCatConfig config = LongCatTinyAvatarRelease.Config;
        using LongCatTinyAvatarRelease release = new(onePointSix, int8, dense);
        List<(float[] Speech, int Fps)> featureCalls = [];

        using LongCatRecipeStandIns standIns = new(config.CaptionChannels, config.InChannels);
        using IBackend backend = (makeBackend ?? (() => new CpuBackend()))();
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
