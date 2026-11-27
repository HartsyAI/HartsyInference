using System.Globalization;
using HartsyInference.Core.Tensors;
using HartsyInference.Audio.Dsp.Loudness;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Numerics;
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

    [Fact]
    public void AGgufRepackRunsTheRecipeAsItsDecodedWeightsDo()
    {
        // The 1.5 build with its distillation LoRA, whose fused qkv and kv modules land on the weights the loader joins from the repack's parts.
        using IDisposable f32 = LongCatTestData.PinF32Dit();
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(LongCatTinyAvatarRelease.Config, seed: 71);
        Dictionary<string, Tensor>? decoded = null;
        try
        {
            List<(int Frames, float[] Values)> repack = Run(onePointSix: false, audioSeed: 1,
                writeDit: directory => decoded = LongCatGgufTestData.Write(directory, weights, "Q8_0", "Q8_0")).Vae.Decoded;
            List<(int Frames, float[] Values)> dense = Run(onePointSix: false, audioSeed: 1, dense: decoded).Vae.Decoded;
            List<(int Frames, float[] Values)> original = Run(onePointSix: false, audioSeed: 1, dense: weights).Vae.Decoded;

            Assert.Equal(dense.Count, repack.Count);
            for (int segment = 0; segment < dense.Count; segment++)
            {
                double error = LongCatTestData.RelRms(repack[segment].Values, dense[segment].Values);
                Assert.True(error < 1e-5, $"segment {segment}: relRMS {error}");
                Assert.True(LongCatTestData.RelRms(repack[segment].Values, original[segment].Values) > 1e-5, "the repack is quantized");
            }
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
            if (decoded is not null)
                LongCatTestData.DisposeAll(decoded);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SavedNoiseNamedPerSegmentReplacesEverySegmentsDraw(bool onePointSix)
    {
        // Each run is five or six latent frames of 2 x 2: segment 0 as it is generated, a later one with its reference slot in front.
        const int FirstFrames = 5, LaterFrames = 6, Conditioning = 5;
        string directory = Path.Combine(Path.GetTempPath(), "longcat-segment-noise-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            int channels = LongCatTinyAvatarRelease.Config.InChannels;
            string pattern = Path.Combine(directory, "noise_{segment}.npy");
            void Write(int segment, int seed, bool conditioningOnly = false)
            {
                int frames = segment == 0 ? FirstFrames : LaterFrames, perFrame = 4;
                float[] values = LongCatTestData.RandomValues(new Random(100 * segment + 7), channels * frames * perFrame);
                float[] other = LongCatTestData.RandomValues(new Random(seed), values.Length);
                // A change of every value, or of the frames the conditioning overwrites alone.
                for (int i = 0; i < values.Length; i++)
                    if (seed != 0 && (!conditioningOnly || i / perFrame % frames < Conditioning))
                        values[i] = other[i];
                File.WriteAllBytes(pattern.Replace("{segment}", segment.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal),
                    NpyFile.Float32(values, [1, channels, frames, 2, 2]));
            }

            List<(int Frames, float[] Values)> Segments(string? noise, int seed)
            {
                using IDisposable? scope = noise is null ? null
                    : KnobProfile.Create("longcat-segment-noise").With(EngineKnobs.LongCatNoiseNpy, noise).Push();
                return Run(onePointSix, audioSeed: 1, seed: seed).Vae.Decoded;
            }

            for (int segment = 0; segment < 3; segment++)
                Write(segment, seed: 0);
            List<(int Frames, float[] Values)> saved = Segments(pattern, seed: 3), otherSeed = Segments(pattern, seed: 4);
            // With a file for every segment the seed draws nothing.
            for (int segment = 0; segment < 3; segment++)
                Assert.Equal(saved[segment].Values, otherSeed[segment].Values);

            // The reference and context frames of a later segment's file are never read.
            Write(1, seed: 55, conditioningOnly: true);
            List<(int Frames, float[] Values)> conditioningChanged = Segments(pattern, seed: 3);
            for (int segment = 0; segment < 3; segment++)
                Assert.Equal(saved[segment].Values, conditioningChanged[segment].Values);

            // The last segment's file reaches the last segment, and no earlier one.
            Write(1, seed: 0);
            Write(2, seed: 56);
            List<(int Frames, float[] Values)> lastChanged = Segments(pattern, seed: 3);
            Assert.Equal(saved[0].Values, lastChanged[0].Values);
            Assert.Equal(saved[1].Values, lastChanged[1].Values);
            Assert.NotEqual(saved[2].Values, lastChanged[2].Values);

            // A path without the mark replaces the first segment's draw alone: the later ones still follow the seed.
            string first = Path.Combine(directory, "noise_0.npy");
            List<(int Frames, float[] Values)> firstOnly = Segments(first, seed: 3), firstOnlyOtherSeed = Segments(first, seed: 4);
            Assert.Equal(firstOnly[0].Values, firstOnlyOtherSeed[0].Values);
            Assert.NotEqual(firstOnly[1].Values, firstOnlyOtherSeed[1].Values);

            // Every segment the mark names is required.
            File.Delete(Path.Combine(directory, "noise_2.npy"));
            Assert.Throws<FileNotFoundException>(() => Segments(pattern, seed: 3));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>Runs the chain on the backend <paramref name="makeBackend"/> makes (CPU by default). Shared by the CPU tests above and the device parity tests.</summary>
    internal static (VideoGenerationResult Result, LongCatStandInVae Vae, List<(float[] Speech, int Fps)> FeatureCalls) Run(bool onePointSix,
        int audioSeed, Func<IBackend>? makeBackend = null, bool? int8 = null, Dictionary<string, Tensor>? dense = null, int seed = 3,
        Action<string>? writeDit = null)
    {
        LongCatConfig config = LongCatTinyAvatarRelease.Config;
        using LongCatTinyAvatarRelease release = new(onePointSix, int8, dense, writeDit);
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
            Prompt = "a person speaking", NegativePrompt = "", Seed = seed, ChainTotalFrames = ChainFrames,
            InitImage = LongCatTinyAvatarRelease.Portrait(Size), VideoAudioReference = LongCatTinyAvatarRelease.Speech(1.2),
        };
        VideoEffectiveSettings settings = new()
        {
            Width = Size, Height = Size, Frames = SegmentFrames, Fps = LongCatAvatarRecipe.OutputFps, Steps = 8, CfgScale = 1f, Seed = seed,
            ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
        };

        VideoGenerationResult result = pipeline.Generate(request, settings, progress: null, CancellationToken.None);

        return (result, standIns.Vae, featureCalls);
    }
}
