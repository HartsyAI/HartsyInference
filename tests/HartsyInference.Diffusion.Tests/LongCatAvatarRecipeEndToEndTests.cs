using HartsyInference.Audio.Io;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
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

    /// <summary>The tiny avatar DiT with the real audio window: 5 blocks of 1280 channels.</summary>
    private static LongCatConfig Config => LongCatTestData.TinyAvatar with { AudioBlock = 5, AudioChannel = 1280 };

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
        LongCatConfig config = Config;
        string root = Path.Combine(Path.GetTempPath(), "longcat-avatar-e2e-" + Guid.NewGuid().ToString("N"));
        // The release layouts: the DiT folder, with the 1.5 LoRA in lora/ or the 1.6 pair in loraA/ and loraB/ beside it.
        string checkpoint = Path.Combine(root, onePointSix ? "base_model_int8" : "base_model");
        Directory.CreateDirectory(checkpoint);
        try
        {
            WriteCheckpoint(checkpoint, config, onePointSix);
            foreach ((string folder, string file, int seed) in onePointSix
                ? new[] { ("loraA", "lora.safetensors", 81), ("loraB", "lora.safetensors", 82) }
                : [("lora", "dmd_lora.safetensors", 81)])
            {
                Directory.CreateDirectory(Path.Combine(root, folder));
                using LongCatLoraFixture lora = new(Path.Combine(root, folder, file), config, seed);
            }

            List<(int Samples, int Fps)> featureCalls = [];
            LongCatDrivingFeatures features = (speech, fps) =>
            {
                featureCalls.Add((speech.Length, fps));
                Random rng = new(audioSeed);
                return LongCatTestData.RandomValues(rng, (int)(speech.Length / (double)SampleRate * fps) * config.AudioBlock * config.AudioChannel);
            };

            using LongCatRecipeStandIns standIns = new(config.CaptionChannels, config.InChannels);
            using CpuBackend backend = new();
            using IVideoRecipePipeline pipeline = new LongCatAvatarRecipe(standIns.Sides, features).Construct(new RecipeContext
            {
                CheckpointPath = checkpoint, Backend = backend,
                VariantHints = onePointSix ? [LongCatAvatarVariants.Avatar16FamilyId] : [],
            });
            VideoRequest request = new()
            {
                Prompt = "a person speaking", NegativePrompt = "", Seed = 3, ChainTotalFrames = ChainFrames,
                InitImage = Portrait(), VideoAudioReference = Speech(1.2),
            };
            VideoEffectiveSettings settings = new()
            {
                Width = Size, Height = Size, Frames = SegmentFrames, Fps = LongCatAvatarRecipe.OutputFps, Steps = 8, CfgScale = 1f, Seed = 3,
                ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
            };

            VideoGenerationResult result = pipeline.Generate(request, settings, progress: null, CancellationToken.None);

            return (result, standIns.Vae, featureCalls);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void WriteCheckpoint(string directory, LongCatConfig config, bool int8)
    {
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(config, seed: 71);
        Dictionary<string, Tensor> written = int8 ? LongCatTestData.QuantizePerRow(weights) : weights;
        try
        {
            LongCatTestData.WriteCheckpoint(directory, written, writeConfig: false,
                indexFileName: int8 ? LongCatCheckpoint.Int8IndexFileName : LongCatCheckpoint.IndexFileName);
            File.WriteAllText(Path.Combine(directory, "config.json"), $$"""
                {"in_channels":{{config.InChannels}},"out_channels":{{config.OutChannels}},"hidden_size":{{config.HiddenSize}},
                 "depth":{{config.Depth}},"num_heads":{{config.NumHeads}},"caption_channels":{{config.CaptionChannels}},"mlp_ratio":{{config.MlpRatio}},
                 "adaln_tembed_dim":{{config.AdalnTembedDim}},"frequency_embedding_size":{{config.FrequencyEmbeddingSize}},"patch_size":[1,2,2],
                 "text_tokens_zero_pad":true,"output_dim":{{config.AudioOutputDim}},"audio_window":{{config.AudioWindow}},
                 "audio_block":{{config.AudioBlock}},"audio_channel":{{config.AudioChannel}},"intermediate_dim":{{config.AudioIntermediateDim}},
                 "context_tokens":{{config.AudioContextTokens}},"vae_scale":{{config.AudioVaeScale}}}
                """);
        }
        finally
        {
            foreach ((string key, Tensor tensor) in written)
                if (!weights.TryGetValue(key, out Tensor? dense) || !ReferenceEquals(dense, tensor))
                    tensor.Dispose();
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static ImageData Portrait()
    {
        byte[] rgb = new byte[Size * Size * 3];
        for (int i = 0; i < rgb.Length; i++)
            rgb[i] = (byte)(i % 251);
        return new ImageData { Rgb = rgb, Width = Size, Height = Size };
    }

    private static AudioClip Speech(double seconds)
    {
        float[] samples = new float[(int)(seconds * SampleRate)];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = 0.2f * MathF.Sin(2f * MathF.PI * 220f * i / SampleRate);
        using MemoryStream stream = new();
        WavFile.WriteMono16(stream, samples, SampleRate);
        return new AudioClip { Data = stream.ToArray(), Format = "wav" };
    }
}
