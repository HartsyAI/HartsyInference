using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The base LongCat recipe run end to end on a tiny checkpoint written to disk: the recipe loads the DiT from the folder, takes umT5 and
/// the VAE from injected stand-ins (no side-model downloads), encodes the prompt, denoises, and decodes frames. The values are not meaningful; the
/// test checks that the path runs and returns the requested clip. The real umT5 and VAE are checked at the real-weight run (B6/B8).</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatRecipeEndToEndTests
{
    [Theory]
    [InlineData("base")]
    [InlineData("1.5")]
    [InlineData("1.6")]
    public void ConstructingARecipeWithAUserLoraIsRefused(string build)
    {
        using LongCatTinyAvatarRelease release = new(onePointSix: build == "1.6");
        LongCatConfig config = LongCatTinyAvatarRelease.Config;
        // Stand-ins, so a build that stopped refusing would fail here without fetching anything.
        using LongCatRecipeStandIns standIns = new(config.CaptionChannels, config.InChannels);
        IVideoRecipe recipe = build == "base" ? new LongCatVideoRecipe(standIns.Sides)
            : new LongCatAvatarRecipe(standIns.Sides, LongCatTinyAvatarRelease.Features(seed: 1));
        using CpuBackend backend = new();

        RecipeContext context = new()
        {
            CheckpointPath = release.Checkpoint, Backend = backend,
            VariantHints = build == "1.6" ? [LongCatAvatarVariants.Avatar16FamilyId] : [],
            Loras = new LoraStack { Entries = [new LoraEntry { Model = "some-lora" }] },
        };

        // A pipeline built anyway is disposed, so the release folder can still be removed and the assertion is what fails.
        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => recipe.Construct(context).Dispose());

        Assert.Contains("user LoRA", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASavedNoiseFileReplacesTheSeededNoise()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        using LongCatRecipeStandIns standIns = new(config.CaptionChannels, config.InChannels);
        string directory = Path.Combine(Path.GetTempPath(), "longcat-noise-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed: 71);
            LongCatTestData.WriteCheckpoint(directory, weights);
            LongCatTestData.DisposeAll(weights);
            // 5 frames at 16x16 are 2 latent frames of 2x2.
            int[] shape = [1, config.InChannels, 2, 2, 2];
            string first = WriteNoise(directory, "first.npy", shape, seed: 11), second = WriteNoise(directory, "second.npy", shape, seed: 12);

            using CpuBackend backend = new();
            using IVideoRecipePipeline pipeline = new LongCatVideoRecipe(standIns.Sides).Construct(new RecipeContext
            {
                CheckpointPath = directory, Backend = backend,
            });

            float[] Latents(string? noise, int seed)
            {
                VideoRequest request = new() { Prompt = "a lighthouse at dusk", NegativePrompt = "", Seed = seed };
                VideoEffectiveSettings settings = new()
                {
                    Width = 16, Height = 16, Frames = 5, Fps = 8, Steps = 2, CfgScale = 1f, Seed = seed,
                    ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
                };
                using IDisposable? scope = noise is null ? null
                    : KnobProfile.Create("longcat-noise").With(EngineKnobs.LongCatNoiseNpy, noise).Push();
                pipeline.Generate(request, settings, progress: null, CancellationToken.None);
                return standIns.Vae.Decoded[^1].Values;
            }

            // With a file the seed no longer matters, and the file does.
            Assert.Equal(Latents(first, seed: 3), Latents(first, seed: 4));
            Assert.NotEqual(Latents(first, seed: 3), Latents(second, seed: 3));
            Assert.NotEqual(Latents(null, seed: 3), Latents(null, seed: 4));
            string wrong = WriteNoise(directory, "wrong.npy", [1, config.InChannels, 3, 2, 2], seed: 13);
            Assert.Throws<InvalidDataException>(() => Latents(wrong, seed: 3));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static string WriteNoise(string directory, string name, int[] shape, int seed)
    {
        float[] values = LongCatTestData.RandomValues(new Random(seed), shape.Aggregate(1, (a, d) => a * d));
        string path = Path.Combine(directory, name);
        File.WriteAllBytes(path, NpyFile.Float32(values, shape));
        return path;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheBaseRecipeGeneratesTheRequestedClipFromATinyCheckpoint(bool int8)
    {
        VideoGenerationResult result = RunBaseClip(int8, () => new CpuBackend());

        Assert.Equal(5, result.Frames.Count);
        Assert.All(result.Frames, frame => Assert.NotNull(frame));
    }

    [Fact]
    public void TheBaseContinuationKeepsItsContextFramesAndGeneratesTheRest()
    {
        // The reference's 13-frame context (four latents) leads the output, so 17 output frames are the context plus one latent group.
        byte[][] source = Enumerable.Range(0, 13).Select(i => Enumerable.Range(0, 16 * 16 * 3).Select(b => (byte)((b + i * 7) % 251)).ToArray()).ToArray();

        VideoGenerationResult result = RunBaseClip(false, () => new CpuBackend(), frames: 17, continuation: source);

        Assert.Equal(17, result.Frames.Count);
        Assert.All(result.Frames, frame => Assert.NotNull(frame));
    }

    /// <summary>Runs the base recipe on a tiny checkpoint written to disk, on the backend <paramref name="makeBackend"/> makes (CPU, CUDA or
    /// Vulkan). Shared by the CPU smoke test above and the device parity tests (<c>LongCatRecipeGpuParityTests</c>), so the two run the same body.
    /// <paramref name="continuation"/> (RGB24 frames at 16×16) runs the continuation instead of a text clip.</summary>
    internal static VideoGenerationResult RunBaseClip(bool int8, Func<IBackend> makeBackend, int frames = 5, byte[][]? continuation = null)
    {
        using LongCatRecipeStandIns standIns = new(LongCatTestData.Tiny.CaptionChannels, LongCatTestData.Tiny.InChannels);
        string directory = Path.Combine(Path.GetTempPath(), "longcat-e2e-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            Dictionary<string, Tensor> weights = LongCatTestData.Weights(LongCatTestData.Tiny, seed: 71);
            if (int8)
            {
                // The release's per-row INT8 layout: the quantized linears carry .weight_int8 and .weight_scale, under the INT8 index name.
                Dictionary<string, Tensor> quantized = LongCatTestData.QuantizePerRow(weights);
                LongCatTestData.WriteCheckpoint(directory, quantized, indexFileName: LongCatCheckpoint.Int8IndexFileName);
                foreach ((string key, Tensor tensor) in quantized)
                {
                    if (!weights.TryGetValue(key, out Tensor? dense) || !ReferenceEquals(dense, tensor))
                        tensor.Dispose();
                }
            }
            else
            {
                LongCatTestData.WriteCheckpoint(directory, weights);
            }
            LongCatTestData.DisposeAll(weights);

            using IBackend backend = makeBackend();
            IVideoRecipePipeline pipeline = new LongCatVideoRecipe(standIns.Sides).Construct(new RecipeContext
            {
                CheckpointPath = directory, Backend = backend,
            });
            try
            {
                VideoRequest request = new()
                {
                    Prompt = "a lighthouse at dusk", NegativePrompt = "", Seed = 3,
                };
                VideoEffectiveSettings settings = new()
                {
                    Width = 16, Height = 16, Frames = frames, Fps = 8, Steps = 2, CfgScale = 1f, Seed = 3,
                    ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
                };

                return continuation is null
                    ? pipeline.Generate(request, settings, progress: null, CancellationToken.None)
                    : ((LongCatVideoRecipePipeline)pipeline).GenerateFrom(request, settings, continuation, progress: null, CancellationToken.None);
            }
            finally
            {
                pipeline.Dispose();
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
