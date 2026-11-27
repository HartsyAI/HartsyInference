using HartsyInference.Core.Backends.Cpu;
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
    [InlineData(false)]
    [InlineData(true)]
    public void TheBaseRecipeGeneratesTheRequestedClipFromATinyCheckpoint(bool int8)
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

            using CpuBackend backend = new();
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
                    Width = 16, Height = 16, Frames = 5, Fps = 8, Steps = 2, CfgScale = 1f, Seed = 3,
                    ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
                };

                VideoGenerationResult result = pipeline.Generate(request, settings, progress: null, CancellationToken.None);

                Assert.Equal(5, result.Frames.Count);
                Assert.All(result.Frames, frame => Assert.NotNull(frame));
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
