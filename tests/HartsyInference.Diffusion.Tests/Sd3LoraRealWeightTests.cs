using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 1.5: <see cref="Sd3Recipe"/> now merges a requested LoRA stack
/// into the transformer/CLIP-L/CLIP-G weight dicts before <c>LoadWeights</c> (mirroring the existing
/// <c>SdxlRecipe</c>/<c>Flux1Recipe</c> call shape) and declares <see cref="ImageFeatures.Lora"/>. Same-seed A/B
/// (LoRA off vs. on) on real SD3.5 Medium weights — confirms the merge actually reaches the transformer, not
/// just that <c>Supports</c> stopped refusing the request.</summary>
[Trait("Category", "Integration")]
public sealed class Sd3LoraRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public Sd3LoraRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Sd3Recipe_WithLora_ProducesVisiblyDifferentOutputThanWithout()
    {
        string checkpointPath = TestPaths.Sd35.MediumTransformerOnly;
        string loraPath = "/home/hartsy/Desktop/HartsyInference/Models/Lora/SD3/trpfrog-sd3.5-medium-lora.safetensors";
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: SD3.5 checkpoint not found at {checkpointPath}.");
            return;
        }
        if (!File.Exists(loraPath))
        {
            _output.WriteLine($"SKIPPED: SD3 LoRA not found at {loraPath}.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        Assert.True(new Sd3Recipe().Supports.HasFlag(ImageFeatures.Lora), "Sd3Recipe should now declare ImageFeatures.Lora.");

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photo of a small robot in a garden",
            Width = 512,
            Height = 512,
            Steps = 8,
            CfgScale = 4.5f,
            Seed = 424242,
        };

        Directory.CreateDirectory(TestPaths.OutputDir);
        byte[] withoutLora = Generate(checkpointPath, ptxDir, request, loras: null);
        _output.WriteLine($"Generated without LoRA ({withoutLora.Length} bytes).");
        string withoutPath = Path.Combine(TestPaths.OutputDir, "sd3_lora_off.rgb");
        File.WriteAllBytes(withoutPath, withoutLora);
        _output.WriteLine($"Wrote {withoutPath}.");

        LoraStack loraStack = new LoraStack { Entries = [new LoraEntry { Model = loraPath, Weight = 1.0 }] };
        byte[] withLora = Generate(checkpointPath, ptxDir, request, loras: loraStack);
        _output.WriteLine($"Generated with LoRA ({withLora.Length} bytes).");
        string withPath = Path.Combine(TestPaths.OutputDir, "sd3_lora_on.rgb");
        File.WriteAllBytes(withPath, withLora);
        _output.WriteLine($"Wrote {withPath}.");

        Assert.Equal(withoutLora.Length, withLora.Length);
        long diffSum = 0;
        for (int i = 0; i < withoutLora.Length; i++)
        {
            diffSum += Math.Abs(withoutLora[i] - withLora[i]);
        }
        double meanAbsDiff = diffSum / (double)withoutLora.Length;
        _output.WriteLine($"Mean absolute per-byte difference: {meanAbsDiff:F2} (0 would mean byte-identical, i.e. the LoRA had no effect).");
        Assert.True(meanAbsDiff > 1.0, $"LoRA-on vs. LoRA-off outputs are nearly identical (mean abs diff {meanAbsDiff:F2}) — the merge likely isn't reaching the transformer.");
    }

    private static byte[] Generate(string checkpointPath, string ptxDir, ImageRequest request, LoraStack? loras)
    {
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        RecipeContext context = new RecipeContext
        {
            CheckpointPath = checkpointPath,
            Backend = backend,
            Loras = loras,
        };
        using IRecipePipeline pipeline = new Sd3Recipe().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
