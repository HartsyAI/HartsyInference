using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 1.4 (Ideogram 4) — the last of the three architectures this
/// item covers. <see cref="Ideogram4Pipeline"/> already accepted a <c>RegionalPlan</c> and threaded it into its
/// per-step forward pass with zero callers, same as Flux.1/Z-Image. Unlike Flux (recipe pipeline has no encoder
/// reference) and like Z-Image (recipe pipeline holds the SAME <c>LlamaStyleEncoder</c> instance the underlying
/// pipeline uses internally), but Ideogram's base-prompt encode is a multi-layer INTERLEAVED tap
/// (<c>Ideogram4Config.QwenActivationLayersHf</c>, 13 layers × 4096 = <c>LlmFeaturesDim</c>), not a single
/// penultimate layer — so region text needs the identical multi-layer call, added as
/// <c>Ideogram4Pipeline.EncodeRegionText</c> (mirrors <c>FluxPipeline.EncodeRegionText</c>'s shape, not
/// Z-Image's direct-field-access shape, because getting the tap-layer configuration right needs to live next to
/// where the base prompt's own call is).</summary>
[Trait("Category", "Integration")]
public sealed class Ideogram4RegionalPromptingRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public Ideogram4RegionalPromptingRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Ideogram4Recipe_TwoRegionPrompt_LandsSpatiallyDistinct()
    {
        string checkpointPath = Path.Combine(TestPaths.Ideogram4.Dir, "ideogram4_fp8_scaled.safetensors");
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: Ideogram 4 checkpoint not found at {checkpointPath}.");
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

        Assert.True(new Ideogram4Recipe().Supports.HasFlag(ImageFeatures.Regional), "Ideogram4Recipe should now declare ImageFeatures.Regional.");

        const int width = 512, height = 512;
        // Structured JSON caption per Ideogram4RecipePipeline's own warning: a bare prompt trips its safety filter.
        const string prompt = "{\"style\": \"minimalist studio background\"} "
            + "<region:0,0,0.5,1>{\"subject\": \"solid bright red color, pure red, #ff0000\"}"
            + "<region:0.5,0,0.5,1>{\"subject\": \"solid bright blue color, pure blue, #0000ff\"}"
            + "<region:end>";

        byte[] rgb = Generate(ptxDir, checkpointPath, new ImageRequest
        {
            Prompt = prompt,
            Width = width,
            Height = height,
            Steps = 12,
            Seed = 123,
        });

        string outPath = Path.Combine(RepoRoot.Path, "ideogram4_regional_output.rgb");
        File.WriteAllBytes(outPath, rgb);
        _output.WriteLine($"Wrote {outPath} ({rgb.Length} bytes, {width}x{height}) for visual inspection.");

        (double lR, double lG, double lB) = MeanRgb(rgb, width, height, 0, width / 2);
        (double rR, double rG, double rB) = MeanRgb(rgb, width, height, width / 2, width);
        _output.WriteLine($"Left half mean RGB:  ({lR:F1}, {lG:F1}, {lB:F1})");
        _output.WriteLine($"Right half mean RGB: ({rR:F1}, {rG:F1}, {rB:F1})");

        double leftRedDominance = lR - lB;
        double rightBlueDominance = rB - rR;
        _output.WriteLine($"Left red-dominance (R-B): {leftRedDominance:F1}; Right blue-dominance (B-R): {rightBlueDominance:F1}");
        Assert.True(leftRedDominance > 10.0, $"Left half is not red-dominant (R-B = {leftRedDominance:F1}) — regional conditioning may not be landing spatially.");
        Assert.True(rightBlueDominance > 10.0, $"Right half is not blue-dominant (B-R = {rightBlueDominance:F1}) — regional conditioning may not be landing spatially.");
    }

    private static (double R, double G, double B) MeanRgb(byte[] rgb, int width, int height, int xStart, int xEnd)
    {
        long sumR = 0, sumG = 0, sumB = 0;
        long count = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = xStart; x < xEnd; x++)
            {
                int i = (y * width + x) * 3;
                sumR += rgb[i];
                sumG += rgb[i + 1];
                sumB += rgb[i + 2];
                count++;
            }
        }
        return (sumR / (double)count, sumG / (double)count, sumB / (double)count);
    }

    private static byte[] Generate(string ptxDir, string checkpointPath, ImageRequest request)
    {
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        RecipeContext context = new RecipeContext
        {
            CheckpointPath = checkpointPath,
            Backend = backend,
        };
        using IRecipePipeline pipeline = new Ideogram4Recipe().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
