using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 1.4 (Z-Image). Same wiring shape as
/// <see cref="Flux1RegionalPromptingRealWeightTests"/>, but simpler: <see cref="ZImageRecipePipeline"/> already
/// owns the Qwen3-4B encoder directly (<c>ZImagePipeline</c> accepts pre-computed caption embeddings — the
/// text-encoder forward lives outside it entirely), so no new encode method was needed on the pipeline class,
/// just a <c>RegionalPromptResolver.Resolve</c> call using the already-resident encoder before its
/// <c>FreeWeights</c>.</summary>
[Trait("Category", "Integration")]
public sealed class ZImageRegionalPromptingRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public ZImageRegionalPromptingRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ZImageRecipe_TwoRegionPrompt_LandsSpatiallyDistinct()
    {
        if (!File.Exists(TestPaths.ZImage.Turbo))
        {
            _output.WriteLine($"SKIPPED: Z-Image Turbo checkpoint not found at {TestPaths.ZImage.Turbo}.");
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

        Assert.True(new ZImageRecipe().Supports.HasFlag(ImageFeatures.Regional), "ZImageRecipe should now declare ImageFeatures.Regional.");

        const int width = 512, height = 512;
        const string prompt = "a simple flat-color studio background, minimalist "
            + "<region:0,0,0.5,1>solid bright red color, pure red, #ff0000"
            + "<region:0.5,0,0.5,1>solid bright blue color, pure blue, #0000ff"
            + "<region:end>";

        byte[] rgb = Generate(ptxDir, new ImageRequest
        {
            Prompt = prompt,
            Width = width,
            Height = height,
            Steps = 8,
            CfgScale = 1.0f,
            Seed = 123,
        });

        string outPath = Path.Combine(RepoRoot.Path, "zimage_regional_output.rgb");
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

    private static byte[] Generate(string ptxDir, ImageRequest request)
    {
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        RecipeContext context = new RecipeContext
        {
            CheckpointPath = TestPaths.ZImage.Turbo,
            Backend = backend,
        };
        using IRecipePipeline pipeline = new ZImageRecipe().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
