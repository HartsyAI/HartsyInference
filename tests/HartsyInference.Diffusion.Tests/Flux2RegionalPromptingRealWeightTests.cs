using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 3.7 (Flux.2 half): <see cref="Flux2Transformer"/>'s double/single
/// blocks gained an <c>attnBias</c> slot (mirroring <see cref="FluxTransformer"/>'s — same "text-first" joint
/// [txt|img] concat order, confirmed by reading both transformers' block code before wiring this, not assumed by
/// analogy), wired through <c>Flux2Pipeline.GenerateFromTokens</c> and a new <c>Flux2RecipePipeline.
/// BuildRegionalPlan</c> (mirrors <c>Flux1RecipePipeline</c>'s). Uses the local GGUF (Q4_K_S) Dev checkpoint — no
/// fp8/safetensors Flux.2 Dev build is present on this box.
/// <para>Verification per the same bar Tier 1.4 used: a two-region prompt, confirming each region's content lands
/// SPATIALLY DISTINCT — checked by comparing each half's mean color channel against the other half's, not merely
/// diffing against a no-region baseline.</para></summary>
[Trait("Category", "Integration")]
public sealed class Flux2RegionalPromptingRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public Flux2RegionalPromptingRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Flux2Recipe_TwoRegionPrompt_LandsSpatiallyDistinct()
    {
        // No fp8/safetensors Dev build on this box — the local checkpoint is the Q4_K_S GGUF repack, which
        // Flux2Recipe already detects and loads (isGguf branch).
        const string checkpointPath = "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Flux2/flux2-dev-Q4_K_S.gguf";
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: Flux.2 Dev GGUF checkpoint not found at {checkpointPath}.");
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

        Assert.True(new Flux2Recipe().Supports.HasFlag(ImageFeatures.Regional), "Flux2Recipe should now declare ImageFeatures.Regional.");

        // Same construction as Flux1RegionalPromptingRealWeightTests: pure color halves are a more reliable
        // spatial-distinctness signal than two different subjects (no vision tooling needed to check placement).
        const int width = 512, height = 512;
        const string prompt = "a simple flat-color studio background, minimalist "
            + "<region:0,0,0.5,1>solid bright red color, pure red, #ff0000"
            + "<region:0.5,0,0.5,1>solid bright blue color, pure blue, #0000ff"
            + "<region:end>";

        byte[] rgb = Generate(ptxDir, checkpointPath, new ImageRequest
        {
            Prompt = prompt,
            Width = width,
            Height = height,
            Steps = 8,
            CfgScale = 0f,
            Seed = 123,
        });

        string outPath = Path.Combine(RepoRoot.Path, "flux2_regional_output.rgb");
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
        using IRecipePipeline pipeline = new Flux2Recipe().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
