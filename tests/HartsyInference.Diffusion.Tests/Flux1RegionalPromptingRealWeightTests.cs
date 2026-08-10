using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 1.4 (Flux.1 only — see class docs on <c>Flux1RecipePipeline.
/// BuildRegionalPlan</c> and <c>FluxPipeline.EncodeRegionText</c> for the wiring). <c>FluxPipeline.
/// GenerateFromTokens</c> already accepted a <see cref="HartsyInference.Diffusion.Prompting.RegionalPlan"/> and
/// threaded it into its per-step forward pass with zero callers; <c>RegionalPromptResolver.Resolve</c> already
/// existed and was tested in isolation with zero callers. The missing piece was purely structural: recipes own
/// the tokenizer, pipelines own the text encoder, and the resolver's <c>encodeRegion</c> delegate needs both —
/// resolved by adding <c>FluxPipeline.EncodeRegionText</c> (runs the SAME T5 encoder instance + backend the base
/// prompt uses) so the recipe pipeline can tokenize with its own tokenizer and hand the ids across.
/// <see cref="HartsyInference.Diffusion.Prompting.RegionalPlan.BaseCond"/> is a required field on the resolver's
/// signature that <c>FluxPipeline</c>'s regional path never actually reads (confirmed by a whole-repo grep for
/// <c>.BaseCond</c> — zero hits) — a throwaway placeholder tensor satisfies it.
/// <para>Verification per the plan's own bar for this item: a two-region prompt, confirming each region's
/// content lands SPATIALLY DISTINCT, not blended — checked by comparing each half's mean color channel against
/// the OTHER half's, not merely diffing against a no-region baseline (a diff alone cannot distinguish "regional
/// conditioning worked" from "regional conditioning corrupted the whole image differently").</para></summary>
[Trait("Category", "Integration")]
public sealed class Flux1RegionalPromptingRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public Flux1RegionalPromptingRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Flux1Recipe_TwoRegionPrompt_LandsSpatiallyDistinct()
    {
        if (!File.Exists(TestPaths.Flux.Schnell))
        {
            _output.WriteLine($"SKIPPED: Flux Schnell checkpoint not found at {TestPaths.Flux.Schnell}.");
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

        Assert.True(new Flux1Recipe().Supports.HasFlag(ImageFeatures.Regional), "Flux1Recipe should now declare ImageFeatures.Regional.");

        // Left half: pure red on a plain background. Right half: pure blue. A strong, unambiguous color signal
        // is a more reliable spatial-distinctness check than two different subjects would be (subject shape
        // detection would need vision tooling this test doesn't have; color-channel dominance needs none).
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
            Steps = 4,
            CfgScale = 0f,
            Seed = 123,
        });

        string outPath = Path.Combine(RepoRoot.Path, "flux1_regional_output.rgb");
        File.WriteAllBytes(outPath, rgb);
        _output.WriteLine($"Wrote {outPath} ({rgb.Length} bytes, {width}x{height}) for visual inspection.");

        (double lR, double lG, double lB) = MeanRgb(rgb, width, height, 0, width / 2);
        (double rR, double rG, double rB) = MeanRgb(rgb, width, height, width / 2, width);
        _output.WriteLine($"Left half mean RGB:  ({lR:F1}, {lG:F1}, {lB:F1})");
        _output.WriteLine($"Right half mean RGB: ({rR:F1}, {rG:F1}, {rB:F1})");

        // "Spatially distinct" = the red region is actually redder than the blue region, and vice versa —
        // not just "the two halves differ somehow" (which a corrupted/noisy image would also satisfy).
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
            CheckpointPath = TestPaths.Flux.Schnell,
            Backend = backend,
        };
        using IRecipePipeline pipeline = new Flux1Recipe().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
