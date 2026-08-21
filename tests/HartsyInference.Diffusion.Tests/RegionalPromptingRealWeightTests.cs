using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression for Tier 1.4/3.7: <see cref="HartsyInference.Diffusion.Prompting.RegionalPlan"/>
/// wired into each architecture's forward pass, verified by whether a two-region prompt actually lands SPATIALLY
/// DISTINCT content — not merely "the two halves differ somehow" (a corrupted/noisy image would also satisfy
/// that), but checked by comparing each half's mean color channel against the other half's. One shared harness
/// parameterized per model instead of a near-identical file per model (was: Flux.1/Flux.2/Ideogram 4/Krea
/// 2/Z-Image, 577 lines combined, differing only in the recipe type, checkpoint, request shape, and — for
/// Ideogram 4 alone — the prompt syntax (structured JSON; a bare prompt trips its safety filter). Each model's
/// wiring is architecturally distinct enough to be worth its own real-weight proof (GQA vs non-GQA bias
/// broadcast, single-tap vs multi-layer-interleaved region encode, whether the recipe pipeline owns the text
/// encoder directly or has to reach into the base pipeline for it) — see git history on the individual classes
/// this replaced for those per-architecture notes — but the actual generate-and-check-spatial-distinctness logic
/// was identical five times over.</summary>
[Trait("Category", "Integration")]
public sealed class RegionalPromptingRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public RegionalPromptingRealWeightTests(ITestOutputHelper output) => _output = output;

    private const string ColorHalvesPrompt = "a simple flat-color studio background, minimalist "
        + "<region:0,0,0.5,1>solid bright red color, pure red, #ff0000"
        + "<region:0.5,0,0.5,1>solid bright blue color, pure blue, #0000ff"
        + "<region:end>";

    // Structured JSON caption per Ideogram4RecipePipeline's own warning: a bare prompt trips its safety filter.
    private const string Ideogram4ColorHalvesPrompt = "{\"style\": \"minimalist studio background\"} "
        + "<region:0,0,0.5,1>{\"subject\": \"solid bright red color, pure red, #ff0000\"}"
        + "<region:0.5,0,0.5,1>{\"subject\": \"solid bright blue color, pure blue, #0000ff\"}"
        + "<region:end>";

    public sealed record Case(
        string Name, string CheckpointPath, Func<IArchitectureRecipe> RecipeFactory,
        string Prompt, int Steps, float? Cfg, int Seed);

    public static TheoryData<Case> Cases() => new()
    {
        new Case("Flux1", TestPaths.Flux.Schnell, () => new Flux1Recipe(), ColorHalvesPrompt, 4, 0f, 123),
        // No fp8/safetensors Flux.2 Dev build on this box — the local checkpoint is the Q4_K_S GGUF repack, which
        // Flux2Recipe already detects and loads (isGguf branch).
        new Case("Flux2", "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Flux2/flux2-dev-Q4_K_S.gguf",
            () => new Flux2Recipe(), ColorHalvesPrompt, 8, 0f, 123),
        new Case("Ideogram4", Path.Combine(TestPaths.Ideogram4.Dir, "ideogram4_fp8_scaled.safetensors"),
            () => new Ideogram4Recipe(), Ideogram4ColorHalvesPrompt, 12, null, 123),
        // Krea2Recipe.Construct reads Path.GetFileName(context.CheckpointPath) directly (a single file, not the
        // Base/Turbo bundle layout TestPaths.Krea2 describes), so hardcode the real path.
        new Case("Krea2", "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Krea2/krea2_raw_fp8_scaled.safetensors",
            () => new Krea2Recipe(), ColorHalvesPrompt, 8, 0f, 123),
        new Case("ZImage", TestPaths.ZImage.Turbo, () => new ZImageRecipe(), ColorHalvesPrompt, 8, 1.0f, 123),
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Recipe_TwoRegionPrompt_LandsSpatiallyDistinct(Case c)
    {
        if (!File.Exists(c.CheckpointPath))
        {
            _output.WriteLine($"SKIPPED: {c.Name} checkpoint not found at {c.CheckpointPath}.");
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

        IArchitectureRecipe recipe = c.RecipeFactory();
        Assert.True(recipe.Supports.HasFlag(ImageFeatures.Regional), $"{recipe.GetType().Name} should declare ImageFeatures.Regional.");

        const int width = 512, height = 512;
        byte[] rgb = Generate(ptxDir, c, new ImageRequest
        {
            Prompt = c.Prompt,
            Width = width,
            Height = height,
            Steps = c.Steps,
            CfgScale = c.Cfg,
            Seed = c.Seed,
        });

        string outPath = Path.Combine(RepoRoot.Path, $"{c.Name.ToLowerInvariant()}_regional_output.rgb");
        File.WriteAllBytes(outPath, rgb);
        _output.WriteLine($"[{c.Name}] Wrote {outPath} ({rgb.Length} bytes, {width}x{height}) for visual inspection.");

        (double lR, double lG, double lB) = MeanRgb(rgb, width, height, 0, width / 2);
        (double rR, double rG, double rB) = MeanRgb(rgb, width, height, width / 2, width);
        _output.WriteLine($"[{c.Name}] Left half mean RGB:  ({lR:F1}, {lG:F1}, {lB:F1})");
        _output.WriteLine($"[{c.Name}] Right half mean RGB: ({rR:F1}, {rG:F1}, {rB:F1})");

        // "Spatially distinct" = the red region is actually redder than the blue region, and vice versa — not
        // just "the two halves differ somehow" (which a corrupted/noisy image would also satisfy).
        double leftRedDominance = lR - lB;
        double rightBlueDominance = rB - rR;
        _output.WriteLine($"[{c.Name}] Left red-dominance (R-B): {leftRedDominance:F1}; Right blue-dominance (B-R): {rightBlueDominance:F1}");
        Assert.True(leftRedDominance > 10.0, $"[{c.Name}] Left half is not red-dominant (R-B = {leftRedDominance:F1}) — regional conditioning may not be landing spatially.");
        Assert.True(rightBlueDominance > 10.0, $"[{c.Name}] Right half is not blue-dominant (B-R = {rightBlueDominance:F1}) — regional conditioning may not be landing spatially.");
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

    private static byte[] Generate(string ptxDir, Case c, ImageRequest request)
    {
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        RecipeContext context = new RecipeContext { CheckpointPath = c.CheckpointPath, Backend = backend };
        using IRecipePipeline pipeline = c.RecipeFactory().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
