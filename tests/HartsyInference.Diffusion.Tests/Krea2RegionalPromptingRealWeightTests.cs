using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 3.7 (Krea 2 half): <see cref="Krea2Attention"/> gained an
/// <c>attnBias</c> slot, threaded through <see cref="Krea2Block"/> and <see cref="Krea2Transformer"/>'s eager
/// forward path (the graph route and DiT-sharded route are both excluded when a regional plan is active, mirroring
/// FluxTransformer's cache/graph exclusions). The two risks this architecture specifically raised over Flux.1/
/// Flux.2 (confirmed by reading the code before wiring, not assumed by analogy):
/// <list type="bullet">
/// <item>Token order: Krea 2's <c>ForwardEmbedIn</c> concats <c>[txt, img]</c> — same order as Flux, so
/// <c>RegionalAttentionBias.Build</c>'s (totalLen, txtLen, imgLen) signature applies unchanged.</item>
/// <item>GQA: <c>Krea2Attention</c> repeats K/V to the full head count (<c>RepeatKvHeads</c>) BEFORE the SDPA
/// call, so the bias broadcasts against the post-repeat head count exactly like a non-GQA architecture — no
/// GQA-aware bias construction needed.</item>
/// </list>
/// <para>Verification per the same bar Tier 1.4/Flux.2 used: a two-region prompt, per-half color-channel
/// dominance.</para></summary>
[Trait("Category", "Integration")]
public sealed class Krea2RegionalPromptingRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public Krea2RegionalPromptingRealWeightTests(ITestOutputHelper output) => _output = output;

    // TestPaths.Krea2.BaseDir points at a directory with only text_encoder/ + vae/ subdirs on this box — the
    // actual transformer file sits directly under Krea2/, not nested in Base/. Krea2Recipe.Construct reads
    // Path.GetFileName(context.CheckpointPath) directly (a single file, not the Base/Turbo bundle layout the
    // TestPaths doc comment describes), so hardcode the real path rather than the mismatched TestPaths helper —
    // same precedent as FluxStepCacheRealWeightTests/ChromaStepCacheRealWeightTests this session.
    private const string CheckpointPath = "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Krea2/krea2_raw_fp8_scaled.safetensors";

    [Fact]
    public void Krea2Recipe_TwoRegionPrompt_LandsSpatiallyDistinct()
    {
        if (!File.Exists(CheckpointPath))
        {
            _output.WriteLine($"SKIPPED: Krea 2 checkpoint not found at {CheckpointPath}.");
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

        Assert.True(new Krea2Recipe().Supports.HasFlag(ImageFeatures.Regional), "Krea2Recipe should now declare ImageFeatures.Regional.");

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
            CfgScale = 0f,
            Seed = 123,
        });

        string outPath = Path.Combine(RepoRoot.Path, "krea2_regional_output.rgb");
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
            CheckpointPath = CheckpointPath,
            Backend = backend,
        };
        using IRecipePipeline pipeline = new Krea2Recipe().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
