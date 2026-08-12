using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight test for Tier 3.1's PostApply hand-off (hires-fix / 2-pass upscale,
/// <c>RefinerUpscale != 1</c>): decode the base pass → resize (<c>TorchResize.BicubicAntialiasChw</c>) → VAE
/// re-encode through the SAME untiled encoder ordinary img2img already uses → redenoise the whole refiner
/// schedule with the official SDXL refiner UNet's CLIP-G-only/aesthetic-score conditioning
/// (<see cref="Diffusion.Pipelines.SdxlRefinerPipeline"/>, previously built with zero callers/zero test
/// coverage anywhere in the repo — this is its first exercise against a real checkpoint).
/// <para><b>Why untiled re-encode carries no new risk despite the tiled-VAE-encoder segfault
/// (ROADMAP.md)</b>: that crash is specifically the BF16 cuDNN conv fast path engaging on a
/// dtype-matched tile. SDXL's VAE was never in Tier 1.1's BF16-adoption list — it still runs F32 —
/// and this PostApply path uses the SAME untiled <c>VaeEncoder.Encode</c> call ordinary SDXL img2img
/// has used since Tier 0. Confirmed by reading the code before treating 3.1 as blocked (it wasn't —
/// only the TILED enhancement is; see the ROADMAP.md entry this test corrects).</para></summary>
[Trait("Category", "Integration")]
public sealed class SdxlPostApplyHiresFixRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public SdxlPostApplyHiresFixRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SdxlRecipe_PostApplyUpscale_ProducesLargerCoherentImage()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!File.Exists(TestPaths.Sdxl.SingleFile))
        {
            _output.WriteLine($"SKIPPED: SDXL checkpoint not found at {TestPaths.Sdxl.SingleFile}.");
            return;
        }
        if (!File.Exists(TestPaths.Sdxl.RefinerSingleFile))
        {
            _output.WriteLine($"SKIPPED: SDXL refiner checkpoint not found at {TestPaths.Sdxl.RefinerSingleFile}.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        const int baseWidth = 512, baseHeight = 512;
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        RecipeContext context = new RecipeContext { CheckpointPath = TestPaths.Sdxl.SingleFile, Backend = backend };
        using IRecipePipeline pipeline = new SdxlRecipe().Construct(context);

        // Base-only control, no refiner — confirms PostApply is additive, not required for a plain generation.
        ImageResult baseOnly = pipeline.Generate(
            new ImageRequest { Prompt = "a lighthouse on a rocky cliff at sunset, dramatic clouds", Width = baseWidth, Height = baseHeight, Steps = 16, CfgScale = 7.0f, Seed = 1234 },
            progress: null, cancel: default);
        _output.WriteLine($"Base only: {baseOnly.Width}x{baseOnly.Height}, {baseOnly.Rgb.Length} bytes, meta={string.Join(",", baseOnly.Meta.Keys)}.");
        Assert.Equal(baseWidth, baseOnly.Width);
        Assert.Equal(baseHeight, baseOnly.Height);
        Assert.False(baseOnly.Meta.ContainsKey("refiner_post_apply"));
        string baseOnlyPath = Path.Combine(RepoRoot.Path, "sdxl_postapply_base_only.rgb");
        File.WriteAllBytes(baseOnlyPath, baseOnly.Rgb);
        _output.WriteLine($"Wrote {baseOnlyPath} for visual inspection ({baseOnly.Width}x{baseOnly.Height}, RGB24 raw) — hires-fix control.");

        // 1.5x hires-fix through the OFFICIAL SDXL refiner checkpoint (the only architecture PostApply supports
        // in this slice — SdxlRefinerLoader hardcodes UNetConfig.SdxlRefiner's 4-level/CLIP-G-only shape; a
        // base-architecture checkpoint as Refiner.Model throws NotSupportedException instead, see
        // SdxlRefinerLoader.Load's own comment — that's a separate, unbuilt "same-checkpoint hires-fix" mechanism).
        ImageRequest hires = new ImageRequest
        {
            Prompt = "a lighthouse on a rocky cliff at sunset, dramatic clouds",
            Width = baseWidth,
            Height = baseHeight,
            Steps = 16,
            CfgScale = 7.0f,
            Seed = 1234,
            Refiner = new Refiner
            {
                Model = TestPaths.Sdxl.RefinerSingleFile,
                Control = 0.4,
                Upscale = 1.5,
                Steps = 16,
                CfgScale = 7.0,
            },
        };
        ImageResult hiresResult = pipeline.Generate(hires, progress: null, cancel: default);
        _output.WriteLine($"PostApply hires-fix: {hiresResult.Width}x{hiresResult.Height}, {hiresResult.Rgb.Length} bytes, meta={string.Join(",", hiresResult.Meta.Select(kv => $"{kv.Key}={kv.Value}"))}.");

        // Rounded-to-multiple-of-8: 512*1.5=768 exactly, no rounding surprise for this particular factor.
        Assert.Equal(768, hiresResult.Width);
        Assert.Equal(768, hiresResult.Height);
        Assert.Equal(768 * 768 * 3, hiresResult.Rgb.Length);
        Assert.True(hiresResult.Meta.TryGetValue("refiner_post_apply", out string? model) && model == TestPaths.Sdxl.RefinerSingleFile);

        string outPath = Path.Combine(RepoRoot.Path, "sdxl_postapply_hires.rgb");
        File.WriteAllBytes(outPath, hiresResult.Rgb);
        _output.WriteLine($"Wrote {outPath} for visual inspection ({hiresResult.Width}x{hiresResult.Height}, RGB24 raw).");

        // Sanity floor against a degenerate output (solid color / near-black from a broken VAE roundtrip or a
        // shape mismatch silently zero-filling) — real photographic content has real per-pixel variance.
        double mean = 0; for (int i = 0; i < hiresResult.Rgb.Length; i++) mean += hiresResult.Rgb[i];
        mean /= hiresResult.Rgb.Length;
        double variance = 0; for (int i = 0; i < hiresResult.Rgb.Length; i++) { double d = hiresResult.Rgb[i] - mean; variance += d * d; }
        variance /= hiresResult.Rgb.Length;
        _output.WriteLine($"Output mean={mean:F1}, stddev={Math.Sqrt(variance):F1}.");
        Assert.True(Math.Sqrt(variance) > 10.0, $"Hires-fix output has near-zero pixel variance (stddev {Math.Sqrt(variance):F2}) — looks degenerate, not a real photo.");

        // Non-exact upscale factor (1.5 above lands on an exact multiple of 8 and would never exercise the
        // rounding path) — 512*1.3=665.6, RoundToMultipleOf8 must round to the nearest 8, not truncate/ceil.
        // Reuses the already-loaded pipeline (base checkpoint load is the expensive part of this test).
        ImageRequest hiresOddFactor = hires with { Refiner = hires.Refiner! with { Upscale = 1.3 } };
        ImageResult oddResult = pipeline.Generate(hiresOddFactor, progress: null, cancel: default);
        _output.WriteLine($"PostApply hires-fix (1.3x): {oddResult.Width}x{oddResult.Height}.");
        Assert.Equal(664, oddResult.Width);
        Assert.Equal(664, oddResult.Height);
    }
}
