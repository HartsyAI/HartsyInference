using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 3.6: seamless/circular tiling. Wiring:
/// <c>ImageRequest.SeamlessTiling</c> (string, SwarmUI core's shared "false"/"true"/"X-Only"/"Y-Only"
/// vocabulary — deliberately NOT a new Hartsy-flagged duplicate param, since <c>SeamlessTileable</c> already
/// carries its own <c>"seamless"</c> FeatureFlag rather than a Comfy-only one) → <c>RecipeRequestMapper</c> →
/// <c>TextToImageRequest.SeamlessTiling</c> → <see cref="SdxlPipeline"/> sets
/// <see cref="Core.Backends.IBackend.SeamlessTilingX"/>/<see cref="Core.Backends.IBackend.SeamlessTilingY"/> on
/// both <c>Backend</c> and <c>VaeBackend</c> for the duration of the call (try/finally reset — these backends are
/// long-lived and shared across generations, so a leaked flag would silently corrupt the next request). The actual
/// wrap-pad happens once, centrally, inside <see cref="CudaBackend.Conv2D"/> — every conv on the hot path (UNet
/// AND VAE, both route through that one method) gets the wrapped-edge input instead of zero-padding when the flag
/// is set, so no per-callsite plumbing was needed anywhere in <c>UNet</c>/<c>ResNetBlock2D</c>/etc.
/// <para><b>Verification metric:</b> seamless tiling doesn't change how "correct" a single image looks — it
/// changes whether its opposite edges are continuous. So this test doesn't do a mean-abs-diff comparison at all;
/// it measures edge discontinuity directly (mean abs diff between column 0 and the last column, and between row 0
/// and the last row) for the same seed/prompt with the flag off vs. on, and asserts the flag actually reduces it.
/// The 2x2 tiled mosaic written to disk is the real verification — a bad wrap shows as a visible cross through
/// the middle of the tiled image; a working one reads as one continuous texture.</para></summary>
[Trait("Category", "Integration")]
public sealed class SdxlSeamlessTilingRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public SdxlSeamlessTilingRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SdxlRecipe_SeamlessTilingEnabled_EdgesContinueAcrossTheWrap()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!File.Exists(TestPaths.Sdxl.SingleFile))
        {
            _output.WriteLine($"SKIPPED: SDXL checkpoint not found at {TestPaths.Sdxl.SingleFile}.");
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

        const int width = 512, height = 512;
        // A pattern-heavy prompt makes a broken seam obvious on inspection (a plain sky/gradient prompt would
        // hide a seam that's still numerically there).
        ImageRequest baseRequest = new ImageRequest
        {
            Prompt = "seamless repeating pattern of red brick wall texture, tileable, photographic",
            Width = width,
            Height = height,
            Steps = 20,
            CfgScale = 6.0f,
            Seed = 4242,
        };
        ImageRequest withTiling = baseRequest with { SeamlessTiling = "true" };

        Assert.True(new SdxlRecipe().Supports.HasFlag(ImageFeatures.SeamlessTiling), "SdxlRecipe should declare ImageFeatures.SeamlessTiling.");

        byte[] off = Generate(ptxDir, baseRequest);
        _output.WriteLine($"Generated seamless=off ({off.Length} bytes).");
        byte[] on = Generate(ptxDir, withTiling);
        _output.WriteLine($"Generated seamless=true ({on.Length} bytes).");
        Assert.Equal(off.Length, on.Length);

        File.WriteAllBytes(Path.Combine(RepoRoot.Path, "sdxl_seamless_off.rgb"), off);
        File.WriteAllBytes(Path.Combine(RepoRoot.Path, "sdxl_seamless_on.rgb"), on);

        // 2x2 tiled mosaic of the seamless=on image — the actual visual-inspection artifact. A broken wrap shows
        // as a visible cross through the middle.
        byte[] tiled = TileTwoByTwo(on, width, height);
        string tiledPath = Path.Combine(RepoRoot.Path, "sdxl_seamless_on_2x2.rgb");
        File.WriteAllBytes(tiledPath, tiled);
        _output.WriteLine($"Wrote {tiledPath} ({width * 2}x{height * 2}) — 2x2 tiled mosaic for visual seam inspection.");

        double offEdgeDiff = EdgeDiscontinuity(off, width, height);
        double onEdgeDiff = EdgeDiscontinuity(on, width, height);
        _output.WriteLine($"Edge discontinuity (mean abs diff, col0-vs-colLast + row0-vs-rowLast): off={offEdgeDiff:F2}, on={onEdgeDiff:F2}.");

        Assert.True(onEdgeDiff < offEdgeDiff * 0.6,
            $"seamless=true did not meaningfully reduce edge discontinuity (off={offEdgeDiff:F2}, on={onEdgeDiff:F2}) — the wrap-pad may not be reaching Conv2D.");
    }

    /// <summary>Mean abs diff between column 0 and the last column, and between row 0 and the last row — the
    /// numeric proxy for "does this image tile continuously". Lower = more continuous.</summary>
    private static double EdgeDiscontinuity(byte[] rgb, int width, int height)
    {
        long sum = 0;
        long count = 0;
        for (int y = 0; y < height; y++)
        {
            int left = (y * width + 0) * 3;
            int right = (y * width + (width - 1)) * 3;
            for (int c = 0; c < 3; c++) { sum += Math.Abs(rgb[left + c] - rgb[right + c]); count++; }
        }
        for (int x = 0; x < width; x++)
        {
            int top = (0 * width + x) * 3;
            int bottom = ((height - 1) * width + x) * 3;
            for (int c = 0; c < 3; c++) { sum += Math.Abs(rgb[top + c] - rgb[bottom + c]); count++; }
        }
        return sum / (double)count;
    }

    /// <summary>Lays four copies of the same image edge-to-edge in a 2x2 grid — the seam between adjacent copies
    /// is exactly the wrap boundary being tested.</summary>
    private static byte[] TileTwoByTwo(byte[] rgb, int width, int height)
    {
        int outW = width * 2, outH = height * 2;
        byte[] outBuf = new byte[outW * outH * 3];
        for (int ty = 0; ty < 2; ty++)
        {
            for (int tx = 0; tx < 2; tx++)
            {
                for (int y = 0; y < height; y++)
                {
                    int srcOffset = (y * width) * 3;
                    int dstOffset = ((ty * height + y) * outW + tx * width) * 3;
                    Array.Copy(rgb, srcOffset, outBuf, dstOffset, width * 3);
                }
            }
        }
        return outBuf;
    }

    private static byte[] Generate(string ptxDir, ImageRequest request)
    {
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        RecipeContext context = new RecipeContext
        {
            CheckpointPath = TestPaths.Sdxl.SingleFile,
            Backend = backend,
        };
        using IRecipePipeline pipeline = new SdxlRecipe().Construct(context);
        ImageResult result = pipeline.Generate(request, progress: null, cancel: default);
        return result.Rgb;
    }
}
