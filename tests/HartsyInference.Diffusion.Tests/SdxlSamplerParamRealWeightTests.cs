using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for a bug found 2026-08-10 while adding "tcd" to the extension's
/// "HartsyInference Sampler" dropdown: <see cref="SdxlRecipePipeline.BuildInner"/> (and the SD1.5 sibling) built
/// the inner <c>TextToImageRequest</c> from <c>request.Scheduler</c>, but the extension ALWAYS sends that field
/// null ("the Engine resolves the family's canonical schedule") — the actual sampler choice lives on
/// <c>request.Sampler</c>, which was never read. The "HartsyInference Sampler" param (euler/ddim/dpm++2m/lcm/tcd)
/// has therefore been silently inert for SDXL and SD1.5 since it was built: every generation ran Euler
/// regardless of what was selected. Fixed by reading <c>request.Sampler ?? request.Scheduler</c>. Verified by
/// requesting "dpm++2m" (a scheduler with materially different step behavior than Euler) and confirming the
/// output differs from the default — before the fix this would have been byte-identical (both silently ran
/// Euler).</summary>
[Trait("Category", "Integration")]
public sealed class SdxlSamplerParamRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public SdxlSamplerParamRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SdxlRecipe_WithDpmPlusPlus2mSampler_ProducesVisiblyDifferentOutputThanEulerDefault()
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

        ImageRequest baseRequest = new ImageRequest
        {
            Prompt = "a photo of an old lighthouse on a cliff, stormy sky",
            Width = 512,
            Height = 512,
            Steps = 12,
            CfgScale = 6.0f,
            Seed = 314159,
        };
        ImageRequest withDpm = baseRequest with { Sampler = "dpm++2m" };

        byte[] euler = Generate(ptxDir, baseRequest);
        _output.WriteLine($"Generated with default (Euler) sampler ({euler.Length} bytes).");
        byte[] dpm = Generate(ptxDir, withDpm);
        _output.WriteLine($"Generated with Sampler=\"dpm++2m\" ({dpm.Length} bytes).");

        Assert.Equal(euler.Length, dpm.Length);
        long diffSum = 0;
        for (int i = 0; i < euler.Length; i++) diffSum += Math.Abs(euler[i] - dpm[i]);
        double meanAbsDiff = diffSum / (double)euler.Length;
        _output.WriteLine($"Mean absolute per-byte difference: {meanAbsDiff:F2} (0 would mean byte-identical, i.e. Sampler had no effect — the exact bug this test catches).");
        Assert.True(meanAbsDiff > 1.0, $"Sampler=\"dpm++2m\" vs. default output are nearly identical (mean abs diff {meanAbsDiff:F2}) — the Sampler param likely isn't reaching SchedulerFactory again.");
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
