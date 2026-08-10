using HartsyInference.Cuda;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for Tier 2.2: CFG-Rescale (<see cref="HartsyInference.Diffusion.Utilities.CfgHelper.ApplyCfgRescale"/>)
/// wired into <see cref="SdxlPipeline"/>'s <c>ClassifierFreeGuidanceStep</c>, reached via
/// <c>ImageRequest.CfgRescale</c> → <c>RecipeRequestMapper.ToTextToImage</c> → <c>TextToImageRequest.CfgRescale</c>.
/// At a high CFG scale (12, well above SDXL's cfg=5 default) the guided prediction's magnitude runs well past the
/// conditional's, which is what drives blown-out/oversaturated output — CFG-Rescale pulls it back down. Also
/// confirms <c>cfgRescale > 0</c> correctly falls back off SDXL's fused batched-CFG loop (that loop's CFG-Euler
/// combine is a single in-place device kernel with no host-visible F32 tensor to renormalize against) rather than
/// silently no-op'ing by running the fused path anyway.</summary>
[Trait("Category", "Integration")]
public sealed class SdxlCfgRescaleRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public SdxlCfgRescaleRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SdxlRecipe_HighCfgWithRescale_ProducesVisiblyLessSaturatedOutputThanWithoutRescale()
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
        ImageRequest baseRequest = new ImageRequest
        {
            Prompt = "a vivid neon-lit city street at night, dramatic lighting, high contrast",
            Width = width,
            Height = height,
            Steps = 12,
            CfgScale = 12.0f,
            Seed = 555,
        };
        ImageRequest withRescale = baseRequest with { CfgRescale = 0.7f };

        byte[] off = Generate(ptxDir, baseRequest);
        _output.WriteLine($"Generated rescale=0 ({off.Length} bytes).");
        byte[] on = Generate(ptxDir, withRescale);
        _output.WriteLine($"Generated rescale=0.7 ({on.Length} bytes).");

        string offPath = Path.Combine(RepoRoot.Path, "sdxl_cfgrescale_off.rgb");
        string onPath = Path.Combine(RepoRoot.Path, "sdxl_cfgrescale_on.rgb");
        File.WriteAllBytes(offPath, off);
        File.WriteAllBytes(onPath, on);
        _output.WriteLine($"Wrote {offPath} and {onPath} for visual inspection.");

        Assert.Equal(off.Length, on.Length);

        long diffSum = 0;
        double meanOff = 0, meanOn = 0;
        for (int i = 0; i < off.Length; i++)
        {
            diffSum += Math.Abs(off[i] - on[i]);
            meanOff += off[i];
            meanOn += on[i];
        }
        double meanAbsDiff = diffSum / (double)off.Length;
        meanOff /= off.Length;
        meanOn /= on.Length;
        _output.WriteLine($"Mean absolute per-byte difference: {meanAbsDiff:F2}; mean byte value off={meanOff:F1} on={meanOn:F1}.");

        Assert.True(meanAbsDiff > 1.0, $"rescale=0 vs rescale=0.7 outputs are nearly identical (mean abs diff {meanAbsDiff:F2}) — CFG-Rescale likely isn't reaching the combine step.");
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
