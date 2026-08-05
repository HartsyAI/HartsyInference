using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>TE/VAE component placement through the full engine path with the on-disk SDXL base checkpoint —
/// the same contract as <see cref="FluxComponentPlacementEngineTests"/> on the smaller, faster model (F16
/// weights, so cross-card numeric drift is smaller than the fp8 Flux case; the SSIM bar reflects an
/// effectively-identical image).</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class SdxlComponentPlacementEngineTests
{
    private readonly ITestOutputHelper _output;
    public SdxlComponentPlacementEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TeAndVaePlacement_RealEngine_MatchesSingleGpuBaseline()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Sdxl.SingleFile;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        ModelSpec spec = ModelResolver.Resolve("sdxl", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: sdxl not resolvable with the explicit path."); return; }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 1024,
            Height = 1024,
            Steps = 12,
            CfgScale = 6.0f,
            Seed = 42,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/2] Baseline (everything on ordinal 0)...");
        ImageResult baseline;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            baseline = await engine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(baseline.Rgb, "baseline");

        sw.Restart();
        _output.WriteLine("[2/2] TE + VAE placed on ordinal 1 (UNet stays on ordinal 0)...");
        PlacementConfig placement = new PlacementConfig { TextEncoderDevice = "cuda:1", VaeDevice = "cuda:1" };
        ImageResult placed;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            placed = await engine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  placed: {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(placed.Rgb, "placed");

        double ssim = Ssim.Compute(baseline.Rgb, placed.Rgb, placed.Width, placed.Height);
        _output.WriteLine($"SSIM(baseline, TE+VAE placed) = {ssim:F4}");
        Assert.True(ssim > 0.9, $"TE/VAE-placed SDXL output diverged from baseline (SSIM={ssim:F4}).");
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }
}
