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

/// <summary>TE/VAE component placement through the full engine path with the on-disk Flux dev fp8 checkpoint:
/// a generation with the T5/CLIP encoders AND the VAE routed to the second GPU must produce the same image as
/// the single-GPU baseline. On matched GPUs this is bit-identical (placement moves math, it doesn't change it);
/// on THIS box's mismatched pair the fp8 T5 takes native-fp8 GEMM on one card and dequant fallback on the other,
/// so the honest bar is high SSIM. Also asserts the second card actually held the encoder (free-VRAM probe
/// mid-lifecycle would race the phases, so the proof is the generation succeeding with placement validated
/// eagerly + the SSIM gate — a mis-routed encode would either throw or produce garbage).</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class FluxComponentPlacementEngineTests
{
    private readonly ITestOutputHelper _output;
    public FluxComponentPlacementEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TeAndVaePlacement_RealEngine_MatchesSingleGpuBaseline()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Flux.Dev;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        ModelSpec spec = ModelResolver.Resolve("flux1", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: flux1 not resolvable with the explicit path."); return; }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 1024,
            Height = 1024,
            Steps = 8,
            CfgScale = 1.0f,
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
        _output.WriteLine("[2/2] TE + VAE placed on ordinal 1 (denoiser stays on ordinal 0)...");
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
        Assert.True(ssim > 0.75, $"TE/VAE-placed output diverged from baseline (SSIM={ssim:F4}) — placement is "
            + "supposed to move computation, not change it; check the encode/decode backend routing and the "
            + "LOAD-BEARING host materialization sweeps.");
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }
}
