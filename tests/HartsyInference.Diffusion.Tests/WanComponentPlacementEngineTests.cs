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

/// <summary>Wan TE placement through the full engine path with the real TI2V-5B checkpoint: a short generation
/// with the umT5 (T5-XXL-class, the biggest single TE win) on the second GPU must match the single-GPU baseline.
/// The umT5 is fp8-scaled, so on this box's mismatched SM pair the encode legitimately drifts (native fp8 GEMM vs
/// dequant fallback) — first-frame SSIM is the bar, same rationale as the Flux placement twin.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanComponentPlacementEngineTests
{
    private readonly ITestOutputHelper _output;
    public WanComponentPlacementEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TePlacement_RealEngine_MatchesSingleGpuBaseline()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl)) return;

        ModelSpec spec = ModelResolver.Resolve("wan", TestPaths.WanVideo.Ti2V5B, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: wan not resolvable with the explicit path."); return; }

        VideoRequest request = new VideoRequest
        {
            Prompt = "a red ball bouncing on a wooden table",
            Width = 480,
            Height = 480,
            Frames = 9,
            Steps = 6,
            CfgScale = 5.0f,
            Seed = 42,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/2] Baseline (everything on ordinal 0)...");
        VideoGenerationResult baseline;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            baseline = await engine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {baseline.Frames.Count} frames, {sw.Elapsed.TotalSeconds:F1}s");

        sw.Restart();
        _output.WriteLine("[2/2] umT5 placed on ordinal 1 (DiT/VAE stay on ordinal 0)...");
        PlacementConfig placement = new PlacementConfig { TextEncoderDevice = "cuda:1" };
        VideoGenerationResult placed;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            placed = await engine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  placed: {placed.Frames.Count} frames, {sw.Elapsed.TotalSeconds:F1}s");

        Assert.Equal(baseline.Frames.Count, placed.Frames.Count);
        VideoFrame b0 = baseline.Frames[0];
        VideoFrame p0 = placed.Frames[0];
        double ssim = Ssim.Compute(b0.Rgb, p0.Rgb, b0.Width, b0.Height);
        _output.WriteLine($"first-frame SSIM(baseline, TE placed) = {ssim:F4}");
        Assert.True(ssim > 0.75, $"TE-placed Wan output diverged from baseline (SSIM={ssim:F4}) — check the umT5 " +
            "encode routing and the load-bearing host materialization in WanVideoRecipePipeline.");
    }
}
