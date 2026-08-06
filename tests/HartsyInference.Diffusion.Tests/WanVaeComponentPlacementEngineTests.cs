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

/// <summary>Wan VAE placement through the full engine path with the real TI2V-5B checkpoint: a short T2V generation
/// with the Wan2.2 VAE (encode+decode) on the second GPU must match the single-GPU baseline. This is the twin of
/// <see cref="WanComponentPlacementEngineTests"/> (TE placement) — Wan had zero VAE placement before
/// <c>WanVideoPipeline.VaeBackend</c> was wired (see the class doc on <c>WanVideoPipeline</c> and
/// <c>docs/MULTI_GPU.md</c>'s TE/VAE placement row). Same SSIM rationale as the TE twin: this box's mismatched SM
/// pair (3060/4090) legitimately takes different conv/GEMM paths for the VAE decode, so first-frame SSIM is the bar,
/// not bit-exactness.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanVaeComponentPlacementEngineTests
{
    private readonly ITestOutputHelper _output;
    public WanVaeComponentPlacementEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task VaePlacement_RealEngine_MatchesSingleGpuBaseline()
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
        _output.WriteLine("[2/2] Wan2.2 VAE placed on ordinal 1 (DiT/umT5 stay on ordinal 0)...");
        PlacementConfig placement = new PlacementConfig { VaeDevice = "cuda:1" };
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
        _output.WriteLine($"first-frame SSIM(baseline, VAE placed) = {ssim:F4}");
        Assert.True(ssim > 0.75, $"VAE-placed Wan output diverged from baseline (SSIM={ssim:F4}) — check the "
            + "Wan22VaeDecoder backend routing and the LOAD-BEARING host-resident-latent invariant in WanVideoPipeline.");
    }
}
