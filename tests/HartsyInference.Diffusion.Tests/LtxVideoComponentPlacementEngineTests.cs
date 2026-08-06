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
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Diffusion.Tests;

/// <summary>LTX-Video (LTX-1) TE+VAE placement through the full engine path with the on-disk 0.9 2B single-file
/// checkpoint: a generation with the T5-XXL prompt encoder AND the bundled VAE routed to the second GPU must
/// reproduce the single-GPU baseline. LTX-1 had TE placement already (<c>LtxVideoRecipePipeline</c>'s
/// <c>_textBackend</c>, wired the same way as Wan's) but zero VAE placement before <c>LtxVideoPipeline.VaeBackend</c>
/// was wired — this is that pair's first real-weight verification, matching the shape of
/// <see cref="FluxComponentPlacementEngineTests"/> (which also places both halves in one test).</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LtxVideoComponentPlacementEngineTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideoComponentPlacementEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TeAndVaePlacement_RealEngine_MatchesSingleGpuBaseline()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.LtxVideo.SingleFile)) return;

        ModelSpec spec = ModelResolver.Resolve("ltx-video", TestPaths.LtxVideo.SingleFile, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: ltx-video not resolvable with the explicit path."); return; }

        VideoRequest request = new VideoRequest
        {
            Prompt = "a cinematic shot of a cat walking through a sunlit garden",
            NegativePrompt = "blurry, low quality, distorted, watermark",
            Width = 512,
            Height = 320,
            Frames = 17,
            Fps = 24,
            Steps = 6,
            CfgScale = 3.0f,
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
        _output.WriteLine("[2/2] T5-XXL + VAE placed on ordinal 1 (DiT stays on ordinal 0)...");
        PlacementConfig placement = new PlacementConfig { TextEncoderDevice = "cuda:1", VaeDevice = "cuda:1" };
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
        _output.WriteLine($"first-frame SSIM(baseline, TE+VAE placed) = {ssim:F4}");
        Assert.True(ssim > 0.75, $"TE/VAE-placed LTX-Video output diverged from baseline (SSIM={ssim:F4}) — check "
            + "the T5/VAE backend routing and the LOAD-BEARING host materialization in LtxVideoPipeline/LtxVideoRecipePipeline.");
    }
}
