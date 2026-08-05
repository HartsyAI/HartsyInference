using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan CFG-branch parallelism through the full engine path on real TI2V-5B weights — the
/// previously-VRAM-blocked verification. EITHER outcome is a pass, and which one ran is printed and asserted
/// observable: (a) the replica fits the second card → "[CfgParallel] active" and the output matches the
/// sequential baseline (SSIM — the two branches run on different-SM cards, so bit-parity isn't the bar); or
/// (b) it doesn't fit → "[CfgParallel] fell-back(preload-failed...)" and the generation still completes
/// (the OOM→sequential fallback added 2026-08-04; before that a too-small second card killed the generation).
/// On this box TI2V-5B fp16 (~10 GB replicated) vs the 3060's ~11.7 GB free makes the ACTIVE branch genuinely
/// reachable but tight — the test adapts to what the hardware does rather than assuming.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanCfgParallelEngineTests
{
    private readonly ITestOutputHelper _output;
    public WanCfgParallelEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task CfgParallel_RealEngine_EitherOutcome_ObservableAndCorrect()
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
        _output.WriteLine("[1/2] Sequential CFG baseline (single GPU)...");
        VideoGenerationResult baseline;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            baseline = await engine.Video.GenerateAsync(spec, request);
        }
        double baselineSeconds = sw.Elapsed.TotalSeconds;
        _output.WriteLine($"  baseline: {baseline.Frames.Count} frames, {baselineSeconds:F1}s");

        List<string> captured = new();
        Logs.SetLogger((level, message) =>
        {
            lock (captured) captured.Add(message);
            Console.Error.WriteLine($"[{level}] {message}");
        });
        try
        {
            sw.Restart();
            _output.WriteLine("[2/2] CFG-parallel (uncond branch on ordinal 1)...");
            PlacementConfig placement = new PlacementConfig { CfgParallelDevice = "cuda:1" };
            VideoGenerationResult parallel;
            using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
            {
                parallel = await engine.Video.GenerateAsync(spec, request);
            }
            double parallelSeconds = sw.Elapsed.TotalSeconds;
            _output.WriteLine($"  cfg-parallel run: {parallel.Frames.Count} frames, {parallelSeconds:F1}s (baseline {baselineSeconds:F1}s)");

            string[] decisions;
            lock (captured) decisions = captured.Where(m => m.Contains("[CfgParallel]")).ToArray();
            _output.WriteLine($"CfgParallel decisions observed: {string.Join(" | ", decisions)}");
            Assert.True(decisions.Length > 0, "no [CfgParallel] decision was logged — the configured second GPU's " +
                "path must always be observable, active or not.");

            Assert.Equal(baseline.Frames.Count, parallel.Frames.Count);
            VideoFrame b0 = baseline.Frames[0];
            VideoFrame p0 = parallel.Frames[0];
            double ssim = Ssim.Compute(b0.Rgb, p0.Rgb, b0.Width, b0.Height);
            bool active = decisions.Any(d => d.Contains("active"));
            _output.WriteLine($"OUTCOME: {(active ? "ACTIVE (concurrent branches)" : "FALLBACK (sequential)")}; first-frame SSIM = {ssim:F4}");
            // Active: cross-SM branch execution → SSIM bar. Fallback: same math as baseline on the same card →
            // near-identical, but the same SSIM bar keeps the assertion uniform.
            Assert.True(ssim > 0.75, $"cfg-parallel run diverged from sequential baseline (SSIM={ssim:F4}).");
        }
        finally
        {
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
    }
}
