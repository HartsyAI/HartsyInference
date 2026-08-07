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

/// <summary>Wan context parallelism (Phase 2, 2026-08-06) through the full engine path on real TI2V-5B weights —
/// the WanCfgParallelEngineTests geometry. EITHER outcome is a pass and observable: (a) the replica fits rank 1 →
/// "[ContextParallel] active(...)" and the output matches the single-GPU baseline; or (b) it doesn't fit →
/// "[ContextParallel] fell-back(preload-failed...)" and the generation still completes.
/// <para><b>Why the active-path SSIM bar is 0.90, not 0.99</b> (measured 2026-08-06): the CP mechanism itself is
/// byte-exact (the synthetic <c>ContextParallelWanTests</c> match a single-backend forward with max |diff| = 0),
/// but an ACTIVE run computes ~⅓ of tokens on the 3060 (SM 8.6), and cross-ARCHITECTURE kernel selection /
/// reduction order drifts on this fp16 checkpoint at this few-step geometry. Controls, same request/seed:
/// whole model 4090-vs-3060 with NO CP code = SSIM 0.7774 (<c>WanCrossGpuRegimeDiagnosticTests</c>, the drift
/// ceiling); CFG-parallel (whole uncond branch on the 3060, accepted at a 0.75 bar) = 0.8766; CP measured
/// 0.9616 — best of the three mixes, exactly ordered by the fraction of 3060-computed work. A matched-regime
/// run (HP F32 GEMMs + plain SDPA both cards) does NOT close the gap (0.8907) — regime flags cannot equalize
/// different GPU architectures. 0.90 catches a real CP regression (boundary/exchange bugs crater SSIM far
/// below the drift band) while not failing on physics this box cannot avoid.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanContextParallelEngineTests
{
    private readonly ITestOutputHelper _output;
    public WanContextParallelEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task ContextParallel_RealEngine_EitherOutcome_ObservableAndCorrect()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl)) return;
        // TI2V-5B fp16 is ~10 GB REPLICATED per rank; gate on both cards' free VRAM so a busy box skips cleanly.
        IReadOnlyList<GpuTopologyInfo> topo0 = CudaTopology.Probe();
        foreach (GpuTopologyInfo gpu in topo0)
        {
            _output.WriteLine($"GPU {gpu.Ordinal} ({gpu.Name}): {gpu.FreeMemoryBytes >> 20} MiB free");
            if (gpu.FreeMemoryBytes < 11L << 30)
            {
                _output.WriteLine($"SKIPPED: GPU {gpu.Ordinal} has < 11 GiB free.");
                return;
            }
        }

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
        _output.WriteLine("[1/2] Single-GPU baseline...");
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
            IReadOnlyList<GpuTopologyInfo> preCp = CudaTopology.Probe();
            sw.Restart();
            _output.WriteLine("[2/2] Context-parallel (sequence split across cuda:0 + cuda:1)...");
            PlacementConfig placement = new PlacementConfig { ContextParallelDevices = ["cuda:0", "cuda:1"] };
            VideoGenerationResult parallel;
            IReadOnlyList<GpuTopologyInfo> duringCp;
            using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
            {
                parallel = await engine.Video.GenerateAsync(spec, request);
                duringCp = CudaTopology.Probe();   // probed while both ranks' residents are still held
            }
            double parallelSeconds = sw.Elapsed.TotalSeconds;
            _output.WriteLine($"  context-parallel run: {parallel.Frames.Count} frames, {parallelSeconds:F1}s (baseline {baselineSeconds:F1}s)");
            for (int i = 0; i < duringCp.Count && i < preCp.Count; i++)
            {
                long riseMib = (preCp[i].FreeMemoryBytes - duringCp[i].FreeMemoryBytes) >> 20;
                _output.WriteLine($"  VRAM rise GPU {duringCp[i].Ordinal} ({duringCp[i].Name}): {riseMib} MiB");
            }

            string[] decisions;
            lock (captured) decisions = captured.Where(m => m.Contains("[ContextParallel]")).ToArray();
            _output.WriteLine($"ContextParallel decisions observed: {string.Join(" | ", decisions)}");
            Assert.True(decisions.Length > 0, "no [ContextParallel] decision was logged — the configured rank "
                + "list's path must always be observable, active or not.");

            Assert.Equal(baseline.Frames.Count, parallel.Frames.Count);
            VideoFrame bMid = baseline.Frames[baseline.Frames.Count / 2];
            VideoFrame pMid = parallel.Frames[parallel.Frames.Count / 2];
            double ssim = Ssim.Compute(bMid.Rgb, pMid.Rgb, bMid.Width, bMid.Height);
            bool active = decisions.Any(d => d.Contains("active"));
            _output.WriteLine($"OUTCOME: {(active ? "ACTIVE (sequence split)" : "FALLBACK (single-GPU)")}; middle-frame SSIM = {ssim:F4}");
            // 0.90 bar per the class remarks: measured 0.9616 active; cross-arch drift ceiling 0.7774; a real
            // exchange/boundary defect lands far below this band (the synthetic tests pin exactness separately).
            Assert.True(ssim > 0.90, $"context-parallel run diverged from single-GPU baseline (SSIM={ssim:F4}, " +
                "measured 0.9616 when this gate was set) — check the K/V exchange and the frame-aligned split " +
                "before blaming cross-arch drift (ceiling 0.7774 is documented in WanCrossGpuRegimeDiagnosticTests).");
        }
        finally
        {
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
    }
}
