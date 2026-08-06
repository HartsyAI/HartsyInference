using System.Diagnostics;
using System.Text.RegularExpressions;
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

/// <summary>SDXL CFG-branch parallelism through the full engine path on the real SDXL base checkpoint —
/// mirrors <c>WanCfgParallelEngineTests</c> and <c>FluxCfgParallelFallbackTests</c>. EITHER outcome is a pass,
/// observably logged: (a) the UNet replica fits the second card → "[CfgParallel] active" and the output matches
/// the sequential (fused, single-GPU) baseline via SSIM (the two branches run on different-SM cards, so
/// bit-parity isn't the bar); or (b) it doesn't fit / the second card is under contention from other work →
/// "[CfgParallel] fell-back(preload-failed...)" and the generation still completes. SDXL's UNet is ~2.5 GB F16 —
/// far smaller than Flux's ~12 GB fp8 DiT — so unlike <c>FluxCfgParallelFallbackTests</c> (fallback-only on this
/// box), the ACTIVE branch is genuinely reachable here when the second card has room; the test adapts to
/// whatever the hardware does rather than assuming either outcome.
/// <para>Raises <see cref="Logs.MinLevel"/> to <see cref="LogLevel.Verbose"/> for both runs to capture the
/// <c>[sdxl-phase] denoise …ms</c> line — e2e wall-clock is dominated by checkpoint load + text-encode + VAE
/// decode (unaffected by CFG-parallel) plus, on the parallel run, the one-time second-backend UNet preload, so
/// the denoise-phase-only figure is the number that actually isolates the per-step concurrency win (or its
/// absence), matching the caveat <c>WanCfgParallelEngineTests</c>' own benchmark row already calls out.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class SdxlCfgParallelEngineTests
{
    private static readonly Regex DenoisePhaseMs = new Regex(@"\[sdxl-phase\] denoise (\d+)ms", RegexOptions.Compiled);

    private readonly ITestOutputHelper _output;
    public SdxlCfgParallelEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task CfgParallel_RealEngine_EitherOutcome_ObservableAndCorrect()
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

        LogLevel savedMinLevel = Logs.MinLevel;
        Logs.MinLevel = LogLevel.Verbose;
        try
        {
            List<string> baselineLog = new();
            Logs.SetLogger((level, message) =>
            {
                lock (baselineLog) baselineLog.Add(message);
                Console.Error.WriteLine($"[{level}] {message}");
            });

            Stopwatch sw = Stopwatch.StartNew();
            _output.WriteLine("[1/2] Sequential (fused, single-GPU) baseline...");
            ImageResult baseline;
            using (InferenceEngine engine = new InferenceEngine("cuda", 0))
            {
                baseline = await engine.Images.GenerateAsync(spec, request);
            }
            double baselineSeconds = sw.Elapsed.TotalSeconds;
            long? baselineDenoiseMs = ExtractDenoiseMs(baselineLog);
            _output.WriteLine($"  baseline: {baselineSeconds:F1}s e2e"
                + (baselineDenoiseMs is not null ? $", {baselineDenoiseMs}ms denoise-only" : ""));
            AssertCoherent(baseline.Rgb, "baseline");

            List<string> parallelLog = new();
            Logs.SetLogger((level, message) =>
            {
                lock (parallelLog) parallelLog.Add(message);
                Console.Error.WriteLine($"[{level}] {message}");
            });

            sw.Restart();
            _output.WriteLine("[2/2] CFG-parallel (uncond branch on ordinal 1)...");
            PlacementConfig placement = new PlacementConfig { CfgParallelDevice = "cuda:1" };
            ImageResult parallel;
            using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
            {
                parallel = await engine.Images.GenerateAsync(spec, request);
            }
            double parallelSeconds = sw.Elapsed.TotalSeconds;
            long? parallelDenoiseMs = ExtractDenoiseMs(parallelLog);
            _output.WriteLine($"  cfg-parallel run: {parallelSeconds:F1}s e2e (baseline {baselineSeconds:F1}s)"
                + (parallelDenoiseMs is not null ? $", {parallelDenoiseMs}ms denoise-only" : ""));
            if (baselineDenoiseMs is not null && parallelDenoiseMs is not null)
            {
                _output.WriteLine($"  denoise-only: baseline {baselineDenoiseMs}ms vs cfg-parallel {parallelDenoiseMs}ms "
                    + $"({(double)baselineDenoiseMs / parallelDenoiseMs.Value:F2}x)");
            }
            AssertCoherent(parallel.Rgb, "cfg-parallel");

            string[] decisions;
            lock (parallelLog) decisions = parallelLog.Where(m => m.Contains("[CfgParallel]")).ToArray();
            _output.WriteLine($"CfgParallel decisions observed: {string.Join(" | ", decisions)}");
            Assert.True(decisions.Length > 0, "no [CfgParallel] decision was logged — the configured second GPU's "
                + "path must always be observable, active or not.");

            double ssim = Ssim.Compute(baseline.Rgb, parallel.Rgb, parallel.Width, parallel.Height);
            bool active = decisions.Any(d => d.Contains("active"));
            _output.WriteLine($"OUTCOME: {(active ? "ACTIVE (concurrent branches)" : "FALLBACK (sequential/fused)")}; SSIM = {ssim:F4}");
            // Active: cross-SM branch execution → SSIM bar (F16 numerics drift slightly card to card). Fallback:
            // same math as baseline on the same card → near-identical, but the same SSIM bar keeps this uniform.
            Assert.True(ssim > 0.9, $"cfg-parallel run diverged from the sequential baseline (SSIM={ssim:F4}).");
        }
        finally
        {
            Logs.MinLevel = savedMinLevel;
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
    }

    /// <summary>Pulls the last <c>[sdxl-phase] denoise …ms</c> value out of a captured log — the denoise-loop-only
    /// duration, isolated from checkpoint load / text-encode / VAE decode (and, on the parallel run, the
    /// one-time second-backend preload, which happens before the loop starts).</summary>
    private static long? ExtractDenoiseMs(List<string> log)
    {
        lock (log)
        {
            for (int i = log.Count - 1; i >= 0; i--)
            {
                Match m = DenoisePhaseMs.Match(log[i]);
                if (m.Success) return long.Parse(m.Groups[1].Value);
            }
        }
        return null;
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }
}
