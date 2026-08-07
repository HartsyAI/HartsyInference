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

/// <summary>Qwen-Image context parallelism (img-row sequence split, replicated weights + replicated txt stream)
/// through the full engine path on the real Edit-2511 fp8 checkpoint — the WanContextParallelEngineTests geometry.
/// EITHER outcome is a pass and observable: (a) the replica fits rank 1 → "[ContextParallel] active(rows ...)" and
/// the output is compared to the single-GPU baseline; or (b) it doesn't fit → "[ContextParallel]
/// fell-back(preload-failed...)" and the generation still completes single-GPU.
///
/// <para><b>On THIS box (4090 24 GB + 3060 12 GB) active CP is expected to be UNREACHABLE for this model</b>: CP
/// replicates the whole ~19 GB fp8 DiT per rank, and 19 GB does not fit the 3060 — so what these facts actually
/// verify here is the observable preload-OOM fallback path (plus baseline-identical output through it). Active CP
/// gets verified on same-VRAM pairs; the synthetic <c>ContextParallelQwenImageTests</c> pin the split/exchange
/// mechanics regardless.</para>
///
/// <para><b>Why the default-regime SSIM below is INFORMATIONAL (floor 0.05), not a correctness bar</b>: this fp8
/// checkpoint's residual stream reaches ±10M by mid-depth, and the whole model run on the 3060 (no multi-GPU code
/// at all) scores SSIM ~0.14 vs the 4090 — the SM 8.9 native-fp8 vs SM 8.6 seam documented in
/// <c>QwenImageDitShardingEngineTests</c>' class remarks. An ACTIVE cross-device CP run computes a fraction of img
/// rows on the 3060 and inherits a share of that drift, so the default-regime number can only catch genuine
/// incoherence. The matched-regime fact below is the real gate.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class QwenImageContextParallelEngineTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageContextParallelEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task ContextParallel_RealEngine_EitherOutcome_ObservableAndCoherent()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        (double free0, double free1) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        if (free0 < 20.0 || free1 < 9.0)
        {
            _output.WriteLine("SKIPPED: the ~19 GB DiT needs most of ordinal 0 free (rank 1 needs headroom for whichever path it takes).");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("qwen-image", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: qwen-image not resolvable with the explicit path."); return; }

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
        _output.WriteLine("[1/2] Single-GPU baseline (ordinal 0)...");
        ImageResult baseline;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            baseline = await engine.Images.GenerateAsync(spec, request);
        }
        double baselineSeconds = sw.Elapsed.TotalSeconds;
        _output.WriteLine($"  baseline: {baseline.Width}x{baseline.Height}, seed={baseline.Seed}, {baselineSeconds:F1}s");
        AssertCoherent(baseline.Rgb, "baseline");

        List<string> captured = new();
        Logs.SetLogger((level, message) =>
        {
            lock (captured) captured.Add(message);
            Console.Error.WriteLine($"[{level}] {message}");
        });
        try
        {
            sw.Restart();
            _output.WriteLine("[2/2] Context-parallel (img-row split across cuda:0 + cuda:1)...");
            PlacementConfig placement = new PlacementConfig { ContextParallelDevices = ["cuda:0", "cuda:1"] };
            ImageResult parallel;
            using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
            {
                parallel = await engine.Images.GenerateAsync(spec, request);
            }
            _output.WriteLine($"  context-parallel run: {parallel.Width}x{parallel.Height}, {sw.Elapsed.TotalSeconds:F1}s (baseline {baselineSeconds:F1}s)");
            AssertCoherent(parallel.Rgb, "context-parallel");

            string[] decisions;
            lock (captured) decisions = captured.Where(m => m.Contains("[ContextParallel]")).ToArray();
            _output.WriteLine($"ContextParallel decisions observed: {string.Join(" | ", decisions)}");
            Assert.True(decisions.Length > 0, "no [ContextParallel] decision was logged — the configured rank "
                + "list's path must always be observable, active or not.");

            Assert.Equal(baseline.Width, parallel.Width);
            Assert.Equal(baseline.Height, parallel.Height);
            double ssim = Ssim.Compute(baseline.Rgb, parallel.Rgb, parallel.Width, parallel.Height);
            bool active = decisions.Any(d => d.Contains("active"));
            _output.WriteLine($"OUTCOME: {(active ? "ACTIVE (img-row split)" : "FALLBACK (single-GPU)")}; SSIM = {ssim:F4} "
                + "(informational vs the native-fp8-regime baseline — see class remarks)");
            // Informational floor only (class remarks): the fp8-seam drift ceiling on this box is ~0.14 for the
            // WHOLE model on the 3060, so a coherence floor is all the default regime can promise. A fallback run
            // is the same single-GPU computation twice and lands near 1.0.
            Assert.True(ssim > 0.05, $"context-parallel output is not just fp8-regime-divergent but incoherent "
                + $"(SSIM={ssim:F4}) — check the row split, rope-table slicing, and the joint K/V exchange.");
        }
        finally
        {
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
    }

    /// <summary>GATED cross-device tier: identical run, but <c>HARTSY_FP8_NATIVE=0</c> forces BOTH the baseline
    /// and the CP run onto the same (non-native) fp8 GEMM regime, removing the SM 8.9 native-fp8 quantization gap
    /// that makes the default-regime SSIM informational — the
    /// <see cref="QwenImageDitShardingEngineTests.DitSharding_RealEngine_CrossDevice_MatchedFp8Regime_WithinGatedToleranceOfUnsharded"/>
    /// pattern, gated at the Wan cross-arch CP band (&gt; 0.90). On this box the ~19 GB replica is expected to
    /// preload-OOM on the 3060 and take the observable fallback (then the comparison is single-GPU vs single-GPU
    /// and trivially passes — the fallback path itself is what is verified); a same-VRAM pair exercises the active
    /// gate. <c>CudaBackend</c> reads <c>HARTSY_FP8_NATIVE</c> once at construction, so it is set at the very top,
    /// before ANY backend (the VRAM probes included) is constructed — run this fact filter-isolated in its own
    /// process.</summary>
    [Fact]
    public async Task ContextParallel_RealEngine_MatchedFp8Regime_WithinGatedToleranceOfBaseline()
    {
        Environment.SetEnvironmentVariable("HARTSY_FP8_NATIVE", "0");

        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        (double free0, double free1) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        if (free0 < 20.0 || free1 < 9.0)
        {
            _output.WriteLine("SKIPPED: the ~19 GB DiT needs most of ordinal 0 free (rank 1 needs headroom for whichever path it takes).");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("qwen-image", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: qwen-image not resolvable with the explicit path."); return; }

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
        _output.WriteLine("[1/2] Single-GPU baseline (ordinal 0, HARTSY_FP8_NATIVE=0)...");
        ImageResult baseline;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0))
        {
            baseline = await engine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {baseline.Width}x{baseline.Height}, seed={baseline.Seed}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(baseline.Rgb, "baseline");

        List<string> captured = new();
        Logs.SetLogger((level, message) =>
        {
            lock (captured) captured.Add(message);
            Console.Error.WriteLine($"[{level}] {message}");
        });
        try
        {
            sw.Restart();
            _output.WriteLine("[2/2] Context-parallel (cuda:0 + cuda:1, both on the matched regime)...");
            PlacementConfig placement = new PlacementConfig { ContextParallelDevices = ["cuda:0", "cuda:1"] };
            ImageResult parallel;
            using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
            {
                parallel = await engine.Images.GenerateAsync(spec, request);
            }
            _output.WriteLine($"  context-parallel run: {parallel.Width}x{parallel.Height}, {sw.Elapsed.TotalSeconds:F1}s");
            AssertCoherent(parallel.Rgb, "context-parallel");

            string[] decisions;
            lock (captured) decisions = captured.Where(m => m.Contains("[ContextParallel]")).ToArray();
            _output.WriteLine($"ContextParallel decisions observed: {string.Join(" | ", decisions)}");
            Assert.True(decisions.Length > 0, "no [ContextParallel] decision was logged.");
            bool active = decisions.Any(d => d.Contains("active"));

            Assert.Equal(baseline.Width, parallel.Width);
            Assert.Equal(baseline.Height, parallel.Height);
            double ssim = Ssim.Compute(baseline.Rgb, parallel.Rgb, parallel.Width, parallel.Height);
            _output.WriteLine($"OUTCOME: {(active ? "ACTIVE (img-row split)" : "FALLBACK (single-GPU)")}; "
                + $"SSIM (matched fp8 regime) = {ssim:F4} (GATED > 0.90)");
            Assert.True(ssim > 0.90, $"regime-matched context-parallel run diverged from the single-GPU baseline "
                + $"(SSIM={ssim:F4}, path={(active ? "active" : "fallback")}) — with HARTSY_FP8_NATIVE=0 on both "
                + "sides the fp8-regime gap is gone, so check the row split, rope-table slicing, and the joint "
                + "K/V exchange (the Wan cross-arch CP band is > 0.90).");
        }
        finally
        {
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }

    private static (double free0Gb, double free1Gb) ProbeFreeGb()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageContextParallelEngineTests).Assembly.Location)!, "Ptx");
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        (nuint free1, _) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), free1 / (1024.0 * 1024.0 * 1024.0));
    }
}
