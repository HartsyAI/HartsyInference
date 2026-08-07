using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The THROUGHPUT pattern (docs/PARALLELISM_GUIDE.md §"Serve MORE requests"): one independent
/// <c>InferenceEngine</c> per GPU, each holding its own full copy of the same model, with concurrent requests
/// split across them behind a queue — plain data parallelism, no <c>PlacementConfig</c> anywhere. This has been
/// possible since per-backend State landed (Phase 1A; <c>DeviceGate</c> serializes per-ordinal, so engines on
/// DIFFERENT ordinals genuinely run concurrently — the shape that surfaced and verified the
/// <c>Tensor.EnsureCpuData</c> race fix, see <c>TensorConcurrentSyncTests</c>) but had no dedicated test pinning
/// it; this class is that pin. Unlike every sibling <c>*ShardingEngineTests</c>, each request here runs the exact
/// single-GPU code path — the assertions are completion + per-image coherence, NOT cross-run SSIM (the two cards
/// are different architectures, and no cross-GPU hand-off exists to gate).
/// <para><b>Model choice</b>: SDXL via <see cref="TestPaths.Sdxl.SingleFile"/> — the same on-disk checkpoint
/// <c>SdxlCfgParallelEngineTests</c>/<c>SdxlComponentPlacementEngineTests</c> already gate on. Honest headroom
/// note: <c>SdxlRecipe</c> stages the UNet as F32 (~10 GB, measured in the benchmark doc's SDXL CFG-parallel
/// row), so the 12 GB card runs its replica tight — geometry below is 768² (not the siblings' 1024²) to keep
/// activation pressure off the small card. If the first real run still thrashes the 3060, shrink geometry
/// before touching the pattern.</para>
/// <para><b>Timings are printed, not asserted</b>: the concurrent-vs-sequential wall-clock ratio depends on the
/// slower card (the queue's tail request finishes at the 3060's pace), so the honest expectation on a
/// heterogeneous pair is "faster than sequential-on-one-engine, well short of 2×". Numbers become claims only
/// after a real run lands them in the benchmark doc.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DataParallelServingEngineTests
{
    private const int RequestCount = 4;

    private readonly ITestOutputHelper _output;
    public DataParallelServingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DataParallel_TwoEngines_ConcurrentRequests_AllComplete_WallClockAndVramReported()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Sdxl.SingleFile;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(DataParallelServingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        // Free-VRAM gate matching the campaign's per-card minimums: each engine holds a FULL model copy
        // (that is the point of data parallelism), and SDXL's F32-staged UNet needs most of the 12 GB card.
        (double freeGb0, double freeGb1) = ProbeFreeGb(ptxDir);
        _output.WriteLine($"Pre-load free VRAM — ordinal 0: {freeGb0:F2} GB, ordinal 1: {freeGb1:F2} GB");
        if (freeGb0 < 12.0 || freeGb1 < 11.0)
        {
            _output.WriteLine("SKIPPED: insufficient free VRAM for a full SDXL replica per card.");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("sdxl", checkpoint, Modality.Image);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: sdxl not resolvable with the explicit path."); return; }

        ImageRequest MakeRequest(int seed) => new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = 768,
            Height = 768,
            Steps = 10,
            CfgScale = 6.0f,
            Seed = seed,
        };

        using InferenceEngine engine0 = new InferenceEngine("cuda", 0);
        using InferenceEngine engine1 = new InferenceEngine("cuda", 1);
        InferenceEngine[] engines = [engine0, engine1];

        // Warm both engines first (one generation each) so both timed phases below compare steady-state
        // serving, not one-time checkpoint load/cast cost — the same cold/warm discipline the benchmark doc's
        // live-SwarmUI rows use.
        _output.WriteLine("[1/3] Warm-up: one generation per engine (loads the replica on each card)...");
        Stopwatch sw = Stopwatch.StartNew();
        await engine0.Images.GenerateAsync(spec, MakeRequest(seed: 1));
        await engine1.Images.GenerateAsync(spec, MakeRequest(seed: 1));
        _output.WriteLine($"  warm-up: {sw.Elapsed.TotalSeconds:F1}s (both replicas loaded)");
        (double warmFree0, double warmFree1) = ProbeFreeGb(ptxDir);
        _output.WriteLine($"  per-card free VRAM with both replicas resident — ordinal 0: {warmFree0:F2} GB "
            + $"(replica ~{freeGb0 - warmFree0:F2} GB), ordinal 1: {warmFree1:F2} GB (replica ~{freeGb1 - warmFree1:F2} GB)");

        _output.WriteLine($"[2/3] SEQUENTIAL baseline: {RequestCount} requests, all on engine 0...");
        sw.Restart();
        for (int i = 0; i < RequestCount; i++)
        {
            ImageResult result = await engine0.Images.GenerateAsync(spec, MakeRequest(seed: 100 + i));
            AssertCoherent(result.Rgb, $"sequential[{i}]");
        }
        double sequentialSeconds = sw.Elapsed.TotalSeconds;
        _output.WriteLine($"  sequential: {sequentialSeconds:F1}s ({sequentialSeconds / RequestCount:F1}s/request)");

        _output.WriteLine($"[3/3] DATA-PARALLEL: the same {RequestCount} requests split round-robin across both engines...");
        sw.Restart();
        Task<ImageResult>[] inFlight = new Task<ImageResult>[RequestCount];
        for (int i = 0; i < RequestCount; i++)
        {
            inFlight[i] = engines[i % engines.Length].Images.GenerateAsync(spec, MakeRequest(seed: 100 + i));
        }
        ImageResult[] results = await Task.WhenAll(inFlight);
        double parallelSeconds = sw.Elapsed.TotalSeconds;

        Assert.Equal(RequestCount, results.Length);
        for (int i = 0; i < results.Length; i++)
        {
            Assert.Equal(768, results[i].Width);
            Assert.Equal(768, results[i].Height);
            AssertCoherent(results[i].Rgb, $"parallel[{i}] (engine {i % engines.Length})");
        }

        // Informational until a real run lands in the benchmark doc — see the class doc for why the ratio
        // is bounded by the slower card, not 2×.
        _output.WriteLine($"WALL CLOCK: sequential-on-one-engine {sequentialSeconds:F1}s vs data-parallel "
            + $"{parallelSeconds:F1}s ({sequentialSeconds / parallelSeconds:F2}x)");
        (double endFree0, double endFree1) = ProbeFreeGb(ptxDir);
        _output.WriteLine($"Post-run free VRAM — ordinal 0: {endFree0:F2} GB, ordinal 1: {endFree1:F2} GB");
    }

    /// <summary>Throwaway per-ordinal probes, same pattern as <c>Krea2DitShardingEngineTests</c> — per-backend
    /// State makes these safe to construct beside the engines' own backends.</summary>
    private static (double FreeGb0, double FreeGb1) ProbeFreeGb(string ptxDir)
    {
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        (nuint free1, _) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), free1 / (1024.0 * 1024.0 * 1024.0));
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }
}
