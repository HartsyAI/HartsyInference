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
/// split across them behind a queue — plain data parallelism, no <c>PlacementConfig</c> anywhere. Possible since
/// per-backend State landed (<c>DeviceGate</c> serializes per-ordinal, so engines on DIFFERENT ordinals genuinely
/// run concurrently — the shape that surfaced and verified the <c>Tensor.EnsureCpuData</c> race fix); this class
/// is the missing pin. Each request runs the exact single-GPU code path — assertions are completion + non-empty
/// output, NOT cross-run comparison (different architectures, no cross-GPU hand-off to gate).
/// <para><b>Model choice</b>: Llama-3.2-1B Q8 (~1.3 GB — trivially replicable on both cards). Image serving was
/// tried first and is HONESTLY not demonstrable on this box: SDXL's F32-staged UNet replica OOMed the 3060's
/// warm-up even at 512² (measured 2026-08-07: the ~10 GB replica left ~0.5 GB and a 1021 MB weight upload
/// failed), and every other on-disk image checkpoint needs most of both cards. The serving PATTERN is
/// model-agnostic — engines, gates, and per-backend state have no modality dimension — so the text model
/// proves it without VRAM roulette; image DP needs either a smaller image checkpoint or bigger cards.</para>
/// <para><b>Timings are printed, not asserted</b>: the concurrent-vs-sequential wall-clock ratio is bounded by
/// the slower card, so the honest expectation on a heterogeneous pair is "faster than sequential-on-one-engine,
/// well short of 2×". Numbers become claims only after a real run lands them in the benchmark doc.</para></summary>
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
        string checkpoint = TestPaths.Llm.Llama32_1BQ8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(DataParallelServingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        (double freeGb0, double freeGb1) = ProbeFreeGb(ptxDir);
        _output.WriteLine($"Pre-load free VRAM — ordinal 0: {freeGb0:F2} GB, ordinal 1: {freeGb1:F2} GB");
        if (freeGb0 < 3.0 || freeGb1 < 3.0)
        {
            _output.WriteLine("SKIPPED: insufficient free VRAM for a replica per card.");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("llama-3.2-1b", checkpoint, Modality.Text);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: llama-3.2-1b not resolvable with the explicit path."); return; }

        // Distinct prompts so each request does real, distinguishable work; greedy so outputs are stable.
        static TextRequest MakeRequest(int i) => new TextRequest
        {
            Messages = [new TextMessage
            {
                Role = TextRole.User,
                Content = $"List three facts about the number {i + 7}. Be brief.",
            }],
            MaxTokens = 64,
            Greedy = true,
            Temperature = 0.0,
            Seed = 42,
        };

        using InferenceEngine engine0 = new InferenceEngine("cuda", 0);
        using InferenceEngine engine1 = new InferenceEngine("cuda", 1);
        InferenceEngine[] engines = [engine0, engine1];

        // Warm both engines (loads a replica per card) so both timed phases compare steady-state serving.
        _output.WriteLine("[1/3] Warm-up: one generation per engine (loads the replica on each card)...");
        Stopwatch sw = Stopwatch.StartNew();
        await engine0.Text.GenerateAsync(spec, MakeRequest(0));
        await engine1.Text.GenerateAsync(spec, MakeRequest(0));
        _output.WriteLine($"  warm-up: {sw.Elapsed.TotalSeconds:F1}s (both replicas loaded)");
        (double warmFree0, double warmFree1) = ProbeFreeGb(ptxDir);
        _output.WriteLine($"  per-card free VRAM with both replicas resident — ordinal 0: {warmFree0:F2} GB "
            + $"(replica ~{freeGb0 - warmFree0:F2} GB), ordinal 1: {warmFree1:F2} GB (replica ~{freeGb1 - warmFree1:F2} GB)");

        _output.WriteLine($"[2/3] SEQUENTIAL baseline: {RequestCount} requests, all on engine 0...");
        sw.Restart();
        for (int i = 0; i < RequestCount; i++)
        {
            TextResult result = await engine0.Text.GenerateAsync(spec, MakeRequest(i));
            Assert.False(string.IsNullOrWhiteSpace(result.Text), $"sequential[{i}] produced no text");
        }
        double sequentialSeconds = sw.Elapsed.TotalSeconds;
        _output.WriteLine($"  sequential: {sequentialSeconds:F1}s ({sequentialSeconds / RequestCount:F1}s/request)");

        _output.WriteLine($"[3/3] DATA-PARALLEL: the same {RequestCount} requests split round-robin across both engines...");
        sw.Restart();
        Task<TextResult>[] inFlight = new Task<TextResult>[RequestCount];
        for (int i = 0; i < RequestCount; i++)
        {
            inFlight[i] = engines[i % engines.Length].Text.GenerateAsync(spec, MakeRequest(i));
        }
        TextResult[] results = await Task.WhenAll(inFlight);
        double parallelSeconds = sw.Elapsed.TotalSeconds;

        Assert.Equal(RequestCount, results.Length);
        for (int i = 0; i < results.Length; i++)
        {
            Assert.False(string.IsNullOrWhiteSpace(results[i].Text), $"parallel[{i}] (engine {i % engines.Length}) produced no text");
            Assert.True(results[i].CompletionTokens > 0, $"parallel[{i}] generated zero tokens");
        }

        // Informational until the benchmark doc records a run — the ratio is bounded by the slower card.
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
}
