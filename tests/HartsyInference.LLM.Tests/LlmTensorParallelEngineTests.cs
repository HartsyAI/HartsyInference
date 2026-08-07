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

namespace HartsyInference.LLM.Tests;

/// <summary>SKELETON (Phase 3 TP v1 — written, deliberately NOT run yet): real-GGUF tensor-parallel greedy
/// parity through the full engine path, mirroring <see cref="LlmShardingEngineTests"/>' structure with
/// <see cref="PlacementConfig.TensorParallelDegree"/> = 2. BLOCKED on the coordinator wiring
/// <c>TextService</c> to consume the degree (building <c>TpPlacement</c> + <c>TensorParallelTransformer</c>
/// instead of the layer-split branch): until that lands, TextService interprets a 2-entry ShardDevices as a
/// LAYER split, which would ALSO pass the parity assert AND the both-cards-VRAM-rise check below — so this
/// test's evidence CANNOT yet distinguish real TP from a silent layer-split fallback (the DiT-mosaic lesson:
/// green-but-meaningless). Before running, the wiring must emit a distinguishable TP marker (e.g. a
/// "[Placement] LLM tensor parallel degree=2" log or an engine-surfaced placement descriptor) and this test
/// must assert on it. Llama-3.2-1B Q8_0 tiles cleanly at degree 2: 32 Q / 8 KV heads, I=8192 (4096/rank,
/// Q8_0 32-element blocks aligned), o_proj IN=2048 (1024/rank aligned).</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LlmTensorParallelEngineTests
{
    private readonly ITestOutputHelper _output;
    public LlmTensorParallelEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TensorParallel_TwoGpus_ExactTokenParity_VsSingleGpu_Greedy()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Llm.Llama32_1BQ8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        ModelSpec spec = ModelResolver.Resolve("llama-3.2-1b", checkpoint, Modality.Text);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: llama-3.2-1b not resolvable with the explicit path."); return; }

        TextRequest request = new TextRequest
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "What is the capital of France? Answer with just the city name and nothing else." }],
            MaxTokens = 16,
            Greedy = true,
            Temperature = 0.0,
            Seed = 42,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/2] single-GPU baseline (ordinal 0)...");
        TextResult baseline;
        using (InferenceEngine singleEngine = new InferenceEngine("cuda", 0))
        {
            baseline = await singleEngine.Text.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: \"{baseline.Text}\" ({baseline.CompletionTokens} tokens, stop={baseline.Stop}), {sw.Elapsed.TotalSeconds:F1}s");

        sw.Restart();
        _output.WriteLine("[2/2] TENSOR PARALLEL degree 2 (cuda:0 + cuda:1)...");
        // Per-card used-VRAM peaks are necessary but NOT sufficient evidence here (a layer-split fallback also
        // raises both cards) — see the class doc: the run must additionally assert the wiring's TP marker.
        long[] baselineMib = QueryUsedMib();
        long[] peakMib = [.. baselineMib];
        using CancellationTokenSource samplerCts = new();
        Task sampler = baselineMib.Length >= 2
            ? Task.Run(async () =>
            {
                while (!samplerCts.IsCancellationRequested)
                {
                    long[] now = QueryUsedMib();
                    for (int i = 0; i < peakMib.Length && i < now.Length; i++) peakMib[i] = Math.Max(peakMib[i], now[i]);
                    try { await Task.Delay(200, samplerCts.Token); } catch (OperationCanceledException) { }
                }
            })
            : Task.CompletedTask;

        PlacementConfig placement = new PlacementConfig
        {
            TensorParallelDegree = 2,
            ShardDevices = ["cuda:0", "cuda:1"],
        };
        // Capture engine logs so the TP marker can be asserted — the wiring emits
        // "[TensorParallel] active degree=..." (TextService.LoadTensorParallel), which is the discriminator
        // between real TP and a silent layer-split fallback (see class doc).
        List<string> captured = new();
        HartsyInference.Core.Logging.Logs.SetLogger((level, message) =>
        {
            lock (captured) captured.Add(message);
            Console.Error.WriteLine($"[{level}] {message}");
        });
        TextResult tp;
        try
        {
            using InferenceEngine tpEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement });
            tp = await tpEngine.Text.GenerateAsync(spec, request);
        }
        finally
        {
            HartsyInference.Core.Logging.Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
        samplerCts.Cancel();
        await sampler;
        string[] markers;
        lock (captured) markers = captured.Where(m => m.Contains("[TensorParallel] active")).ToArray();
        _output.WriteLine($"TP markers observed: {string.Join(" | ", markers)}");
        Assert.True(markers.Length > 0, "no '[TensorParallel] active' marker was logged — the engine did NOT "
            + "take the tensor-parallel branch (a layer-split fallback would still pass every other assert here).");
        _output.WriteLine($"  tp: \"{tp.Text}\" ({tp.CompletionTokens} tokens, stop={tp.Stop}), {sw.Elapsed.TotalSeconds:F1}s");

        Assert.False(string.IsNullOrWhiteSpace(baseline.Text), "baseline produced no text — check the checkpoint/prompt before trusting the parity comparison.");
        Assert.True(baseline.CompletionTokens > 0, "baseline generated zero tokens.");
        Assert.Equal(baseline.Stop, tp.Stop);
        Assert.Equal(baseline.CompletionTokens, tp.CompletionTokens);
        Assert.Equal(baseline.Text, tp.Text);

        if (baselineMib.Length >= 2)
        {
            long rise0 = peakMib[0] - baselineMib[0], rise1 = peakMib[1] - baselineMib[1];
            _output.WriteLine($"Peak VRAM rise during TP generation: row 0 +{rise0} MiB, row 1 +{rise1} MiB.");
            // Under TP each rank holds roughly HALF the ~1.3 GB Q8 weights plus its KV slice and a CUDA
            // context; a bare context alone (silent single-GPU fallback) cannot reach 550 MiB. NOTE: a
            // layer-split fallback ALSO clears this bar — the TP log-marker assert (see class doc) is the
            // discriminator that must be added at wiring time.
            Assert.True(Math.Min(rise0, rise1) > 550,
                $"expected BOTH cards to hold a TP weight shard, got rises {rise0}/{rise1} MiB — " +
                "check whether the TextService TP wiring actually took the tensor-parallel branch.");
        }
        else
        {
            _output.WriteLine("nvidia-smi unavailable — VRAM assertion skipped (parity check still ran).");
        }
    }

    /// <summary>Per-card used VRAM in MiB from nvidia-smi, or empty when unavailable.</summary>
    private static long[] QueryUsedMib()
    {
        try
        {
            using Process p = Process.Start(new ProcessStartInfo("nvidia-smi", "--query-gpu=memory.used --format=csv,noheader,nounits")
            { RedirectStandardOutput = true, UseShellExecute = false })!;
            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit(5000);
            return [.. output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(line => long.TryParse(line, out long v) ? v : 0)];
        }
        catch (Exception)
        {
            return [];
        }
    }
}
