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

/// <summary>LLM layer-split verification: drives REAL GGUF checkpoints through the full <c>InferenceEngine</c> →
/// <c>TextService</c> → <c>GenericTransformer.ForwardStaged</c> path with <see cref="PlacementConfig.ShardDevices"/>
/// set (the route an operator reaches via <c>--device "cuda:0+cuda:1"</c> / the extension's shard setting), closing
/// the gap left by <see cref="LlmPlacementTests"/> (synthetic-weight unit test — do not modify that file). Two
/// facts: exact greedy-decode token parity between unsharded and 2-GPU-sharded on a small model (placement changes
/// nothing about the math), and real decode tok/s on a model that does not fit either single card in this box's
/// pool. <c>TextResult</c> does not surface raw token IDs, so the parity fact asserts exact decoded-text equality
/// plus equal completion-token counts under GREEDY sampling — with a fixed prompt and a deterministic decode path,
/// two runs can only produce identical text if they walked the identical token sequence.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LlmShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public LlmShardingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task Sharded_TwoGpus_ExactTokenParity_VsUnsharded_GreedyFixedSeed()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Llm.Llama32_1BQ8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        (double free0, long total0, double free1, long total1) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        if (free0 < 1.5 || free1 < 1.5)
        {
            _output.WriteLine("SKIPPED: insufficient free VRAM on one or both cards for even this ~1.3 GB model.");
            return;
        }

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
        _output.WriteLine("[1/2] UNSHARDED baseline (single GPU, ordinal 0)...");
        TextResult baseline;
        using (InferenceEngine unshardedEngine = new InferenceEngine("cuda", 0))
        {
            baseline = await unshardedEngine.Text.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: \"{baseline.Text}\" ({baseline.CompletionTokens} tokens, stop={baseline.Stop}), {sw.Elapsed.TotalSeconds:F1}s");

        sw.Restart();
        _output.WriteLine("[2/2] SHARDED (layer split cuda:0 + cuda:1)...");
        // Per-card used-VRAM peaks sampled during the sharded generation are the "it actually sharded" evidence —
        // text equality alone can't distinguish a real 2-GPU split from a sharded engine that silently fell back
        // to running the whole model on one card (which would ALSO pass the parity assert, proving nothing).
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

        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"] };
        TextResult sharded;
        using (InferenceEngine shardedEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            sharded = await shardedEngine.Text.GenerateAsync(spec, request);
        }
        samplerCts.Cancel();
        await sampler;
        _output.WriteLine($"  sharded:  \"{sharded.Text}\" ({sharded.CompletionTokens} tokens, stop={sharded.Stop}), {sw.Elapsed.TotalSeconds:F1}s");

        Assert.False(string.IsNullOrWhiteSpace(baseline.Text), "baseline produced no text — check the checkpoint/prompt before trusting the parity comparison.");
        Assert.True(baseline.CompletionTokens > 0, "baseline generated zero tokens.");
        Assert.Equal(baseline.Stop, sharded.Stop);
        Assert.Equal(baseline.CompletionTokens, sharded.CompletionTokens);
        Assert.Equal(baseline.Text, sharded.Text);

        if (baselineMib.Length >= 2)
        {
            (int smi0, int smi1) = MapSmiRowsToOrdinals(total0, total1, baselineMib.Length);
            long rise0 = peakMib[smi0] - baselineMib[smi0], rise1 = peakMib[smi1] - baselineMib[smi1];
            _output.WriteLine($"Peak VRAM rise during sharded generation: ordinal 0 +{rise0} MiB, ordinal 1 +{rise1} MiB.");
            // Floor: the smaller card's byte-weighted share of the ~1.3 GB Q8 weights is ~0.45-0.5 GB, plus a fresh
            // CUDA context — a bare context alone (~200-500 MiB, i.e. a silent one-card fallback) cannot reach 550.
            // Measured real min rise on this box: 608 MiB (2026-08-06) — 550 keeps margin on both sides.
            Assert.True(Math.Min(rise0, rise1) > 550,
                $"expected BOTH cards to hold a layer range (pooling), got rises {rise0}/{rise1} MiB (ordinal 0/1) — " +
                "check whether TextService.ResolveShardDevices actually took the LoadSharded branch.");
        }
        else
        {
            _output.WriteLine("nvidia-smi unavailable — VRAM pooling assertion skipped (parity check still ran).");
        }
    }

    /// <summary>Per-card used VRAM in MiB from nvidia-smi, or empty when unavailable.</summary>
    private static long[] QueryUsedMib() => QueryMib("--query-gpu=memory.used");

    /// <summary>Per-card total VRAM in MiB from nvidia-smi, or empty when unavailable.</summary>
    private static long[] QueryTotalMib() => QueryMib("--query-gpu=memory.total");

    private static long[] QueryMib(string query)
    {
        try
        {
            using Process p = Process.Start(new ProcessStartInfo("nvidia-smi", $"{query} --format=csv,noheader,nounits")
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

    /// <summary>Maps CUDA ordinals 0/1 to nvidia-smi row indexes by closest total capacity — nvidia-smi's row order
    /// is its own device index, NOT the CUDA ordinal (they are swapped on this box) — falling back to row order when
    /// nvidia-smi is unavailable or the cards are indistinguishable (same shape as
    /// <c>MiniMaxH3DitShardingEngineTests</c>).</summary>
    private static (int smi0, int smi1) MapSmiRowsToOrdinals(long total0Bytes, long total1Bytes, int rowCount)
    {
        long[] totalMib = QueryTotalMib();
        int smi0 = ClosestByTotal(totalMib, total0Bytes);
        int smi1 = ClosestByTotal(totalMib, total1Bytes);
        return smi0 < 0 || smi1 < 0 || smi0 == smi1 || smi0 >= rowCount || smi1 >= rowCount ? (0, 1) : (smi0, smi1);
    }

    /// <summary>Index into the nvidia-smi row list whose reported total capacity is closest to
    /// <paramref name="cudaTotalBytes"/>, or -1 when nvidia-smi is unavailable.</summary>
    private static int ClosestByTotal(long[] nvidiaSmiTotalMib, long cudaTotalBytes)
    {
        if (nvidiaSmiTotalMib.Length == 0)
        {
            return -1;
        }
        long cudaTotalMib = cudaTotalBytes / (1024 * 1024);
        int best = 0;
        long bestDiff = long.MaxValue;
        for (int i = 0; i < nvidiaSmiTotalMib.Length; i++)
        {
            long diff = Math.Abs(nvidiaSmiTotalMib[i] - cudaTotalMib);
            if (diff < bestDiff)
            {
                bestDiff = diff;
                best = i;
            }
        }
        return best;
    }

    /// <summary>Real decode tok/s under a 2-GPU layer split, using the delta method documented in
    /// <c>benchmarks/results/2026-08-05_multigpu_speeds.md</c> ("Method notes"): tok/s = (T_256max − T_8max) / 248,
    /// greedy, identical prompt — isolates steady-state decode from load + prefill by subtracting two warm-model
    /// runs of different length. The first (untimed) call absorbs the one-time weight load so neither timed call
    /// pays for it. Qwen3-32B Q4_K_M (~19.8 GB) does not fit resident on either single card in this box's pool
    /// (24 GB / 12 GB) once KV + activations + driver overhead are counted — this is the model class the layer
    /// split exists for.</summary>
    [Fact]
    [Trait("Category", "Slow")]
    public async Task Sharded_TwoGpus_Qwen3_32B_DecodeTokPerSec()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Llm.Qwen3_32BQ4KM;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        (double free0, _, double free1, _) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        if (free0 + free1 < 27.0 || Math.Min(free0, free1) < 9.0)
        {
            _output.WriteLine("SKIPPED: insufficient pooled free VRAM for the ~19.8 GB Q4_K_M weights plus KV/activations on both cards.");
            return;
        }

        // TextService.EnsureRamHeadroomFor rejects the load outright below ~2.5x the file size (GgufLanguageModel
        // dequantizes tensors onto host buffers atop the mmap during load) — mirror that gate here so an
        // under-provisioned host reports a clear skip instead of the engine's own HartsyInferenceException.
        double availableRamGb = AvailableRamGb();
        double checkpointGb = new FileInfo(checkpoint).Length / (1024.0 * 1024.0 * 1024.0);
        double requiredRamGb = checkpointGb * 2.5;
        _output.WriteLine($"Host RAM — available: {availableRamGb:F1} GB, required (~2.5x the {checkpointGb:F1} GB checkpoint): {requiredRamGb:F1} GB.");
        if (availableRamGb > 0 && availableRamGb < requiredRamGb)
        {
            _output.WriteLine("SKIPPED: insufficient free host RAM for GGUF dequantization-on-load — free RAM " +
                "(e.g. stop other heavy processes) and retry rather than trusting a forced load.");
            return;
        }

        ModelSpec spec = ModelResolver.Resolve("qwen3-32b", checkpoint, Modality.Text);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: qwen3-32b not resolvable with the explicit path."); return; }

        // A counting continuation almost never hits EOS inside 256 greedy tokens — an early stop would make both
        // timed calls finish at the same token count and silently zero out the measured delta.
        TextRequest BuildRequest(int maxTokens) => new TextRequest
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "Continue this sequence indefinitely, one number per line, no other text: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10," }],
            MaxTokens = maxTokens,
            Greedy = true,
            Temperature = 0.0,
            Seed = 42,
        };

        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"] };
        using InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement });

        Stopwatch loadSw = Stopwatch.StartNew();
        _output.WriteLine("[warmup] loading + priming the sharded model (untimed for the tok/s measurement)...");
        TextResult warmup = await engine.Text.GenerateAsync(spec, BuildRequest(4));
        _output.WriteLine($"  warmup: {warmup.CompletionTokens} tokens in {loadSw.Elapsed.TotalSeconds:F1}s (includes weight load).");

        Stopwatch sw8 = Stopwatch.StartNew();
        TextResult r8 = await engine.Text.GenerateAsync(spec, BuildRequest(8));
        TimeSpan t8 = sw8.Elapsed;
        _output.WriteLine($"[T8]   {r8.CompletionTokens} tokens in {t8.TotalSeconds:F3}s");

        Stopwatch sw256 = Stopwatch.StartNew();
        TextResult r256 = await engine.Text.GenerateAsync(spec, BuildRequest(256));
        TimeSpan t256 = sw256.Elapsed;
        _output.WriteLine($"[T256] {r256.CompletionTokens} tokens in {t256.TotalSeconds:F3}s");

        Assert.True(r8.CompletionTokens >= 8, $"the 8-token call stopped early at {r8.CompletionTokens} tokens (stop={r8.Stop}) — the delta method needs both calls to run their full budget.");
        Assert.True(r256.CompletionTokens >= 256, $"the 256-token call stopped early at {r256.CompletionTokens} tokens (stop={r256.Stop}) — the delta method needs both calls to run their full budget.");

        double deltaSeconds = (t256 - t8).TotalSeconds;
        double tokPerSec = 248.0 / deltaSeconds;
        _output.WriteLine($"Decode speed (sharded, delta method): {tokPerSec:F1} tok/s over {deltaSeconds:F3}s delta.");
        Assert.True(tokPerSec > 0.5, $"measured {tokPerSec:F2} tok/s — implausibly slow, check for a stalled decode path rather than trusting this number.");
    }

    /// <summary>Free host RAM (<c>MemAvailable</c>) in GB from <c>/proc/meminfo</c>, or -1 when unreadable.</summary>
    private static double AvailableRamGb()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/meminfo"))
            {
                if (!line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                    continue;
                string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2 && long.TryParse(parts[1], out long kb))
                    return kb / (1024.0 * 1024.0);
            }
        }
        catch (Exception)
        {
            return -1;
        }
        return -1;
    }

    private static (double free0Gb, long total0Bytes, double free1Gb, long total1Bytes) ProbeFreeGb()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(LlmShardingEngineTests).Assembly.Location)!, "Ptx");
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, nuint total0) = probe0.Context.GetMemoryInfo();
        (nuint free1, nuint total1) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), (long)total0, free1 / (1024.0 * 1024.0 * 1024.0), (long)total1);
    }
}
