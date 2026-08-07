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

/// <summary>VLM layer-split verification: closes the gap the mllama cross-attention staging work (Phase 5,
/// 2026-08-05) opened — <c>TextService.LoadSharded</c> used to skip the mmproj sidecar entirely under
/// <see cref="PlacementConfig.ShardDevices"/>, so no VLM (mllama or splice-style) could answer image questions
/// once sharded. Drives REAL GGUF + mmproj checkpoints through the full <c>InferenceEngine</c> → <c>TextService</c>
/// → <c>MllamaGenerator</c>/<c>MultimodalGenerator</c> → <c>GenericTransformer.ForwardEmbedsStaged</c> path,
/// mirroring <see cref="LlmShardingEngineTests"/>'s exact-text-parity methodology: greedy + fixed seed, so two
/// runs can only produce identical text if they walked the identical token sequence — including the
/// cross-attention-state peer copy for mllama and the plain staged embeds handoff for the splice-style path.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class VlmShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public VlmShardingEngineTests(ITestOutputHelper output) => _output = output;

    /// <summary>Two clean color blocks — a real, unambiguous, greedy-answerable VLM question (not a photo, but a
    /// genuine "describe what's in the image" query the model must actually look at the pixels to answer) built
    /// in-process so the test needs no external image asset.</summary>
    private static ImageData TwoColorBlockImage(int size = 448)
    {
        byte[] rgb = new byte[size * size * 3];
        for (int y = 0; y < size; y++)
        {
            bool top = y < size / 2;
            for (int x = 0; x < size; x++)
            {
                int off = (y * size + x) * 3;
                if (top) { rgb[off] = 220; rgb[off + 1] = 30; rgb[off + 2] = 30; }      // top half: red
                else { rgb[off] = 20; rgb[off + 1] = 60; rgb[off + 2] = 220; }          // bottom half: blue
            }
        }
        return new ImageData { Rgb = rgb, Width = size, Height = size };
    }

    private static TextRequest VisionRequest(string question, int maxTokens) => new TextRequest
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = question, Images = [TwoColorBlockImage()] }],
        MaxTokens = maxTokens,
        Greedy = true,
        Temperature = 0.0,
        Seed = 42,
    };

    [Fact]
    public async Task Sharded_TwoGpus_Mllama_ExactTextParity_VsUnsharded_GreedyFixedSeed()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Llm.Llama32VisionQ4KM;
        string mmproj = Path.Combine(Path.GetDirectoryName(checkpoint)!, "Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf");
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint, mmproj)) return;

        (double free0, long total0, double free1, long total1) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        // ~5.6 GB text + ~1.8 GB vision + KV/activations for the unsharded baseline on ordinal 0 alone.
        if (free0 < 9.0) { _output.WriteLine("SKIPPED: insufficient free VRAM on ordinal 0 for the unsharded baseline (~7.4 GB weights)."); return; }
        if (free0 + free1 < 10.0 || Math.Min(free0, free1) < 1.5) { _output.WriteLine("SKIPPED: insufficient pooled free VRAM for the sharded run."); return; }

        ModelSpec spec = ModelResolver.Resolve("llama-3.2-11b-vision", checkpoint, Modality.Text);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: llama-3.2-11b-vision not resolvable with the explicit path."); return; }

        TextRequest request = VisionRequest(
            "Look at the image. Which color is on top and which is on the bottom? Answer with just the two color names.", 24);

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
            // Floor: the smaller card's split share alone is ~12/40 of the 5.6 GB Q4 text weights (~1.7 GB); the real
            // run measured +7500/+15534 MiB (2026-08-05 benchmark doc) — 1000 is half that share, 2x any bare context.
            Assert.True(Math.Min(rise0, rise1) > 1000,
                $"expected BOTH cards to hold a layer range (pooling), got rises {rise0}/{rise1} MiB (ordinal 0/1) — " +
                "check whether TextService.LoadSharded actually loaded the vision sidecar and reached the staged VLM path.");
        }
        else
        {
            _output.WriteLine("nvidia-smi unavailable — VRAM pooling assertion skipped (parity check still ran).");
        }
    }

    [Fact]
    public async Task Sharded_TwoGpus_Qwen25Vl_ExactTextParity_VsUnsharded_GreedyFixedSeed()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Llm.Qwen25Vl7BQ4KM;
        string mmproj = Path.Combine(Path.GetDirectoryName(checkpoint)!, "mmproj-F16.gguf");
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint, mmproj)) return;

        (double free0, long total0, double free1, long total1) = ProbeFreeGb();
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
        // ~4.7 GB text + ~1.35 GB vision + KV/activations for the unsharded baseline on ordinal 0 alone.
        if (free0 < 7.5) { _output.WriteLine("SKIPPED: insufficient free VRAM on ordinal 0 for the unsharded baseline (~6.1 GB weights)."); return; }
        if (free0 + free1 < 8.0 || Math.Min(free0, free1) < 1.5) { _output.WriteLine("SKIPPED: insufficient pooled free VRAM for the sharded run."); return; }

        ModelSpec spec = ModelResolver.Resolve("qwen2.5-vl-7b", checkpoint, Modality.Text);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: qwen2.5-vl-7b not resolvable with the explicit path."); return; }

        TextRequest request = VisionRequest(
            "Look at the image. Which color is on top and which is on the bottom? Answer with just the two color names.", 24);

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
            // Floor: the smaller card's split share alone is ~8/28 of the 4.7 GB Q4 text weights (~1.35 GB); the real
            // run measured +5864/+13014 MiB (2026-08-05 benchmark doc) — 700 is half that share, above any bare context.
            Assert.True(Math.Min(rise0, rise1) > 700,
                $"expected BOTH cards to hold a layer range (pooling), got rises {rise0}/{rise1} MiB (ordinal 0/1) — " +
                "check whether TextService.LoadSharded actually loaded the vision sidecar and reached the staged VLM path.");
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

    private static (double free0Gb, long total0Bytes, double free1Gb, long total1Bytes) ProbeFreeGb()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(VlmShardingEngineTests).Assembly.Location)!, "Ptx");
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, nuint total0) = probe0.Context.GetMemoryInfo();
        (nuint free1, nuint total1) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), (long)total0, free1 / (1024.0 * 1024.0 * 1024.0), (long)total1);
    }
}
