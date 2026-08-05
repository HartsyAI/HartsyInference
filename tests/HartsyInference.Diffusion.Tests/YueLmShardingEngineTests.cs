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

namespace HartsyInference.Diffusion.Tests;

/// <summary>Audio-LM layer-split verification: drives a REAL YuE generation through the full
/// <c>InferenceEngine</c> → <c>MusicService</c> → <c>YueMusicModel</c> → <c>YuePipeline</c> path with
/// <see cref="PlacementConfig.ShardDevices"/> set (no DiT flag — the LM-only shard route). Proves the whole
/// chain an operator reaches via <c>--lm-shard-gpu</c> / the extension's shard setting: the quant policy
/// auto-resolves to un-quantized (checkpoint bf16) because pooling makes it affordable, the Stage-1 7B is
/// layer-split across both cards via <c>PlacementPlanner.LlmSplitPlan</c> + <c>LlmPlacement</c>, the
/// asymmetric per-stage preload genuinely pools (both cards' VRAM rises), and a coherent WAV comes out.
/// Lives in Diffusion.Tests with the other engine-level placement tests (the only test project that
/// references the Engine), same as the video-family MiniMaxH3 classes.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class YueLmShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public YueLmShardingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task LmSharding_RealEngine_UnquantizedStage1_PooledAcrossGpus_ProducesAudio()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = Path.Combine(RepoPaths.ModelsRoot(), "audio", "music", "yue", "en-cot",
            "model-00001-of-00003.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        // bf16 Stage-1 is ~13.5 GB canonical + KV/activations; require a workable pool on both cards.
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(YueLmShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        using (CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir))
        using (CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir))
        {
            (nuint free0, _) = probe0.Context.GetMemoryInfo();
            (nuint free1, _) = probe1.Context.GetMemoryInfo();
            double freeGb0 = free0 / (1024.0 * 1024.0 * 1024.0), freeGb1 = free1 / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"Free VRAM: ordinal 0 = {freeGb0:F2} GB, ordinal 1 = {freeGb1:F2} GB.");
            if (freeGb0 + freeGb1 < 18.0 || freeGb0 < 6.0 || freeGb1 < 6.0)
            {
                _output.WriteLine("SKIPPED: insufficient pooled free VRAM for the un-quantized 7B Stage-1.");
                return;
            }
        }

        ModelSpec spec = ModelResolver.Resolve("yue", modelPathArg: null, Modality.Music);
        MusicRequest request = new MusicRequest
        {
            Prompt = "uplifting synth pop, catchy chorus",
            Genre = "pop",
            Duration = 4,
            Seed = 7,
        };

        // Per-card used-VRAM peaks sampled during the generation are the pooling evidence: BOTH cards must
        // rise well past their idle baseline (a replicated or single-card load only raises one).
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
                    try { await Task.Delay(2000, samplerCts.Token); } catch (OperationCanceledException) { }
                }
            })
            : Task.CompletedTask;

        Stopwatch sw = Stopwatch.StartNew();
        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"] };
        AudioResult result;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            result = await engine.Music.GenerateAsync(spec, request);
        }
        samplerCts.Cancel();
        await sampler;
        _output.WriteLine($"Generated {result.DurationSeconds:F1}s @ {result.SampleRate} Hz in {sw.Elapsed.TotalSeconds:F1}s.");

        Assert.NotNull(result.Data);
        Assert.True(result.Data.Length > 8000, $"WAV unexpectedly small ({result.Data.Length} bytes).");
        Assert.True(result.DurationSeconds > 0.2, $"duration {result.DurationSeconds:F2}s — Stage-1 produced no usable frames.");
        // The runner cache key carries the resolved policy — proves the sharded default engaged un-quantized.
        Assert.Contains("lmq=Off", result.Meta["model"]);
        Assert.Contains("shard=", result.Meta["model"]);

        if (baselineMib.Length >= 2)
        {
            long rise0 = peakMib[0] - baselineMib[0], rise1 = peakMib[1] - baselineMib[1];
            _output.WriteLine($"Peak VRAM rise during generation: card0 +{rise0} MiB, card1 +{rise1} MiB.");
            Assert.True(Math.Min(rise0, rise1) > 1000,
                $"expected BOTH cards to hold a Stage-1 layer range (pooling), got rises {rise0}/{rise1} MiB — " +
                "check the per-stage asymmetric preload in YuePipeline.PreloadStage1.");
        }
        else
        {
            _output.WriteLine("nvidia-smi unavailable — VRAM pooling assertion skipped (audio + cache-key checks still ran).");
        }
    }

    /// <summary>The explicit env override must beat the sharded auto-default (Off): with
    /// <c>HARTSY_AUDIO_LM_QUANT=q4k</c> a sharded run still quantizes, observable in the runner cache key.</summary>
    [Fact]
    public async Task LmSharding_EnvQuantOverride_WinsOverShardedDefault()
    {
        Environment.SetEnvironmentVariable("HARTSY_AUDIO_LM_QUANT", "q4k");
        try
        {
            if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
            if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
            string checkpoint = Path.Combine(RepoPaths.ModelsRoot(), "audio", "music", "yue", "en-cot",
                "model-00001-of-00003.safetensors");
            if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

            ModelSpec spec = ModelResolver.Resolve("yue", modelPathArg: null, Modality.Music);
            MusicRequest request = new MusicRequest { Prompt = "gentle piano", Genre = "pop", Duration = 3, Seed = 7 };
            PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"] };
            AudioResult result;
            using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
            {
                result = await engine.Music.GenerateAsync(spec, request);
            }
            Assert.True(result.Data is { Length: > 0 });
            Assert.Contains("lmq=Q4K", result.Meta["model"]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_AUDIO_LM_QUANT", null);
        }
    }

    /// <summary>Per-card used VRAM in MiB from nvidia-smi, or empty when unavailable.</summary>
    private static long[] QueryUsedMib()
    {
        try
        {
            using Process p = Process.Start(new ProcessStartInfo("nvidia-smi",
                "--query-gpu=memory.used --format=csv,noheader,nounits")
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
