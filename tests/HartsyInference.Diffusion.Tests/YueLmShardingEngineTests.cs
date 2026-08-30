using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Cpu;
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
/// asymmetric per-stage preload genuinely pools (both cards' VRAM rises), and a coherent WAV comes out — verified
/// both by the pooling/cache-key assertions below and, for the primary fact, by feeding the WAV through a Whisper
/// STT content-word-recall gate (see <see cref="AssertLyricsAreIntelligible"/>).
/// Lives in Diffusion.Tests with the other engine-level placement tests, alongside <c>HartsyInference.LLM.Tests</c>'
/// own real-checkpoint engine-sharding tests — both test projects reference the Engine.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class YueLmShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public YueLmShardingEngineTests(ITestOutputHelper output) => _output = output;

    /// <summary>Real distinct content words in <see cref="LyricsPrompt"/>'s [verse]/[chorus] lines — the Whisper
    /// recall target. Deliberately NOT the genre/style text: an earlier draft of this test fed a style description
    /// ("uplifting synth pop, catchy chorus") as <see cref="MusicRequest.Prompt"/>, which YuE's tokenizer treats as
    /// the sung LYRICS (<c>YueTokenizer.SplitLyrics</c> finds no <c>[label]</c> markers in it and falls back to
    /// singing the raw style text verbatim) — Whisper correctly transcribed that as nothing, which looked like a
    /// broken pipeline but was actually a broken test.</summary>
    private static readonly string[] LyricContentWords = ["morning", "ocean", "flying", "burning"];

    /// <summary>Real structured lyrics (<c>[verse]</c>/<c>[chorus]</c> markers YuE's tokenizer splits on) — this is
    /// what <see cref="MusicRequest.Prompt"/> means for YuE; style/genre goes in <see cref="MusicRequest.Genre"/>.</summary>
    private const string LyricsPrompt = "[verse]\nGolden morning breaks across the ocean\n[chorus]\nWe keep flying through the burning sky";

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
            Prompt = LyricsPrompt,
            Genre = "uplifting synth pop, catchy chorus",
            Duration = 8,
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

        await AssertLyricsAreIntelligible(result, LyricContentWords);
    }

    /// <summary>Whisper content-word-recall gate mirroring <c>TranscribeWavFileTests</c>' harness (that test lives
    /// in a sibling test project this one doesn't reference, and exposes no separately-callable helper — its whole
    /// transcribe+recall logic is inline in its own <c>[Fact]</c> — so this reproduces the minimal piece directly
    /// against the already-referenced <c>HartsyInference.Audio</c>/<c>HartsyInference.Cpu</c> src packages).
    /// <c>WhisperPipeline.TranscribeWav</c> resamples internally, so the raw 44.1 kHz vocoder output is fed as-is.
    /// Requires at least half the content words, not all: sung-vocal ASR is measurably harder than the
    /// spoken-TTS word-perfect bar other models hit (docs/Checklists/MODEL_STATUS_AUDIO.md calls YuE's own manual
    /// verification "near-verbatim", not word-perfect), so a >=50% recall is the genuine-intelligibility line
    /// without being flaky on a single mis-heard word.</summary>
    private async Task AssertLyricsAreIntelligible(AudioResult result, string[] targetWords)
    {
        const string whisperRepo = "openai/whisper-medium";
        string whisperDir = AudioModelCache.GetRepoDirectory(whisperRepo, "stt");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(whisperDir, "model.safetensors"))) return;

        string wavPath = Path.Combine(Path.GetTempPath(), $"yue_lm_sharding_{Guid.NewGuid():N}.wav");
        try
        {
            File.WriteAllBytes(wavPath, result.Data);
            using CpuBackend backend = new();
            using WhisperPipeline stt = await WhisperPipeline.LoadAsync(whisperRepo);
            string heard = stt.TranscribeWav(backend, wavPath,
                new WhisperOptions { Language = "en", Translate = false, WithTimestamps = false }).Trim();
            _output.WriteLine($"Whisper heard: \"{heard}\"");

            string lower = heard.ToLowerInvariant();
            string[] hits = [.. targetWords.Where(w => lower.Contains(w.ToLowerInvariant()))];
            double recall = hits.Length / (double)targetWords.Length;
            _output.WriteLine($"Content-word recall: {hits.Length}/{targetWords.Length} ({recall:P0}) — {string.Join(",", hits)}");
            Assert.True(recall >= 0.5,
                $"Whisper recall {recall:P0} ({hits.Length}/{targetWords.Length}) on \"{heard}\" — sung lyrics are not "
                + "intelligible; check the Stage-2/vocoder path didn't silently fall back to the cb0-only 16 kHz draft.");
        }
        finally
        {
            File.Delete(wavPath);
        }
    }

    /// <summary>The explicit setting override must beat the sharded auto-default (Off): with
    /// <c>numerics.audioLmQuant=q4k</c> a sharded run still quantizes, observable in the runner cache key.</summary>
    [Fact]
    public async Task LmSharding_SettingQuantOverride_WinsOverShardedDefault()
    {
        KnobStore.Set(EngineKnobs.AudioLmQuant, "q4k");
        try
        {
            if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
            if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
            string checkpoint = Path.Combine(RepoPaths.ModelsRoot(), "audio", "music", "yue", "en-cot",
                "model-00001-of-00003.safetensors");
            if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

            ModelSpec spec = ModelResolver.Resolve("yue", modelPathArg: null, Modality.Music);
            MusicRequest request = new MusicRequest
            {
                Prompt = "[verse]\nSoft rain falls on the quiet town",
                Genre = "gentle piano",
                Duration = 3,
                Seed = 7,
            };
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
            KnobStore.Clear(EngineKnobs.AudioLmQuant);
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
