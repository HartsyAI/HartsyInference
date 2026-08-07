using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Audio-LM layer-split verification for CosyVoice 2 (the second Qwen2Model-backed audio LM to get the
/// sharding wire-up, after YuE Stage-1 — see <see cref="HartsyInference.Diffusion.Tests.YueLmShardingEngineTests"/>
/// for the reference pattern this mirrors). Drives a REAL zero-shot synthesis through the full
/// <c>InferenceEngine</c> → <c>SpeechService</c> → <c>CosyVoiceModel</c> → <c>CosyVoicePipeline</c> path with
/// <see cref="PlacementConfig.ShardDevices"/> set: the Qwen2.5-0.5B backbone is layer-split across both cards via
/// <c>PlacementPlanner.LlmSplitPlan</c> + <c>LlmPlacement</c>, the per-stage asymmetric preload pools VRAM (both
/// cards' usage should rise), and a coherent WAV comes out — verified by a Whisper STT content-word-recall gate on
/// the SPOKEN (not sung) output text.
///
/// <para>Unlike YuE's 7B Stage-1 (which needed sharding to afford checkpoint-precision bf16), CosyVoice's
/// Qwen2.5-0.5B backbone is tiny (~1 GB) — the point of this test is proving the placement plumbing pools VRAM and
/// produces byte-for-byte-plausible audio, not that sharding was NECESSARY to fit the model. Report the real
/// numbers honestly; a small VRAM rise on both cards is the expected, correct result here, not a red flag.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class CosyVoiceLmShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public CosyVoiceLmShardingEngineTests(ITestOutputHelper output) => _output = output;

    /// <summary>The canonical whisper.cpp <c>jfk.wav</c> sample, vendored in this repo for the Silero-VAD reference
    /// fixtures (16 kHz mono, 11.0s) — verified elsewhere (<c>ZipVoiceE2eTests</c>) to be the exact whisper.cpp
    /// clip, so its transcript below is not a guess.</summary>
    private static string JfkReferencePath() =>
        Path.Combine(RepoPaths.RepoRoot(), "tests", "python-reference", "silerovad_reference", "jfk.wav");

    private const string JfkTranscript =
        "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";

    /// <summary>Target text for the CLONED voice to speak — deliberately NOT the reference transcript, so the
    /// Whisper recall gate below proves the LM actually generated new speech tokens for new text rather than just
    /// echoing (or partially replaying) the reference clip.</summary>
    private const string TargetText = "Good morning, the weather today is bright and sunny across the valley.";
    private static readonly string[] TargetContentWords = ["morning", "weather", "bright", "valley"];

    [Fact]
    public async Task LmSharding_RealEngine_QwenBackbonePooledAcrossGpus_ProducesIntelligibleSpeech()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }

        string llmPath = Path.Combine(RepoPaths.ModelsRoot(), "audio", "tts", "FunAudioLLM--CosyVoice2-0.5B", "llm.pt");
        string flowPath = Path.Combine(RepoPaths.ModelsRoot(), "audio", "tts", "FunAudioLLM--CosyVoice2-0.5B", "flow.pt");
        string hiftPath = Path.Combine(RepoPaths.ModelsRoot(), "audio", "tts", "FunAudioLLM--CosyVoice2-0.5B", "hift.pt");
        string s3genPath = Path.Combine(RepoPaths.ModelsRoot(), "audio", "tts", "ResembleAI--chatterbox", "s3gen.safetensors");
        string jfkPath = JfkReferencePath();
        if (!RealWeightGate.Require(_output.WriteLine, llmPath, flowPath, hiftPath, s3genPath, jfkPath)) return;

        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(CosyVoiceLmShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        // The Qwen2.5-0.5B backbone is tiny — this gate is just "the box has a working, non-starved CUDA context
        // on both cards", not a tight-fit VRAM budget like YuE's 7B bf16 test.
        long total0;
        long total1;
        using (CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir))
        using (CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir))
        {
            (nuint free0, nuint totalBytes0) = probe0.Context.GetMemoryInfo();
            (nuint free1, nuint totalBytes1) = probe1.Context.GetMemoryInfo();
            total0 = (long)totalBytes0;
            total1 = (long)totalBytes1;
            double freeGb0 = free0 / (1024.0 * 1024.0 * 1024.0), freeGb1 = free1 / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"Free VRAM: ordinal 0 = {freeGb0:F2} GB, ordinal 1 = {freeGb1:F2} GB.");
            if (freeGb0 < 2.0 || freeGb1 < 2.0)
            {
                _output.WriteLine("SKIPPED: insufficient free VRAM for a two-card CUDA context.");
                return;
            }
        }

        ModelSpec spec = ModelResolver.Resolve("cosyvoice", modelPathArg: null, Modality.Speech);
        SpeechRequest request = new SpeechRequest
        {
            Text = TargetText,
            RefText = JfkTranscript,
            Reference = new AudioClip { Data = await File.ReadAllBytesAsync(jfkPath), Format = "wav" },
            Seed = 7,
        };

        // Per-card used-VRAM peaks sampled during the generation are the pooling evidence: BOTH cards must rise
        // above their idle baseline (a replicated or single-card load only raises one) — same shape as
        // YueLmShardingEngineTests, but with a much smaller expected rise given the 0.5B backbone.
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
                    try { await Task.Delay(500, samplerCts.Token); } catch (OperationCanceledException) { }
                }
            })
            : Task.CompletedTask;

        Stopwatch sw = Stopwatch.StartNew();
        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"] };
        AudioResult result;
        using (InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            result = await engine.Speech.SynthesizeAsync(spec, request);
        }
        samplerCts.Cancel();
        await sampler;
        _output.WriteLine($"Generated {result.DurationSeconds:F1}s @ {result.SampleRate} Hz in {sw.Elapsed.TotalSeconds:F1}s.");

        Assert.NotNull(result.Data);
        Assert.True(result.Data.Length > 4000, $"WAV unexpectedly small ({result.Data.Length} bytes).");
        Assert.True(result.DurationSeconds > 0.5, $"duration {result.DurationSeconds:F2}s — the LM produced no usable speech tokens.");
        // The runner cache key carries the resolved placement — proves the sharded path actually engaged.
        Assert.Contains("shard=", result.Meta["model"]);

        if (baselineMib.Length >= 2)
        {
            (int smi0, int smi1) = MapSmiRowsToOrdinals(total0, total1, baselineMib.Length);
            long rise0 = peakMib[smi0] - baselineMib[smi0], rise1 = peakMib[smi1] - baselineMib[smi1];
            _output.WriteLine($"Peak VRAM rise during generation: ordinal 0 +{rise0} MiB, ordinal 1 +{rise1} MiB.");
            // Floor: the smaller stage is 7/24 of the ~1 GB bf16 backbone (~290 MiB) plus a fresh CUDA context and
            // KV — real run measured +1040/+2270 MiB (2026-08-05 benchmark doc); a bare context alone
            // (~200-500 MiB, i.e. a silent one-card fallback) cannot reach 600.
            Assert.True(Math.Min(rise0, rise1) > 600,
                $"expected BOTH cards to hold part of the Qwen2.5-0.5B backbone (pooling), got rises {rise0}/{rise1} MiB (ordinal 0/1) — " +
                "check the per-stage asymmetric preload in CosyVoicePipeline.PreloadWeights.");
        }
        else
        {
            _output.WriteLine("nvidia-smi unavailable — VRAM pooling assertion skipped (audio + cache-key checks still ran).");
        }

        await AssertSpeechIsIntelligible(result, TargetContentWords);
    }

    /// <summary>Whisper content-word-recall gate, same harness shape as
    /// <c>YueLmShardingEngineTests.AssertLyricsAreIntelligible</c>. Spoken zero-shot TTS is measurably easier ASR
    /// than sung vocals, so the bar here is intentionally higher than YuE's 50% sung-lyrics line — validated by a
    /// real run on this box (2026-08-05, <c>benchmarks/results/2026-08-05_multigpu_speeds.md</c>: Whisper heard the
    /// target sentence verbatim, 4/4 content-word recall). Loosen it only if a genuine real-weight run legitimately
    /// lands lower.</summary>
    private async Task AssertSpeechIsIntelligible(AudioResult result, string[] targetWords)
    {
        const string whisperRepo = "openai/whisper-medium";
        string whisperDir = AudioModelCache.GetRepoDirectory(whisperRepo, "stt");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(whisperDir, "model.safetensors"))) return;

        string wavPath = Path.Combine(Path.GetTempPath(), $"cosyvoice_lm_sharding_{Guid.NewGuid():N}.wav");
        try
        {
            await File.WriteAllBytesAsync(wavPath, result.Data);
            using CpuBackend backend = new();
            using WhisperPipeline stt = await WhisperPipeline.LoadAsync(whisperRepo);
            string heard = stt.TranscribeWav(backend, wavPath,
                new WhisperOptions { Language = "en", Translate = false, WithTimestamps = false }).Trim();
            _output.WriteLine($"Whisper heard: \"{heard}\"");

            string lower = heard.ToLowerInvariant();
            string[] hits = [.. targetWords.Where(w => lower.Contains(w.ToLowerInvariant()))];
            double recall = hits.Length / (double)targetWords.Length;
            _output.WriteLine($"Content-word recall: {hits.Length}/{targetWords.Length} ({recall:P0}) — {string.Join(",", hits)}");
            Assert.True(recall >= 0.75,
                $"Whisper recall {recall:P0} ({hits.Length}/{targetWords.Length}) on \"{heard}\" — spoken zero-shot "
                + "TTS should be near-word-perfect; check the sharded Qwen2 backbone didn't silently corrupt the "
                + "speech-token stream.");
        }
        finally
        {
            File.Delete(wavPath);
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
}
