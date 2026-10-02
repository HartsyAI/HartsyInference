using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Replays the AudioLab TTS sweep that left Dia out of VRAM and Orpheus streaming its weights: eight models
/// loaded in a row through <see cref="InferenceEngine.Speech"/> on one engine, so each switch decides eviction on a
/// card the earlier ones filled. Records, per model, free VRAM before it, the outcome, wall time, the pipeline's own
/// generation time where it logs one, and a digest of the audio; then runs Orpheus again on a freed card as its
/// free-card reference, and transcribes the sweep's Dia and Orpheus output with Whisper medium.en. It asserts nothing:
/// the same file runs on the branch and on its base, and the two reports are compared.
///
/// <para>Opt-in (<c>HARTSY_AUDIO_SWEEP_PROBE=1</c>): roughly 30 GB of weights through one GPU, so it runs only under
/// this repo's swarm-quiet-window and lock protocol. <c>HARTSY_AUDIO_SWEEP_OUT</c> names the folder the WAVs and the
/// report land in. Never downloads: a missing checkpoint skips.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
[Trait("Category", "Slow")]
public sealed class AudioSweepEvictionProbeTests
{
    private const string GateEnvVar = "HARTSY_AUDIO_SWEEP_PROBE";
    private const string OutEnvVar = "HARTSY_AUDIO_SWEEP_OUT";
    private const int Seed = 42;
    private const string Line = "Hello, this is a short test of the audio sweep, running through the engine on a shared card.";
    private const string JfkText = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
    private const string DiaLine = "[S1] Hello there, this is a short test of the audio sweep running through the engine. "
        + "[S2] It sounds good to me, and the second speaker answers right away. [S1] Great, thanks for checking.";

    private static readonly Regex GenerationLine = new(@"^(Orpheus|Dia): .* in (\d+)ms\.$", RegexOptions.Compiled);

    /// <summary>The sweep order, with the checkpoint file each model's loader opens (relative to the audio cache root),
    /// which is what decides whether the probe can run at all.</summary>
    private static readonly (string Id, string Weights, bool NeedsReference)[] Sweep =
    [
        ("bark", "tts/suno--bark/pytorch_model.bin", false),
        ("sparktts", "tts/SparkAudio--Spark-TTS-0.5B/LLM/model.safetensors", false),
        ("cosyvoice", "tts/FunAudioLLM--CosyVoice2-0.5B/llm.pt", true),
        ("vibevoice", "tts/microsoft--VibeVoice-1.5B/model-00001-of-00003.safetensors", true),
        ("fishspeech", "tts/fishaudio--fish-speech-1.5/model.pth", false),
        ("f5", "tts/SWivid--F5-TTS/F5TTS_v1_Base/model_1250000.safetensors", true),
        ("dia", "tts/nari-labs--Dia-1.6B-0626/pytorch_model.bin", false),
        ("orpheus", "tts/unsloth--orpheus-3b-0.1-ft/model.safetensors", false),
    ];

    private readonly ITestOutputHelper _out;

    public AudioSweepEvictionProbeTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task TtsSweep_ThenDiaAndOrpheus_OnTheCardTheSweepFilled()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run (loads ~30 GB of audio weights onto one GPU — follow the "
                + "swarm-quiet-window and lock protocol first).");
            return;
        }
        string? unavailable = BackendGate.UnavailableReason("cuda");
        if (unavailable is not null)
        {
            _out.WriteLine($"SKIPPED: {unavailable}");
            return;
        }
        string jfk = Path.Combine(RepoPaths.RepoRoot(), "tests", "python-reference", "silerovad_reference", "jfk.wav");
        string whisperWeights = Path.Combine(AudioModelCache.CacheRoot, "stt", "openai--whisper-medium.en", "model.safetensors");
        string[] required = [jfk, whisperWeights, .. Sweep.Select(step => Path.Combine(AudioModelCache.CacheRoot, step.Weights))];
        if (!RealWeightGate.Require(_out.WriteLine, required))
        {
            return;
        }
        string outDir = Environment.GetEnvironmentVariable(OutEnvVar)
            ?? Path.Combine(Path.GetTempPath(), $"hartsy-audio-sweep-{DateTime.UtcNow:yyyyMMddHHmmss}");
        Directory.CreateDirectory(outDir);
        AudioClip reference = new() { Data = await File.ReadAllBytesAsync(jfk), Format = "wav" };

        ConcurrentQueue<string> engineLines = new();
        Logs.SetLogger((level, message) =>
        {
            engineLines.Enqueue(message);
            Console.Error.WriteLine($"[{level}] {message}");
        });
        List<string> rows = ["model\tfree_before_gb\tstatus\twall_ms\tgen_ms\taudio_s\tsha256\tdetail"];
        Dictionary<string, byte[]> outputs = new(StringComparer.Ordinal);
        try
        {
            using InferenceEngine engine = new("cuda");
            (long startFree, long total) = engine.Backend.GetVramInfo();
            _out.WriteLine($"device {total >> 20} MB, {startFree >> 20} MB free before the sweep");
            foreach ((string id, _, bool needsReference) in Sweep)
            {
                SpeechRequest request = Request(id, needsReference ? reference : null);
                byte[]? wav = await RunStepAsync(engine, id, id, request, engineLines, rows, outDir);
                if (wav is not null)
                {
                    outputs[id] = wav;
                }
            }

            // Orpheus on a card holding nothing else of this engine's: the reference its sweep time is read against.
            engine.FreeMemory();
            await RunStepAsync(engine, "orpheus", "orpheus-free-card", Request("orpheus", null), engineLines, rows, outDir);

            engine.FreeMemory();
            ModelSpec whisper = ModelResolver.Resolve("whisper:openai/whisper-medium.en", modelPathArg: null, Modality.Transcribe);
            foreach (string id in new[] { "dia", "orpheus" })
            {
                if (!outputs.TryGetValue(id, out byte[]? wav))
                {
                    rows.Add($"whisper:{id}\t\tskipped\t\t\t\t\tno {id} output to transcribe");
                    continue;
                }
                TranscriptResult heard = await engine.Transcribe.RunAsync(whisper,
                    new AudioRequest { Audio = new AudioClip { Data = wav, Format = "wav" } });
                rows.Add($"whisper:{id}\t\tok\t\t\t\t\t{heard.Text.Trim()}");
            }
        }
        finally
        {
            Logs.SetLogger(null!);
            string report = Path.Combine(outDir, "report.tsv");
            await File.WriteAllLinesAsync(report, rows);
            foreach (string row in rows)
            {
                _out.WriteLine(row);
            }
            _out.WriteLine($"report: {report}");
        }
    }

    /// <summary>Synthesizes one sweep step, writing its WAV and a report row; returns the WAV, or null when it failed.</summary>
    private async Task<byte[]?> RunStepAsync(InferenceEngine engine, string id, string label, SpeechRequest request,
        ConcurrentQueue<string> engineLines, List<string> rows, string outDir)
    {
        ModelSpec spec = ModelResolver.Resolve(id, modelPathArg: null, Modality.Speech);
        (long free, _) = engine.Backend.GetVramInfo();
        string freeGb = (free / (double)(1L << 30)).ToString("F2", CultureInfo.InvariantCulture);
        engineLines.Clear();
        Stopwatch wall = Stopwatch.StartNew();
        try
        {
            AudioResult result = await engine.Speech.SynthesizeAsync(spec, request);
            wall.Stop();
            await File.WriteAllBytesAsync(Path.Combine(outDir, $"{label}.wav"), result.Data);
            string sha = Convert.ToHexString(SHA256.HashData(result.Data)).ToLowerInvariant();
            rows.Add($"{label}\t{freeGb}\tok\t{wall.ElapsedMilliseconds}\t{GenerationMs(engineLines)}\t"
                + $"{result.DurationSeconds.ToString("F2", CultureInfo.InvariantCulture)}\t{sha}\t");
            return result.Data;
        }
        catch (Exception ex)
        {
            // The probe's job is to record each model's outcome, failures included, and keep sweeping.
            wall.Stop();
            string message = ex.Message.Replace('\t', ' ').Replace('\n', ' ');
            rows.Add($"{label}\t{freeGb}\t{ex.GetType().Name}\t{wall.ElapsedMilliseconds}\t\t\t\t{message}");
            return null;
        }
    }

    /// <summary>The generation time the pipeline itself last logged for this step (Orpheus and Dia log one), else empty.</summary>
    private static string GenerationMs(IEnumerable<string> lines)
    {
        string found = "";
        foreach (string line in lines)
        {
            Match match = GenerationLine.Match(line);
            if (match.Success)
            {
                found = match.Groups[2].Value;
            }
        }
        return found;
    }

    private static SpeechRequest Request(string id, AudioClip? reference) => id switch
    {
        "dia" => new SpeechRequest { Text = DiaLine, Seed = Seed },
        _ => new SpeechRequest { Text = Line, Seed = Seed, Reference = reference, RefText = reference is null ? "" : JfkText },
    };
}
