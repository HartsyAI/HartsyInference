using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro per-sentence latency bench on the RTX 3060 — the number the phone-call plan's bring-up gate
/// (15-word sentence ≤ 250 ms, Whisper-verify content-word recall ≥ 80 %) is evaluated against. Opt-in with
/// <c>HARTSY_KOKORO_BENCH=1</c>; otherwise it returns early so the CPU lane and unattended GPU runs never pay for it.
///
/// <para>Device rule: opens <c>CudaBackend(ordinal)</c> with <c>HARTSY_KOKORO_BENCH_CUDA_ORDINAL</c> (default 1,
/// the 3060's engine ordinal on the reference box — CUDA enumerates fastest-first, the opposite of nvidia-smi) and
/// fails unless the device name contains "3060".</para>
///
/// <para>Method: 2 warm-up + 5 timed <see cref="KokoroPipeline.Synthesize"/> calls per 5 / 15 / 30-word sentence
/// (G2P outside the timer); median / p95 / min wall, RTF, and the backend's lazy device→host sync count for one
/// call (each one is a full stream drain — the residency metric this bench exists to drive to a handful).
/// Every case's float PCM is written to <c>HARTSY_KOKORO_BENCH_OUT_DIR</c> (raw <c>.f32</c> + 16-bit WAV) so a
/// later run can gate itself against it: with <c>HARTSY_KOKORO_BENCH_REF_DIR</c> set, the same-named <c>.f32</c>
/// there is compared (max-abs, waveform and log-spectral correlation, length; see <see cref="AudioParityMetrics"/>).
/// Whisper-verify uses the multilingual <c>openai/whisper-tiny</c> (the <c>.en</c> checkpoints are being fixed on
/// another branch). <c>HARTSY_KOKORO_BENCH_PROFILE=1</c> additionally runs profiled 15-word synths with the per-op
/// profiler (<c>diagnostics.profile</c> + <c>profileSync</c>) and dumps the table beside the PCM.
/// Tables go to the test output and, when <c>HARTSY_KOKORO_BENCH_OUT</c> names a file, are appended there.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class KokoroBenchTests
{
    private const string GateEnvVar = "HARTSY_KOKORO_BENCH";
    private const string OrdinalEnvVar = "HARTSY_KOKORO_BENCH_CUDA_ORDINAL";
    private const string OutEnvVar = "HARTSY_KOKORO_BENCH_OUT";
    private const string OutDirEnvVar = "HARTSY_KOKORO_BENCH_OUT_DIR";
    private const string RefDirEnvVar = "HARTSY_KOKORO_BENCH_REF_DIR";
    private const string ProfileEnvVar = "HARTSY_KOKORO_BENCH_PROFILE";
    private const string RequiredDeviceSubstring = "3060";
    private const int WarmRuns = 2;
    private const int TimedRuns = 5;
    private const int WhisperRate = 16_000;
    private const string WhisperTiny = "openai/whisper-tiny";
    private const string KokoroRepackRepo = "Hartsy/kokoro-82m-safetensors";
    private const string KokoroRepo = "hexgrad/Kokoro-82M";
    private const string KokoroVoice = "af_heart";

    private static readonly (int Words, string Text)[] Sentences =
    [
        (5, "Please hold while I check."),
        (15, "Thanks for calling, I can see your appointment is booked for Tuesday afternoon at three."),
        (30, "I have updated the delivery address on your order, the driver will call you when they are ten minutes "
            + "away, and you will receive a message with the tracking link."),
    ];

    private readonly ITestOutputHelper _out;

    public KokoroBenchTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Kokoro_Sentences_3060_Latency_And_Parity()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the Kokoro 3060 bench.");
            return;
        }
        string cmudict = Path.Combine(ModelsRoot(), "audio", "cmudict.dict");
        if (!RealWeightGate.Require(_out.WriteLine, KokoroFiles().Concat(WhisperFiles(WhisperTiny)).Concat([cmudict]).ToArray()))
        {
            return;
        }
        string? outDir = Environment.GetEnvironmentVariable(OutDirEnvVar);
        string? refDir = Environment.GetEnvironmentVariable(RefDirEnvVar);
        if (!string.IsNullOrEmpty(outDir))
        {
            Directory.CreateDirectory(outDir);
        }
        using IBackend backend = OpenBackend();

        EnglishG2P g2p = new EnglishG2P(cmudict);
        Stopwatch load = Stopwatch.StartNew();
        using KokoroPipeline kokoro = await KokoroPipeline.LoadAsync();
        using WhisperPipeline verify = await WhisperPipeline.LoadAsync(WhisperTiny);
        _out.WriteLine($"Kokoro + {WhisperTiny} loaded in {load.Elapsed.TotalSeconds:F1}s (excluded from every row)");
        int kokoroRate = kokoro.Config.SampleRate;
        Resampler toWhisper = Resampler.Create(kokoroRate, WhisperRate);
        WhisperOptions options = new WhisperOptions { Language = "en" };

        StringBuilder table = new StringBuilder();
        table.AppendLine($"### Kokoro af_heart per sentence — {backend.Capabilities.DeviceName}, in-process, G2P outside the timer");
        table.AppendLine();
        table.AppendLine("| Words | median ms | p95 ms | min ms | audio s | RTF | D2H syncs/call | sha256[:12] | vs ref max-abs | vs ref wave corr | vs ref log-spec corr | verify recall | transcript |");
        table.AppendLine("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|");

        foreach ((int words, string text) in Sentences)
        {
            Assert.Equal(words, text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length);
            string ipa = g2p.ToIpa(text);
            float[] wave = [];
            long syncs = 0;
            for (int i = 0; i < WarmRuns; i++)
            {
                wave = kokoro.Synthesize(backend, ipa, KokoroVoice, 1f);
            }
            List<double> samples = new List<double>(TimedRuns);
            for (int i = 0; i < TimedRuns; i++)
            {
                long before = backend.GetD2hSyncCount();
                Stopwatch sw = Stopwatch.StartNew();
                wave = kokoro.Synthesize(backend, ipa, KokoroVoice, 1f);
                sw.Stop();
                syncs = backend.GetD2hSyncCount() - before;
                samples.Add(sw.Elapsed.TotalSeconds);
            }
            Stats stats = Stats.Of(samples);
            double audioSeconds = wave.Length / (double)kokoroRate;
            string digest = AudioParityMetrics.Sha256Hex(wave)[..12];
            string name = $"kokoro_{words}w";
            if (!string.IsNullOrEmpty(outDir))
            {
                File.WriteAllBytes(Path.Combine(outDir, name + ".f32"), MemoryMarshal.AsBytes<float>(wave).ToArray());
                WavFile.WriteMono16(Path.Combine(outDir, name + ".wav"), wave, kokoroRate);
            }
            string maxAbs = "—", corr = "—", specCorr = "—";
            if (!string.IsNullOrEmpty(refDir) && File.Exists(Path.Combine(refDir, name + ".f32")))
            {
                float[] reference = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(refDir, name + ".f32"))).ToArray();
                (double ma, double c) = AudioParityMetrics.Compare(reference, wave);
                maxAbs = reference.Length == wave.Length ? ma.ToString("E2", CultureInfo.InvariantCulture)
                    : $"len {reference.Length}→{wave.Length}";
                corr = c.ToString("F6", CultureInfo.InvariantCulture);
                specCorr = AudioParityMetrics.LogSpectralCorrelation(reference, wave).ToString("F6", CultureInfo.InvariantCulture);
            }
            string heard = verify.TranscribeAudio(backend, toWhisper.Resample(wave), WhisperRate, options);
            double recall = AudioParityMetrics.ContentWordRecall(text, heard);
            table.AppendLine($"| {words} | {Ms(stats.Median)} | {Ms(stats.P95)} | {Ms(stats.Min)} | {audioSeconds:F2} | "
                + $"{stats.Median / audioSeconds:F3} | {syncs} | `{digest}` | {maxAbs} | {corr} | {specCorr} | {recall:P0} | {AudioParityMetrics.Cell(heard)} |");
            _out.WriteLine($"Kokoro {words}w: median {Ms(stats.Median)} ms p95 {Ms(stats.P95)} min {Ms(stats.Min)} | "
                + $"{audioSeconds:F2}s audio | {syncs} D2H syncs | sha {digest} | ref max-abs {maxAbs} corr {corr} log-spec corr {specCorr} | "
                + $"recall {recall:P0} | {heard.Trim()}");
        }
        Emit(table);

        if (Environment.GetEnvironmentVariable(ProfileEnvVar) == "1")
        {
            RunProfiled(backend, kokoro, g2p.ToIpa(Sentences[1].Text), outDir);
        }
    }

    /// <summary>Profiled 15-word synths with the per-op profiler syncing after every op (true per-op cost,
    /// serialized) and the pipeline's stage timer; both land in the test output. The engine log is captured into a
    /// queue while it runs — never written straight to <see cref="_out"/> from the logger, which outlives the test.</summary>
    private void RunProfiled(IBackend backend, KokoroPipeline kokoro, string ipa, string? outDir)
    {
        System.Collections.Concurrent.ConcurrentQueue<string> captured = new();
        Logs.SetLogger((level, message) => captured.Enqueue($"[{level}] {message}"));
        KnobStore.Set(EngineKnobs.Profile, true);
        KnobStore.Set(EngineKnobs.ProfileSync, true);
        try
        {
            // First profiled call warms the profile-mode code paths; the following three are reported so a
            // stage whose time moves between calls shows as such.
            kokoro.Synthesize(backend, ipa, KokoroVoice, 1f);
            captured.Clear();
            NvtxRange.ResetProfile();
            string profilePath = Path.Combine(string.IsNullOrEmpty(outDir) ? Path.GetTempPath() : outDir, "kokoro_15w_profile.txt");
            for (int i = 0; i < 3; i++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                kokoro.Synthesize(backend, ipa, KokoroVoice, 1f);
                sw.Stop();
                _out.WriteLine($"profiled 15w synth #{i + 1} (profileSync on, serialized): {Ms(sw.Elapsed.TotalSeconds)} ms");
                while (captured.TryDequeue(out string? line))
                {
                    _out.WriteLine(line);
                }
            }
            NvtxRange.DumpProfile(profilePath);
            _out.WriteLine($"per-op table (3 synths) at {profilePath}");
            if (File.Exists(profilePath))
            {
                _out.WriteLine(File.ReadAllText(profilePath));
            }
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Profile);
            KnobStore.Clear(EngineKnobs.ProfileSync);
            Logs.SetLogger((level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
    }

    /// <summary>Opens the CUDA device named by <see cref="OrdinalEnvVar"/> (default 1) and asserts it is the 3060.</summary>
    private IBackend OpenBackend()
    {
        string? ordinalText = Environment.GetEnvironmentVariable(OrdinalEnvVar);
        int ordinal = string.IsNullOrEmpty(ordinalText) ? 1 : int.Parse(ordinalText, CultureInfo.InvariantCulture);
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptx = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptx is null, "no compiled PTX directory beside the tests or in the repo");
        CudaBackend backend = new CudaBackend(ordinal, ptx);
        string device = backend.Capabilities.DeviceName;
        _out.WriteLine($"CUDA ordinal {ordinal}: {device}; models root {ModelsRoot()}; audio cache {AudioModelCache.CacheRoot}");
        if (!device.Contains(RequiredDeviceSubstring, StringComparison.Ordinal))
        {
            backend.Dispose();
            Assert.Fail($"ordinal {ordinal} is '{device}', not a {RequiredDeviceSubstring}. Set {OrdinalEnvVar} to the 3060's engine ordinal.");
        }
        return backend;
    }

    private static string ModelsRoot() =>
        EngineKnobs.ModelsRoot.Value is { Length: > 0 } root ? Path.GetFullPath(root) : TestPaths.ModelsDir;

    private static string[] WhisperFiles(string repo)
    {
        string dir = AudioModelCache.GetRepoDirectory(repo, "stt");
        return WhisperPipeline.ModelFiles.Where(f => f.Required).Select(f => Path.Combine(dir, f.Name)).ToArray();
    }

    private static string[] KokoroFiles()
    {
        string repack = AudioModelCache.GetRepoDirectory(KokoroRepackRepo, "tts");
        string canonical = AudioModelCache.GetRepoDirectory(KokoroRepo, "tts");
        return
        [
            Path.Combine(repack, "kokoro-82m.safetensors"),
            Path.Combine(canonical, "config.json"),
            Path.Combine(canonical, "voices", KokoroVoice + ".bin"),
        ];
    }

    private static string Ms(double seconds) => AudioParityMetrics.Ms(seconds);

    private void Emit(StringBuilder table)
    {
        string text = table.ToString();
        _out.WriteLine(text);
        string? outPath = Environment.GetEnvironmentVariable(OutEnvVar);
        if (!string.IsNullOrEmpty(outPath))
        {
            File.AppendAllText(outPath, text + Environment.NewLine);
            _out.WriteLine($"appended to {outPath}");
        }
    }

    private readonly record struct Stats(double Median, double P95, double Min)
    {
        public static Stats Of(List<double> samples)
        {
            List<double> sorted = samples.OrderBy(s => s).ToList();
            return new Stats(Percentile(sorted, 0.5), Percentile(sorted, 0.95), sorted[0]);
        }

        private static double Percentile(List<double> sorted, double p)
        {
            if (sorted.Count == 1)
            {
                return sorted[0];
            }
            double position = p * (sorted.Count - 1);
            int low = (int)Math.Floor(position);
            int high = Math.Min(low + 1, sorted.Count - 1);
            return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
        }
    }
}
