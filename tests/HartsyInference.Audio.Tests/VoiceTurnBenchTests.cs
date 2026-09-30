using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Phase 1 voice-turn latency probes: the in-process, real-weight numbers on the 3060 that the
/// phone-call plan's model bring-up gates are evaluated against (Whisper small.en per-utterance ≤ 350 ms,
/// Kokoro 15-word sentence ≤ 250 ms, narrowband recall ≥ 16 kHz baseline − 10 pts, Kokoro Whisper-verify
/// recall ≥ 80 %). Opt-in with <c>HARTSY_VOICE_BENCH=1</c>; otherwise every probe returns early so the CPU lane
/// and unattended GPU runs never pay for it.
///
/// <para>Device rule: opens <c>CudaBackend(ordinal)</c> with the ordinal from <c>HARTSY_VOICE_BENCH_CUDA_ORDINAL</c>
/// (default 1) and FAILS unless that device reports a name containing "3060" — engine ordinal 0 is the 4090 on
/// the reference box (fastest-first), the opposite of the nvidia-smi index, and <see cref="SttBenchTests"/>'s
/// hard-coded ordinal 0 is exactly the trap this guards against.</para>
///
/// <para>Path rule: every asset is resolved the way the engine resolves it — Whisper and Kokoro through
/// <see cref="AudioModelCache"/>, the CMU dictionary and RNNoise weights under the engine's models root
/// (<c>paths.modelsRoot</c> knob, else the test-lane <see cref="TestPaths.ModelsDir"/>), never a hard-coded home
/// path — and the exact file each loader opens is checked BEFORE loading, because <c>LoadAsync</c> downloads on a
/// miss and a benchmark must never start a download. Mandatory assets go through
/// <see cref="RealWeightGate.Require"/>; medium.en and RNNoise are optional rows that log SKIPPED when absent.</para>
///
/// <para>Method: 2 warm-up + 5 timed calls per case; median / p95 (linear interpolation between order
/// statistics) / min wall-clock of the pipeline call only (model load excluded). Recall is content-word recall:
/// both texts lower-cased, punctuation stripped, split on whitespace, stop words
/// (a, an, and, the, of, for, to, so, my, your, you, what, can, do, not, is, it, in, on) removed from the
/// reference; recall = |reference content words present in the hypothesis| / |reference content words|. The
/// narrowband variant is the engine's own polyphase <see cref="Resampler"/> 16 k → 8 k → 16 k (not the pure-Python
/// filter Phase 0 uses, so the two phases' narrowband rows are not directly comparable). Whisper pads every clip
/// to 30 s, so the recall probes run on the full 11 s JFK clip at no extra cost. Tables go to the test output and,
/// when <c>HARTSY_VOICE_BENCH_OUT</c> names a file, are appended there as markdown.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class VoiceTurnBenchTests
{
    private const string GateEnvVar = "HARTSY_VOICE_BENCH";
    private const string OrdinalEnvVar = "HARTSY_VOICE_BENCH_CUDA_ORDINAL";
    private const string OutEnvVar = "HARTSY_VOICE_BENCH_OUT";
    private const string RequiredDeviceSubstring = "3060";
    private const int WarmRuns = 2;
    private const int TimedRuns = 5;
    private const int SampleRate = 16_000;
    private const string WhisperSmallEn = "openai/whisper-small.en";
    private const string WhisperMediumEn = "openai/whisper-medium.en";
    private const string KokoroRepackRepo = "Hartsy/kokoro-82m-safetensors";
    private const string KokoroRepo = "hexgrad/Kokoro-82M";
    private const string KokoroVoice = "af_heart";
    private const string JfkTranscript =
        "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

    private static readonly int[] SliceSeconds = [2, 5, 10];

    private static readonly (int Words, string Text)[] Sentences =
    [
        (5, "Please hold while I check that."),
        (15, "Thanks for calling, I can see your appointment is booked for Tuesday at three in the afternoon."),
        (30, "I have updated the delivery address on your order, the driver will call you when they are ten minutes "
            + "away, and you will receive a text message with the tracking link shortly."),
    ];

    private static readonly HashSet<string> StopWords = new(StringComparer.Ordinal)
    {
        "a", "an", "and", "the", "of", "for", "to", "so", "my", "your", "you", "what", "can", "do", "not", "is", "it", "in", "on",
    };

    private readonly ITestOutputHelper _out;

    public VoiceTurnBenchTests(ITestOutputHelper output) => _out = output;

    /// <summary>Probe (a): Whisper small.en (medium.en as an optional row) on JFK slices of 2 / 5 / 10 s at
    /// 16 kHz and after the narrowband round-trip.</summary>
    [Fact]
    public async Task ProbeA_WhisperSmallEn_Slices_16k_And_Narrowband()
    {
        if (!Gated()) return;
        string jfk = JfkPath();
        if (!RealWeightGate.Require(_out.WriteLine, WhisperFiles(WhisperSmallEn).Concat([jfk]).ToArray())) return;
        using IBackend backend = OpenBackend();

        float[] audio16k = LoadJfk16k(jfk);
        float[] narrowband = NarrowbandRoundTrip(audio16k);
        WhisperOptions options = new WhisperOptions { Language = null };

        StringBuilder table = new StringBuilder();
        table.AppendLine("### Probe A — Whisper `.en` per-utterance wall (3060, in-process, 30 s pad)");
        table.AppendLine();
        table.AppendLine("| Model | Slice | Variant | median ms | p95 ms | min ms | transcript |");
        table.AppendLine("|---|---:|---|---:|---:|---:|---|");

        List<string> models = [WhisperSmallEn];
        if (WhisperFiles(WhisperMediumEn).All(File.Exists))
        {
            models.Add(WhisperMediumEn);
        }
        else
        {
            _out.WriteLine($"SKIPPED optional row: {WhisperMediumEn} not cached under {AudioModelCache.GetRepoDirectory(WhisperMediumEn, "stt")}");
            table.AppendLine($"| {WhisperMediumEn} | — | — | SKIPPED | — | — | weights not cached |");
        }

        foreach (string repo in models)
        {
            Stopwatch load = Stopwatch.StartNew();
            using WhisperPipeline whisper = await WhisperPipeline.LoadAsync(repo);
            _out.WriteLine($"{repo} loaded in {load.Elapsed.TotalSeconds:F1}s (excluded from every row)");
            foreach (int seconds in SliceSeconds)
            {
                foreach ((string variant, float[] source) in new[] { ("16k", audio16k), ("narrowband 8k→16k", narrowband) })
                {
                    float[] slice = source[..Math.Min(source.Length, seconds * SampleRate)];
                    (Stats stats, string text) = Measure(() => whisper.TranscribeAudio(backend, slice, SampleRate, options));
                    table.AppendLine($"| {repo} | {seconds} s | {variant} | {Ms(stats.Median)} | {Ms(stats.P95)} | {Ms(stats.Min)} | {Cell(text)} |");
                    _out.WriteLine($"{repo} {seconds}s {variant}: median {Ms(stats.Median)} ms p95 {Ms(stats.P95)} min {Ms(stats.Min)} | {text.Trim()}");
                }
            }
        }
        Emit(table);
    }

    /// <summary>Probe (b): Kokoro <c>af_heart</c> per 5 / 15 / 30-word sentence through <see cref="EnglishG2P"/>,
    /// plus Whisper-verify recall of each synthesized sentence with small.en.</summary>
    [Fact]
    public async Task ProbeB_Kokoro_Sentences_Via_EnglishG2P()
    {
        if (!Gated()) return;
        string cmudict = CmudictPath();
        if (!RealWeightGate.Require(_out.WriteLine, KokoroFiles().Concat(WhisperFiles(WhisperSmallEn)).Concat([cmudict]).ToArray())) return;
        using IBackend backend = OpenBackend();

        EnglishG2P g2p = new EnglishG2P(cmudict);
        _out.WriteLine($"cmudict: {cmudict} ({g2p.WordCount} entries)");
        Stopwatch load = Stopwatch.StartNew();
        using KokoroPipeline kokoro = await KokoroPipeline.LoadAsync();
        using WhisperPipeline verify = await WhisperPipeline.LoadAsync(WhisperSmallEn);
        _out.WriteLine($"Kokoro + {WhisperSmallEn} loaded in {load.Elapsed.TotalSeconds:F1}s (excluded from every row)");
        int kokoroRate = kokoro.Config.SampleRate;
        Resampler toWhisper = Resampler.Create(kokoroRate, SampleRate);
        WhisperOptions options = new WhisperOptions { Language = null };

        StringBuilder table = new StringBuilder();
        table.AppendLine("### Probe B — Kokoro af_heart per sentence (3060, in-process; G2P timed separately)");
        table.AppendLine();
        table.AppendLine("| Words | synth median ms | p95 ms | min ms | audio s | RTF (median/audio) | G2P ms | Whisper-verify recall | verify transcript |");
        table.AppendLine("|---:|---:|---:|---:|---:|---:|---:|---:|---|");

        foreach ((int words, string text) in Sentences)
        {
            Assert.Equal(words, text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length);
            Stopwatch g2pTimer = Stopwatch.StartNew();
            string ipa = g2p.ToIpa(text);
            double g2pMs = g2pTimer.Elapsed.TotalMilliseconds;
            (Stats stats, float[] wave) = Measure(() => kokoro.Synthesize(backend, ipa, KokoroVoice, 1f));
            double audioSeconds = wave.Length / (double)kokoroRate;
            string heard = verify.TranscribeAudio(backend, toWhisper.Resample(wave), SampleRate, options);
            double recall = ContentWordRecall(text, heard);
            table.AppendLine($"| {words} | {Ms(stats.Median)} | {Ms(stats.P95)} | {Ms(stats.Min)} | {audioSeconds:F2} | "
                + $"{stats.Median / audioSeconds:F3} | {g2pMs:F1} | {recall:P0} | {Cell(heard)} |");
            _out.WriteLine($"Kokoro {words}w: median {Ms(stats.Median)} ms p95 {Ms(stats.P95)} min {Ms(stats.Min)} | "
                + $"{audioSeconds:F2}s audio | G2P {g2pMs:F1} ms | verify recall {recall:P0} | {heard.Trim()}");
        }
        Emit(table);
    }

    /// <summary>Probe (c): content-word recall of the full JFK clip — 16 kHz baseline, narrowband, and narrowband
    /// after RNNoise (48 kHz hop via <see cref="RnnoiseStream"/>) — with small.en; the RNNoise row is optional.</summary>
    [Fact]
    public async Task ProbeC_Narrowband_Recall_With_And_Without_Rnnoise()
    {
        if (!Gated()) return;
        string jfk = JfkPath();
        if (!RealWeightGate.Require(_out.WriteLine, WhisperFiles(WhisperSmallEn).Concat([jfk]).ToArray())) return;
        using IBackend backend = OpenBackend();

        float[] audio16k = LoadJfk16k(jfk);
        float[] narrowband = NarrowbandRoundTrip(audio16k);
        using WhisperPipeline whisper = await WhisperPipeline.LoadAsync(WhisperSmallEn);
        WhisperOptions options = new WhisperOptions { Language = null };

        StringBuilder table = new StringBuilder();
        table.AppendLine("### Probe C — narrowband content-word recall vs the 16 kHz baseline (small.en, full 11 s clip)");
        table.AppendLine();
        table.AppendLine("| Input | recall | Δ vs 16k (pts) | STT median ms | p95 ms | front-end median ms | transcript |");
        table.AppendLine("|---|---:|---:|---:|---:|---:|---|");

        (Stats baseStats, string baseText) = Measure(() => whisper.TranscribeAudio(backend, audio16k, SampleRate, options));
        double baseRecall = ContentWordRecall(JfkTranscript, baseText);
        table.AppendLine($"| 16 kHz baseline | {baseRecall:P0} | 0 | {Ms(baseStats.Median)} | {Ms(baseStats.P95)} | — | {Cell(baseText)} |");

        (Stats nbStats, string nbText) = Measure(() => whisper.TranscribeAudio(backend, narrowband, SampleRate, options));
        double nbRecall = ContentWordRecall(JfkTranscript, nbText);
        table.AppendLine($"| narrowband 8k→16k | {nbRecall:P0} | {(nbRecall - baseRecall) * 100:+0;-0;0} | {Ms(nbStats.Median)} | {Ms(nbStats.P95)} | — | {Cell(nbText)} |");
        _out.WriteLine($"16k recall {baseRecall:P0} | narrowband recall {nbRecall:P0} (Δ {(nbRecall - baseRecall) * 100:+0;-0;0} pts)");

        string rnnoise = RnnoisePath();
        if (File.Exists(rnnoise))
        {
            RnnoiseWeights weights = new RnnoiseWeights();
            using (SafeTensorsLoader loader = new SafeTensorsLoader())
            {
                loader.Load(rnnoise);
                weights.Load(loader.GetAllTensors());
            }
            using (weights)
            {
                (Stats dnStats, float[] denoised) = Measure(() => Denoise(backend, weights, narrowband));
                (Stats dnSttStats, string dnText) = Measure(() => whisper.TranscribeAudio(backend, denoised, SampleRate, options));
                double dnRecall = ContentWordRecall(JfkTranscript, dnText);
                table.AppendLine($"| narrowband + RNNoise | {dnRecall:P0} | {(dnRecall - baseRecall) * 100:+0;-0;0} | {Ms(dnSttStats.Median)} | "
                    + $"{Ms(dnSttStats.P95)} | {Ms(dnStats.Median)} | {Cell(dnText)} |");
                _out.WriteLine($"narrowband + RNNoise recall {dnRecall:P0} (Δ {(dnRecall - baseRecall) * 100:+0;-0;0} pts); RNNoise pass median {Ms(dnStats.Median)} ms");
            }
        }
        else
        {
            _out.WriteLine($"SKIPPED optional row: RNNoise weights not found at {rnnoise}");
            table.AppendLine($"| narrowband + RNNoise | SKIPPED | — | — | — | — | weights not found at `{rnnoise}` |");
        }
        Emit(table);
    }

    private bool Gated()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) == "1")
        {
            return true;
        }
        _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the voice-turn bench probes.");
        return false;
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

    /// <summary>The engine's models root: the <c>paths.modelsRoot</c> knob when set (what <c>RepoPaths.ModelsRoot()</c>
    /// reads), else the test lane's <see cref="TestPaths.ModelsDir"/>.</summary>
    private static string ModelsRoot() =>
        EngineKnobs.ModelsRoot.Value is { Length: > 0 } root ? Path.GetFullPath(root) : TestPaths.ModelsDir;

    /// <summary>Same file the Kokoro descriptor opens: <c>{models}/audio/cmudict.dict</c>.</summary>
    private static string CmudictPath() => Path.Combine(ModelsRoot(), "audio", "cmudict.dict");

    /// <summary>Same file <c>WakeModelSet.LoadDenoiser</c> opens: <c>{models}/audio/wake/denoise/rnnoise.safetensors</c>.</summary>
    private static string RnnoisePath() => Path.Combine(ModelsRoot(), "audio", "wake", "denoise", "rnnoise.safetensors");

    private static string JfkPath() =>
        Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");

    /// <summary>The files <see cref="WhisperPipeline.LoadAsync"/> fetches, resolved through the cache; checking them
    /// first is what keeps the load from downloading.</summary>
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

    private static float[] LoadJfk16k(string path)
    {
        WavFile.DecodedAudio decoded = WavFile.Read(path);
        float[] mono = decoded.ToMono();
        return decoded.SampleRate == SampleRate ? mono : Resampler.Create(decoded.SampleRate, SampleRate).Resample(mono);
    }

    private static float[] NarrowbandRoundTrip(float[] audio16k)
    {
        float[] down = Resampler.Create(SampleRate, 8_000).Resample(audio16k);
        float[] up = Resampler.Create(8_000, SampleRate).Resample(down);
        return up.Length > audio16k.Length ? up[..audio16k.Length] : up;
    }

    /// <summary>RNNoise over a whole clip: int16-scaled in and out, latency flushed with zeros, trailing partial frame dropped.</summary>
    private static float[] Denoise(IBackend backend, RnnoiseWeights weights, float[] audio16k)
    {
        using RnnoiseStream stream = new RnnoiseStream(weights, SampleRate);
        float[] scaled = new float[audio16k.Length + stream.LatencySamples];
        for (int i = 0; i < audio16k.Length; i++)
        {
            scaled[i] = audio16k[i] * 32768f;
        }
        float[] output = new float[scaled.Length + stream.FrameSize];
        int written = stream.Process(backend, scaled, output);
        int start = Math.Min(stream.LatencySamples, written);
        int count = Math.Min(audio16k.Length, written - start);
        float[] result = new float[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = output[start + i] / 32768f;
        }
        return result;
    }

    private static (Stats Stats, T Last) Measure<T>(Func<T> run)
    {
        T last = default!;
        for (int i = 0; i < WarmRuns; i++)
        {
            last = run();
        }
        List<double> samples = new List<double>(TimedRuns);
        for (int i = 0; i < TimedRuns; i++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            last = run();
            sw.Stop();
            samples.Add(sw.Elapsed.TotalSeconds);
        }
        return (Stats.Of(samples), last);
    }

    private static double ContentWordRecall(string reference, string hypothesis)
    {
        HashSet<string> hyp = new HashSet<string>(Words(hypothesis), StringComparer.Ordinal);
        List<string> content = Words(reference).Where(w => !StopWords.Contains(w)).Distinct().ToList();
        return content.Count == 0 ? 0 : content.Count(hyp.Contains) / (double)content.Count;
    }

    private static IEnumerable<string> Words(string text)
    {
        StringBuilder sb = new StringBuilder(text.Length);
        foreach (char c in text.ToLowerInvariant())
        {
            sb.Append(char.IsLetterOrDigit(c) || c == '\'' ? c : ' ');
        }
        return sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    private static string Ms(double seconds) => (seconds * 1000).ToString("F1", CultureInfo.InvariantCulture);

    private static string Cell(string text) => text.Trim().Replace("|", "\\|").Replace('\n', ' ');

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
