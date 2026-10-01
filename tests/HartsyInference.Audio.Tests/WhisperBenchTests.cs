using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Whisper per-utterance latency bench on the RTX 3060: the numbers the phone-call plan's STT bring-up gate
/// (small.en ≤ 350 ms per utterance for 2 / 5 / 10 s of speech, narrowband content-word recall ≥ the 16 kHz baseline
/// − 10 points on the full clip and on the slices of 5 s and longer, JFK 11/11 words) is evaluated against, plus
/// token-level regression evidence for every other Whisper checkpoint. Opt-in with <c>HARTSY_WHISPER_BENCH=1</c>;
/// otherwise it returns early so the CPU lane and unattended GPU runs never pay for it.
///
/// <para>The 2 s slice is timed but its recall is informational: the cut lands inside "Americans", and the reference
/// model itself drops the word's end on the narrowband samples there, so its recall measures the cut, not narrowband
/// robustness.</para>
///
/// <para>Device rule: opens <c>CudaBackend(ordinal)</c> with <c>HARTSY_WHISPER_BENCH_CUDA_ORDINAL</c> and fails unless
/// the device name contains "3060". Without the variable the ordinal is 0 when <c>CUDA_VISIBLE_DEVICES</c> names one
/// device, else 1 (the 3060's engine ordinal on the reference box, where CUDA enumerates fastest-first).</para>
///
/// <para>Cases match <c>VoiceTurnBenchTests</c> Probe A / C so the two line up: the JFK clip sliced to 2 / 5 / 10 s at
/// 16 kHz and after the engine's own polyphase <see cref="Resampler"/> round trip 16 k → 8 k → 16 k, decoded with
/// <c>Language = null</c> (ignored by <c>.en</c> checkpoints); 2 warm-up + 5 timed calls, median / p95 (linear
/// interpolation) / min wall of <see cref="WhisperPipeline.TranscribeAudio"/> only, and the backend's lazy
/// device→host sync count for one call. Recall is Probe C's content-word recall (same stop words). The full clip also
/// runs the 11-word check of <c>WhisperEnglishOnlyTests</c>, a timestamped decode and a
/// <see cref="WhisperStreamingPipeline"/> pass.</para>
///
/// <para>Models: <c>HARTSY_WHISPER_BENCH_MODELS</c> (comma-separated repo ids, default <c>openai/whisper-small.en</c>).
/// small.en gets the gate protocol above; every other model a regression protocol (full clip at 16 kHz and narrowband
/// plus the 5 s slice, 1 warm-up + 2 timed). Multilingual checkpoints decode with <c>Language = "en"</c>. Each case's
/// generated ids go to <c>HARTSY_WHISPER_BENCH_OUT_DIR</c> (<c>*.tokens</c>, <c>*.txt</c>); with
/// <c>HARTSY_WHISPER_BENCH_REF_DIR</c> set, the same-named files there are compared and the first divergence reported.
/// <c>HARTSY_WHISPER_BENCH_PROFILE=1</c> adds stage-timer runs (<c>diagnostics.profile</c>: stage wall and D2H syncs,
/// each stage closed by a device sync) and per-op runs (<c>diagnostics.profile</c> + <c>profileSync</c>) of small.en;
/// <c>HARTSY_WHISPER_BENCH_EXACT=1</c> runs everything at full F32 (no TF32 GEMMs). <c>HARTSY_WHISPER_BENCH_BACKEND=cpu</c>
/// runs on the CPU backend instead (token evidence without a GPU), <c>HARTSY_WHISPER_BENCH_GATE_MODEL</c> names the
/// model that gets the gate protocol, and <c>HARTSY_WHISPER_BENCH_RUNS=warm,timed</c> overrides the regression
/// protocol's call counts. Tables go to the test output
/// and, when <c>HARTSY_WHISPER_BENCH_OUT</c> names a file, are appended there.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class WhisperBenchTests
{
    private const string GateEnvVar = "HARTSY_WHISPER_BENCH";
    private const string OrdinalEnvVar = "HARTSY_WHISPER_BENCH_CUDA_ORDINAL";
    private const string ModelsEnvVar = "HARTSY_WHISPER_BENCH_MODELS";
    private const string OutEnvVar = "HARTSY_WHISPER_BENCH_OUT";
    private const string OutDirEnvVar = "HARTSY_WHISPER_BENCH_OUT_DIR";
    private const string RefDirEnvVar = "HARTSY_WHISPER_BENCH_REF_DIR";
    private const string ProfileEnvVar = "HARTSY_WHISPER_BENCH_PROFILE";
    private const string ExactEnvVar = "HARTSY_WHISPER_BENCH_EXACT";
    private const string BackendEnvVar = "HARTSY_WHISPER_BENCH_BACKEND";
    private const string RunsEnvVar = "HARTSY_WHISPER_BENCH_RUNS";
    private const string GateModelEnvVar = "HARTSY_WHISPER_BENCH_GATE_MODEL";
    private const string RequiredDeviceSubstring = "3060";
    private const string GateModel = "openai/whisper-small.en";
    private const int SampleRate = 16_000;
    private const int NarrowbandRate = 8_000;
    private const int GateWarmRuns = 2;
    private const int GateTimedRuns = 5;
    private const int RegressionWarmRuns = 1;
    private const int RegressionTimedRuns = 2;
    private const double GateMs = 350;
    // Narrowband may lose at most this many recall points against the 16 kHz transcript, on the full clip and on
    // slices at least MinRecallGateSeconds long.
    private const double RecallFloorPoints = 10;
    private const int MinRecallGateSeconds = 5;
    private const string ShortSliceRecallNote =
        "informational: the 2 s cut lands inside \"Americans\", so recall here measures the cut, not narrowband robustness";
    private const string JfkTranscript =
        "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

    private static readonly int[] SliceSeconds = [2, 5, 10];

    /// <summary>The 11 words <c>WhisperEnglishOnlyTests</c> pins (every word after the leading "and so, my").</summary>
    private static readonly string[] JfkWords =
        ["fellow", "americans", "ask", "not", "what", "your", "country", "can", "do", "for", "you"];

    private static readonly HashSet<string> StopWords = new(StringComparer.Ordinal)
    {
        "a", "an", "and", "the", "of", "for", "to", "so", "my", "your", "you", "what", "can", "do", "not", "is", "it", "in", "on",
    };

    private readonly ITestOutputHelper _out;

    public WhisperBenchTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Whisper_3060_Latency_Recall_And_Tokens()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the Whisper 3060 bench.");
            return;
        }
        string jfk = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(_out.WriteLine, jfk))
        {
            return;
        }
        string? outDir = Environment.GetEnvironmentVariable(OutDirEnvVar);
        if (!string.IsNullOrEmpty(outDir))
        {
            Directory.CreateDirectory(outDir);
        }
        bool exact = Environment.GetEnvironmentVariable(ExactEnvVar) == "1";
        if (exact)
        {
            // Read at backend construction, so they must be set before OpenBackend.
            KnobStore.Set(EngineKnobs.HighPrecisionGemm, true);
            KnobStore.Set(EngineKnobs.NoTf32, true);
        }
        try
        {
            bool cpu = Environment.GetEnvironmentVariable(BackendEnvVar) == "cpu";
            using IBackend backend = cpu ? new CpuBackend() : OpenBackend();
            float[] audio16k = LoadJfk16k(jfk);
            float[] narrowband = NarrowbandRoundTrip(audio16k);
            foreach (string repo in Models())
            {
                if (!RealWeightGate.Require(_out.WriteLine, WhisperFiles(repo)))
                {
                    Emit(new StringBuilder().AppendLine($"`{repo}`: SKIPPED, weights not cached under "
                        + $"{AudioModelCache.GetRepoDirectory(repo, "stt")}"));
                    continue;
                }
                Stopwatch load = Stopwatch.StartNew();
                using WhisperPipeline whisper = await WhisperPipeline.LoadAsync(repo);
                _out.WriteLine($"{repo} loaded in {load.Elapsed.TotalSeconds:F1}s (excluded from every row)");
                if (repo == (Environment.GetEnvironmentVariable(GateModelEnvVar) ?? GateModel))
                {
                    RunGate(backend, whisper, audio16k, narrowband, exact);
                    if (Environment.GetEnvironmentVariable(ProfileEnvVar) == "1")
                    {
                        RunProfiled(backend, whisper, audio16k, outDir);
                    }
                }
                else
                {
                    RunRegression(backend, whisper, audio16k, narrowband, exact);
                }
            }
        }
        finally
        {
            if (exact)
            {
                KnobStore.Clear(EngineKnobs.HighPrecisionGemm);
                KnobStore.Clear(EngineKnobs.NoTf32);
            }
        }
    }

    /// <summary>Latency per slice and variant, recall, the 11-word check, and the timestamped and streaming paths.</summary>
    private void RunGate(IBackend backend, WhisperPipeline whisper, float[] audio16k, float[] narrowband, bool exact)
    {
        WhisperOptions options = Options(whisper);
        StringBuilder table = new();
        table.AppendLine($"### Whisper `{whisper.ModelName}` per utterance — {backend.Capabilities.DeviceName}, in-process, "
            + $"{GateWarmRuns} warm + {GateTimedRuns} timed{(exact ? ", full F32" : "")}");
        table.AppendLine();
        table.AppendLine("| Slice | Variant | median ms | p95 ms | min ms | ≤ 350 ms (median / p95) | D2H syncs/call | tokens | vs ref | transcript |");
        table.AppendLine("|---:|---|---:|---:|---:|---|---:|---:|---|---|");

        Dictionary<(int Seconds, string Variant), string> heard = [];
        foreach (int seconds in SliceSeconds)
        {
            foreach ((string variant, float[] source) in Variants(audio16k, narrowband))
            {
                float[] slice = source[..Math.Min(source.Length, seconds * SampleRate)];
                (Stats stats, long syncs, List<int> tokens) = Measure(backend, whisper, slice, options, GateWarmRuns, GateTimedRuns);
                string text = whisper.DecodeText(tokens);
                heard[(seconds, variant)] = text;
                string name = CaseName(whisper, $"{seconds}s_{variant}");
                string check = CompareAndDump(name, tokens, text);
                table.AppendLine($"| {seconds} s | {variant} | {Ms(stats.Median)} | {Ms(stats.P95)} | {Ms(stats.Min)} | "
                    + $"{Gate(stats.Median)} / {Gate(stats.P95)} | {syncs} | {tokens.Count} | {check} | {Cell(text)} |");
                _out.WriteLine($"{seconds}s {variant}: median {Ms(stats.Median)} p95 {Ms(stats.P95)} min {Ms(stats.Min)} ms | "
                    + $"{syncs} D2H syncs | {tokens.Count} tokens | {check} | {text.Trim()}");
            }
        }
        table.AppendLine();

        table.AppendLine($"| Recall case | recall | Δ vs 16 kHz (pts) | gate (≥ −{RecallFloorPoints:0} pts) | JFK words | transcript |");
        table.AppendLine("|---|---:|---:|---|---:|---|");
        List<int> full16kIds = whisper.TranscribeTokenIds(backend, audio16k, SampleRate, options);
        List<int> fullNarrowIds = whisper.TranscribeTokenIds(backend, narrowband, SampleRate, options);
        string full16k = whisper.DecodeText(full16kIds);
        string fullNarrow = whisper.DecodeText(fullNarrowIds);
        double baseRecall = ContentWordRecall(JfkTranscript, full16k);
        double nbRecall = ContentWordRecall(JfkTranscript, fullNarrow);
        table.AppendLine($"| full clip, 16 kHz | {baseRecall:P0} | 0 | baseline | {JfkWordHits(full16k)}/{JfkWords.Length} | {Cell(full16k)} |");
        table.AppendLine($"| full clip, narrowband | {nbRecall:P0} | {Points(nbRecall - baseRecall)} | {RecallGate(nbRecall - baseRecall)} | "
            + $"{JfkWordHits(fullNarrow)}/{JfkWords.Length} | {Cell(fullNarrow)} |");
        table.AppendLine($"| full clip tokens vs ref | 16 kHz: {CompareAndDump(CaseName(whisper, "full_16k"), full16kIds, full16k)} | "
            + $"narrowband: {CompareAndDump(CaseName(whisper, "full_narrowband"), fullNarrowIds, fullNarrow)} | — | — | — |");
        foreach (int seconds in SliceSeconds)
        {
            // Words the 16 kHz slice decoded that the narrowband slice lost.
            string wide = heard[(seconds, "16k")];
            string narrow = heard[(seconds, "narrowband")];
            double sliceRecall = ContentWordRecall(wide, narrow);
            string gate = seconds >= MinRecallGateSeconds ? RecallGate(sliceRecall - 1) : ShortSliceRecallNote;
            table.AppendLine($"| {seconds} s narrowband vs its 16 kHz transcript | {sliceRecall:P0} | {Points(sliceRecall - 1)} | {gate} | — | "
                + $"{Cell(narrow)} |");
        }
        table.AppendLine();

        List<int> timed = whisper.TranscribeTokenIds(backend, audio16k, SampleRate, options with { WithTimestamps = true });
        IReadOnlyList<WhisperSegment> segments = whisper.SegmentAudio(backend, audio16k, SampleRate, options);
        string segmentText = string.Join(" | ", segments.Select(s => $"{s.Start:F2}-{s.End:F2} {s.Text}"));
        table.AppendLine($"Timestamped full clip: {timed.Count} ids, {CompareAndDump(CaseName(whisper, "full_16k_timestamps"), timed, segmentText)}; "
            + $"segments: {Cell(segmentText)}");
        string streamed = Stream(whisper, backend, audio16k, options);
        table.AppendLine($"Streaming (1 s pushes, LocalAgreement-2): {CompareAndDump(CaseName(whisper, "full_16k_streaming"), [], streamed)}; "
            + $"text: {Cell(streamed)}");
        Emit(table);

        Assert.Equal(JfkWords.Length, JfkWordHits(full16k));
    }

    /// <summary>Full clip at 16 kHz and narrowband plus the 5 s slice: tokens against the reference, wall for context.</summary>
    private void RunRegression(IBackend backend, WhisperPipeline whisper, float[] audio16k, float[] narrowband, bool exact)
    {
        WhisperOptions options = Options(whisper);
        (int warmRuns, int timedRuns) = RegressionRuns();
        StringBuilder table = new();
        table.AppendLine($"### Whisper `{whisper.ModelName}` regression — {backend.Capabilities.DeviceName}, "
            + $"{warmRuns} warm + {timedRuns} timed{(exact ? ", full F32" : "")}, Language = {options.Language ?? "null"}");
        table.AppendLine();
        table.AppendLine("| Case | median ms | min ms | D2H syncs/call | tokens | vs ref | JFK words | transcript |");
        table.AppendLine("|---|---:|---:|---:|---:|---|---:|---|");
        (string Name, float[] Audio)[] cases =
        [
            ("full_16k", audio16k),
            ("full_narrowband", narrowband),
            ("5s_16k", audio16k[..(5 * SampleRate)]),
        ];
        foreach ((string caseName, float[] audio) in cases)
        {
            (Stats stats, long syncs, List<int> tokens) = Measure(backend, whisper, audio, options, warmRuns, timedRuns);
            string text = whisper.DecodeText(tokens);
            string check = CompareAndDump(CaseName(whisper, caseName), tokens, text);
            table.AppendLine($"| {caseName} | {Ms(stats.Median)} | {Ms(stats.Min)} | {syncs} | {tokens.Count} | {check} | "
                + $"{JfkWordHits(text)}/{JfkWords.Length} | {Cell(text)} |");
            _out.WriteLine($"{whisper.ModelName} {caseName}: median {Ms(stats.Median)} ms | {syncs} D2H syncs | {check} | {text.Trim()}");
        }
        Emit(table);
    }

    /// <summary>Stage-timer runs (stage wall + D2H syncs, per-op host issue time) on each 16 kHz slice, then per-op GPU
    /// time (profileSync) on the 10 s slice. The engine log is captured into a queue while this runs — never written
    /// straight to <see cref="_out"/> from the logger, which outlives the test.</summary>
    private void RunProfiled(IBackend backend, WhisperPipeline whisper, float[] audio16k, string? outDir)
    {
        WhisperOptions options = Options(whisper);
        ConcurrentQueue<string> captured = new();
        Logs.SetLogger((level, message) => captured.Enqueue($"[{level}] {message}"));
        KnobStore.Set(EngineKnobs.Profile, true);
        try
        {
            foreach (int seconds in SliceSeconds)
            {
                float[] slice = audio16k[..(seconds * SampleRate)];
                whisper.TranscribeAudio(backend, slice, SampleRate, options);
                captured.Clear();
                for (int i = 0; i < 2; i++)
                {
                    whisper.TranscribeAudio(backend, slice, SampleRate, options);
                }
                _out.WriteLine($"stage timer, {seconds} s slice (diagnostics.profile, stages closed by a device sync):");
                Drain(captured);
            }
            string dir = string.IsNullOrEmpty(outDir) ? Path.GetTempPath() : outDir;
            float[] tenSeconds = audio16k[..(10 * SampleRate)];
            NvtxRange.ResetProfile();
            whisper.TranscribeAudio(backend, tenSeconds, SampleRate, options);
            string hostPath = Path.Combine(dir, "whisper_10s_profile_host.txt");
            NvtxRange.DumpProfile(hostPath);
            _out.WriteLine($"per-op HOST issue time, one 10 s transcription (profileSync off) at {hostPath}");
            PrintFile(hostPath);

            KnobStore.Set(EngineKnobs.ProfileSync, true);
            whisper.TranscribeAudio(backend, tenSeconds, SampleRate, options);
            captured.Clear();
            NvtxRange.ResetProfile();
            for (int i = 0; i < 3; i++)
            {
                whisper.TranscribeAudio(backend, tenSeconds, SampleRate, options);
            }
            string syncPath = Path.Combine(dir, "whisper_10s_profile_sync.txt");
            NvtxRange.DumpProfile(syncPath);
            _out.WriteLine($"per-op GPU time, three 10 s transcriptions (profileSync on, serialized) at {syncPath}");
            Drain(captured);
            PrintFile(syncPath);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Profile);
            KnobStore.Clear(EngineKnobs.ProfileSync);
            Logs.SetLogger((level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
    }

    private static (Stats Stats, long Syncs, List<int> Tokens) Measure(IBackend backend, WhisperPipeline whisper, float[] audio,
        WhisperOptions options, int warmRuns, int timedRuns)
    {
        List<int> tokens = [];
        for (int i = 0; i < warmRuns; i++)
        {
            tokens = whisper.TranscribeTokenIds(backend, audio, SampleRate, options);
        }
        List<double> samples = new(timedRuns);
        long syncs = 0;
        for (int i = 0; i < timedRuns; i++)
        {
            long before = backend.GetD2hSyncCount();
            Stopwatch sw = Stopwatch.StartNew();
            tokens = whisper.TranscribeTokenIds(backend, audio, SampleRate, options);
            sw.Stop();
            syncs = backend.GetD2hSyncCount() - before;
            samples.Add(sw.Elapsed.TotalSeconds);
        }
        return (Stats.Of(samples), syncs, tokens);
    }

    /// <summary>The whole clip pushed in 1 s chunks with a re-decode after each, then flushed.</summary>
    private static string Stream(WhisperPipeline whisper, IBackend backend, float[] audio, WhisperOptions options)
    {
        using WhisperStreamingPipeline stream = new(whisper, backend, options);
        for (int start = 0; start < audio.Length; start += SampleRate)
        {
            stream.PushAudio(audio.AsSpan(start, Math.Min(SampleRate, audio.Length - start)), SampleRate);
            stream.ProcessAvailable();
        }
        return stream.Finish();
    }

    /// <summary>Writes the case's ids and text to the out dir and compares them with the reference dir's.</summary>
    private string CompareAndDump(string name, List<int> tokens, string text)
    {
        string ids = string.Join(",", tokens);
        string? outDir = Environment.GetEnvironmentVariable(OutDirEnvVar);
        if (!string.IsNullOrEmpty(outDir))
        {
            File.WriteAllText(Path.Combine(outDir, name + ".tokens"), ids);
            File.WriteAllText(Path.Combine(outDir, name + ".txt"), text);
        }
        string? refDir = Environment.GetEnvironmentVariable(RefDirEnvVar);
        if (string.IsNullOrEmpty(refDir) || !File.Exists(Path.Combine(refDir, name + ".txt")))
        {
            return "no ref";
        }
        string refText = File.ReadAllText(Path.Combine(refDir, name + ".txt"));
        string refIds = File.Exists(Path.Combine(refDir, name + ".tokens")) ? File.ReadAllText(Path.Combine(refDir, name + ".tokens")) : "";
        if (refIds == ids && refText == text)
        {
            return "identical";
        }
        if (refIds != ids)
        {
            string[] a = refIds.Split(',', StringSplitOptions.RemoveEmptyEntries);
            string[] b = ids.Split(',', StringSplitOptions.RemoveEmptyEntries);
            int i = 0;
            while (i < a.Length && i < b.Length && a[i] == b[i])
            {
                i++;
            }
            return $"TOKENS DIFFER at {i} ({a.Length} vs {b.Length})";
        }
        return "TEXT DIFFERS";
    }

    private static IEnumerable<(string Variant, float[] Audio)> Variants(float[] audio16k, float[] narrowband)
    {
        yield return ("16k", audio16k);
        yield return ("narrowband", narrowband);
    }

    /// <summary>Probe A's options: no language token; English-only checkpoints ignore it, multilingual ones decode English.</summary>
    private static WhisperOptions Options(WhisperPipeline whisper) => new() { Language = whisper.IsMultilingual ? "en" : null };

    private static string CaseName(WhisperPipeline whisper, string caseName) => whisper.ModelName.Replace('/', '_') + "_" + caseName;

    private IEnumerable<string> Models()
    {
        string? list = Environment.GetEnvironmentVariable(ModelsEnvVar);
        return string.IsNullOrWhiteSpace(list) ? [GateModel]
            : list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    /// <summary>Warm-up and timed call counts for the regression protocol: <c>HARTSY_WHISPER_BENCH_RUNS=warm,timed</c>.</summary>
    private static (int Warm, int Timed) RegressionRuns()
    {
        string? runs = Environment.GetEnvironmentVariable(RunsEnvVar);
        if (string.IsNullOrWhiteSpace(runs))
        {
            return (RegressionWarmRuns, RegressionTimedRuns);
        }
        string[] parts = runs.Split(',', StringSplitOptions.TrimEntries);
        return (int.Parse(parts[0], CultureInfo.InvariantCulture), Math.Max(1, int.Parse(parts[1], CultureInfo.InvariantCulture)));
    }

    /// <summary>Opens the CUDA device named by <see cref="OrdinalEnvVar"/> and asserts it is the 3060.</summary>
    private IBackend OpenBackend()
    {
        string? ordinalText = Environment.GetEnvironmentVariable(OrdinalEnvVar);
        string? visible = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");
        int ordinal = !string.IsNullOrEmpty(ordinalText) ? int.Parse(ordinalText, CultureInfo.InvariantCulture)
            : !string.IsNullOrEmpty(visible) && !visible.Contains(',', StringComparison.Ordinal) ? 0 : 1;
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptx = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptx is null, "no compiled PTX directory beside the tests or in the repo");
        CudaBackend backend = new(ordinal, ptx);
        string device = backend.Capabilities.DeviceName;
        _out.WriteLine($"CUDA ordinal {ordinal} (CUDA_VISIBLE_DEVICES={visible ?? "unset"}): {device}; audio cache {AudioModelCache.CacheRoot}");
        if (!device.Contains(RequiredDeviceSubstring, StringComparison.Ordinal))
        {
            backend.Dispose();
            Assert.Fail($"ordinal {ordinal} is '{device}', not a {RequiredDeviceSubstring}. Set {OrdinalEnvVar} to the 3060's engine ordinal.");
        }
        return backend;
    }

    /// <summary>The files <see cref="WhisperPipeline.LoadAsync"/> fetches, resolved through the cache; checking them
    /// first is what keeps the load from downloading.</summary>
    private static string[] WhisperFiles(string repo)
    {
        string dir = AudioModelCache.GetRepoDirectory(repo, "stt");
        return WhisperPipeline.ModelFiles.Where(f => f.Required).Select(f => Path.Combine(dir, f.Name)).ToArray();
    }

    private static float[] LoadJfk16k(string path)
    {
        WavFile.DecodedAudio decoded = WavFile.Read(path);
        float[] mono = decoded.ToMono();
        return decoded.SampleRate == SampleRate ? mono : Resampler.Create(decoded.SampleRate, SampleRate).Resample(mono);
    }

    /// <summary>Probe A's narrowband variant: the engine resampler 16 k → 8 k → 16 k, trimmed to the input length.</summary>
    private static float[] NarrowbandRoundTrip(float[] audio16k)
    {
        float[] down = Resampler.Create(SampleRate, NarrowbandRate).Resample(audio16k);
        float[] up = Resampler.Create(NarrowbandRate, SampleRate).Resample(down);
        return up.Length > audio16k.Length ? up[..audio16k.Length] : up;
    }

    private static double ContentWordRecall(string reference, string hypothesis)
    {
        HashSet<string> hyp = new(Words(hypothesis), StringComparer.Ordinal);
        List<string> content = Words(reference).Where(w => !StopWords.Contains(w)).Distinct().ToList();
        return content.Count == 0 ? 1 : content.Count(hyp.Contains) / (double)content.Count;
    }

    private static int JfkWordHits(string text)
    {
        HashSet<string> words = new(Words(text), StringComparer.Ordinal);
        return JfkWords.Count(words.Contains);
    }

    private static IEnumerable<string> Words(string text)
    {
        StringBuilder sb = new(text.Length);
        foreach (char c in text.ToLowerInvariant())
        {
            sb.Append(char.IsLetterOrDigit(c) || c == '\'' ? c : ' ');
        }
        return sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    private void Drain(ConcurrentQueue<string> captured)
    {
        while (captured.TryDequeue(out string? line))
        {
            if (line.Contains("[Whisper]", StringComparison.Ordinal))
            {
                _out.WriteLine(line);
            }
        }
    }

    private void PrintFile(string path)
    {
        if (File.Exists(path))
        {
            _out.WriteLine(File.ReadAllText(path));
        }
    }

    private static string Gate(double seconds) => seconds * 1000 <= GateMs ? "met" : "MISSED";

    /// <summary>The recall gate for a narrowband-minus-16 kHz recall difference (a fraction).</summary>
    private static string RecallGate(double delta) => delta * 100 >= -RecallFloorPoints - 1e-9 ? "met" : "MISSED";

    private static string Ms(double seconds) => (seconds * 1000).ToString("F1", CultureInfo.InvariantCulture);

    private static string Points(double delta) => (delta * 100).ToString("+0;-0;0", CultureInfo.InvariantCulture);

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
