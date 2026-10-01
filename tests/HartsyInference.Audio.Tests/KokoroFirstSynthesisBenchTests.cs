using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro latency on text it has never spoken: the phone-call gate (a 15-word sentence ≤ 250 ms on the 3060),
/// measured the way a call produces it — every sentence new. The plain bench repeats one sentence, so it reports the
/// warm cost of a length the backend has already set up. Opt-in with <c>HARTSY_KOKORO_FIRST=1</c>.
///
/// <para>One process runs: one warm-up synthesis (the session's <c>WarmAsync</c>), "Okay." twice, then 20 distinct
/// 15-word sentences, then the same 20 again. The first pass is the gate; the second pass speaks lengths the backend has
/// already seen, so the difference between the passes is per-length setup. <c>HARTSY_KOKORO_FIRST_MODE=gaps</c> sleeps
/// 2-5 s (fixed sequence) before every sentence of both passes, the way turns are spaced on a call; comparing it with
/// the default back-to-back mode on the same sentences isolates what idle time costs. <c>HARTSY_KOKORO_FIRST_ORDER=reverse</c>
/// speaks the 20 in reverse order (rows stay keyed by sentence), so two runs can show the audio does not depend on which
/// length came first.</para>
///
/// <para>Per sentence it records wall time (G2P included, as the synthesizer lease runs it), token and frame counts,
/// device→host syncs, and the backend's per-shape setup: cuDNN convolution plans built (from a length bucket's engine
/// choice or by their own heuristic), bucket reference builds, heuristic and finalize time, cuBLASLt plans, and the
/// default pool's reserved bytes. <c>HARTSY_KOKORO_FIRST_PROFILE=1</c> turns the pipeline stage timer on (it syncs at
/// every stage, so gate numbers come from runs without it). <c>HARTSY_KOKORO_FIRST_CLEANUP</c> picks what runs after
/// each synthesis, outside the timer: <c>session</c> (default) is <c>FreeActivations()</c>, which also trims the pool;
/// <c>free</c> keeps the pool (<c>FreeActivations(trimPool: false)</c>, what the voice session runs per job); <c>none</c>
/// runs nothing. <c>HARTSY_KOKORO_FIRST_KNOBS</c> takes an engine settings document for a fix arm, for example
/// <c>{"settings": {"numerics.audioConvCudnn": false}}</c>. Whisper-tiny verifies every first-pass sentence after all
/// timing.</para>
///
/// <para>Output: tables to the test log and <c>HARTSY_KOKORO_FIRST_OUT</c>; with <c>HARTSY_KOKORO_FIRST_OUT_DIR</c>, a
/// per-sentence CSV (UTC start and end, for joining with an nvidia-smi clock log) and each first-pass sentence as
/// <c>.f32</c>/<c>.wav</c>; with <c>HARTSY_KOKORO_FIRST_REF_DIR</c>, each first-pass sentence is compared with the
/// same-named <c>.f32</c> there (byte identity, else log-spectral correlation and max-abs).</para></summary>
public sealed class KokoroFirstSynthesisBenchTests
{
    private const string GateEnvVar = "HARTSY_KOKORO_FIRST";
    private const string ModeEnvVar = "HARTSY_KOKORO_FIRST_MODE";
    private const string OrderEnvVar = "HARTSY_KOKORO_FIRST_ORDER";
    private const string ProfileEnvVar = "HARTSY_KOKORO_FIRST_PROFILE";
    private const string CleanupEnvVar = "HARTSY_KOKORO_FIRST_CLEANUP";
    private const string KnobsEnvVar = "HARTSY_KOKORO_FIRST_KNOBS";
    private const string OrdinalEnvVar = "HARTSY_KOKORO_FIRST_CUDA_ORDINAL";
    private const string OutEnvVar = "HARTSY_KOKORO_FIRST_OUT";
    private const string OutDirEnvVar = "HARTSY_KOKORO_FIRST_OUT_DIR";
    private const string RefDirEnvVar = "HARTSY_KOKORO_FIRST_REF_DIR";
    private const string WhisperTiny = "openai/whisper-tiny";
    private const int WhisperRate = 16_000;
    private const int SamplesPerFrame = 600;
    private const string Voice = GpuBenchSupport.KokoroVoice;
    private const string WarmUpText = "Hello, thank you for calling, how can I help you today?";
    private const string ShortProbe = "Okay.";

    /// <summary>Twenty distinct 15-word sentences, phone-call register, every word in CMUdict.</summary>
    private static readonly string[] Sentences =
    [
        "Your package left our warehouse this morning and should arrive at your door by Friday.",
        "I can move your dentist appointment to Monday if the afternoon still works for you.",
        "The billing team reviewed your account and removed the late fee from your previous statement.",
        "Please bring a photo identification and your insurance card when you visit the clinic tomorrow.",
        "Our technician will arrive between nine and eleven, and he will call before he comes.",
        "Your table for four is reserved on Saturday evening, right beside the large front window.",
        "The pharmacy says your prescription is ready, and you can pick it up after lunch.",
        "We received your payment yesterday, so your internet service should come back within the hour.",
        "I found two flights to Denver on Thursday, one leaves early and one leaves late.",
        "Your warranty covers the broken screen, so the repair will not cost you anything today.",
        "The school called to say the field trip was moved because of the rain forecast.",
        "I have sent the contract to your email, please sign it and send it back.",
        "Your rental car is a silver sedan, and the keys are waiting at the desk.",
        "We need a few more details before we can process the refund for your order.",
        "The plumber found a leak under the kitchen sink and fixed it in an hour.",
        "Your library books are due Friday, but you can renew them online or by phone.",
        "The doctor wants to see you again in two weeks to check on your healing.",
        "Your gym membership will renew on the first of the month unless you cancel it.",
        "I booked the conference room for Wednesday morning and sent invitations to the whole team.",
        "The hotel shuttle runs every thirty minutes from the airport, and the ride is free.",
    ];

    private readonly ITestOutputHelper _out;

    public KokoroFirstSynthesisBenchTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Sentences_AreTwentyDistinctFifteenWordLines()
    {
        Assert.Equal(20, Sentences.Length);
        Assert.Equal(Sentences.Length, Sentences.Distinct(StringComparer.Ordinal).Count());
        Assert.All(Sentences, s => Assert.Equal(15, s.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length));
        Assert.DoesNotContain(WarmUpText, Sentences);
    }

    [Fact]
    public void Gaps_StayWithinTwoToFiveSeconds()
    {
        for (int i = 0; i < Sentences.Length; i++)
        {
            double gap = GapSeconds(i);
            Assert.InRange(gap, 2.0, 5.0);
        }
    }

    [Theory]
    [InlineData("{\"settings\": {\"numerics.audioConvCudnn\": false}}", true)]
    [InlineData("{\"profile\": \"reference\"}", true)]
    [InlineData("{\"numerics.audioConvCudnn\": false}", false)]
    [InlineData("[]", false)]
    public void KnobDocument_NeedsSettingsOrProfile(string json, bool applies) =>
        Assert.Equal(applies, AppliesSettings(json));

    [Fact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "RealWeights")]
    public async Task Kokoro_DistinctSentences_3060_FirstSynthesis()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the Kokoro first-synthesis bench.");
            return;
        }
        string cmudict = GpuBenchSupport.Cmudict();
        if (!RealWeightGate.Require(_out.WriteLine,
                GpuBenchSupport.KokoroFiles().Concat(GpuBenchSupport.WhisperFiles(WhisperTiny)).Concat([cmudict]).ToArray()))
        {
            return;
        }
        string? knobs = Environment.GetEnvironmentVariable(KnobsEnvVar);
        bool profile = Environment.GetEnvironmentVariable(ProfileEnvVar) == "1";
        if (!string.IsNullOrWhiteSpace(knobs))
        {
            // A document without "settings" or "profile" applies nothing, silently: an arm that meant to change the engine
            // would measure the default instead.
            Assert.True(AppliesSettings(knobs), $"{KnobsEnvVar} must be a settings document, e.g. "
                + "{\"settings\": {\"numerics.audioConvCudnn\": false}}");
            KnobFile.Apply(knobs, KnobsEnvVar);
        }
        ConcurrentQueue<string> captured = new();
        if (profile)
        {
            KnobStore.Set(EngineKnobs.Profile, true);
            Logs.SetLogger((level, message) => captured.Enqueue($"[{level}] {message}"));
        }
        try
        {
            await RunAsync(cmudict, knobs, profile, captured);
        }
        finally
        {
            if (profile)
            {
                Logs.SetLogger((level, message) => Console.Error.WriteLine($"[{level}] {message}"));
            }
            KnobFile.Reload();
        }
    }

    private async Task RunAsync(string cmudict, string? knobs, bool profile, ConcurrentQueue<string> captured)
    {
        bool gaps = string.Equals(Environment.GetEnvironmentVariable(ModeEnvVar), "gaps", StringComparison.OrdinalIgnoreCase);
        bool reverse = string.Equals(Environment.GetEnvironmentVariable(OrderEnvVar), "reverse", StringComparison.OrdinalIgnoreCase);
        Cleanup cleanup = (Environment.GetEnvironmentVariable(CleanupEnvVar) ?? "session").ToLowerInvariant() switch
        {
            "session" => Cleanup.Session,
            "free" => Cleanup.FreeKeepPool,
            "none" => Cleanup.None,
            string other => throw new ArgumentException($"{CleanupEnvVar}={other}: expected session, free or none."),
        };
        string? outDir = Environment.GetEnvironmentVariable(OutDirEnvVar);
        string? refDir = Environment.GetEnvironmentVariable(RefDirEnvVar);
        if (!string.IsNullOrEmpty(outDir))
        {
            Directory.CreateDirectory(outDir);
        }

        using CudaBackend backend = GpuBenchSupport.Open3060(_out.WriteLine, OrdinalEnvVar);
        EnglishG2P g2p = new EnglishG2P(cmudict);
        Stopwatch load = Stopwatch.StartNew();
        using KokoroPipeline kokoro = await KokoroPipeline.LoadAsync();
        using WhisperPipeline verify = await WhisperPipeline.LoadAsync(WhisperTiny);
        _out.WriteLine($"Kokoro + {WhisperTiny} loaded in {load.Elapsed.TotalSeconds:F1}s (excluded)");
        string arm = $"mode={(gaps ? "gaps 2-5 s" : "back-to-back")} order={(reverse ? "reverse" : "forward")} cleanup={cleanup} "
            + $"stage-timer={(profile ? "on" : "off")} length-buckets={(EngineKnobs.AudioConvLengthBuckets.Value ? "on" : "off")} "
            + $"knobs={(string.IsNullOrWhiteSpace(knobs) ? "default" : knobs.Trim())}";
        _out.WriteLine($"arm: {arm}");

        // The voice session's warm-up: one synthesis before the first real sentence.
        Measure(backend, kokoro, g2p, WarmUpText, cleanup, captured);
        StringBuilder table = new StringBuilder();
        table.AppendLine($"### Kokoro first synthesis of new text — {backend.Capabilities.DeviceName}, {arm}");
        table.AppendLine();
        Row probe1 = Measure(backend, kokoro, g2p, ShortProbe, cleanup, captured);
        Row probe2 = Measure(backend, kokoro, g2p, ShortProbe, cleanup, captured);
        table.AppendLine($"\"{ShortProbe}\" first {Ms(probe1.WallMs)} ms ({probe1.ConvPlans} conv plans, {probe1.ConvReferences} "
            + $"bucket references, {Ms(probe1.ConvBuildMs)} ms building), second {Ms(probe2.WallMs)} ms ({probe2.ConvPlans} conv plans)");
        table.AppendLine();

        Row[] first = RunPass(backend, kokoro, g2p, gaps, reverse, cleanup, captured);
        Row[] repeat = RunPass(backend, kokoro, g2p, gaps, reverse, cleanup, captured);
        // Read before verification: the recognizer builds plans for its own convolutions on this backend.
        string families = backend.DescribeCudnnConvPlanFamilies(25);

        // Verification after all timing, so the recognizer's own setup never lands inside a timed sentence.
        Resampler toWhisper = Resampler.Create(kokoro.Config.SampleRate, WhisperRate);
        WhisperOptions options = new WhisperOptions { Language = "en" };
        for (int i = 0; i < first.Length; i++)
        {
            string heard = verify.TranscribeAudio(backend, toWhisper.Resample(first[i].Wave), WhisperRate, options);
            first[i] = first[i] with { Heard = heard, Recall = AudioParityMetrics.ContentWordRecall(Sentences[i], heard) };
        }

        Comparison[]? versusReference = string.IsNullOrEmpty(refDir) ? null : CompareWithReference(first, refDir);
        AppendSummary(table, first, repeat, versusReference);
        AppendSentences(table, first, repeat, versusReference);
        table.AppendLine();
        table.AppendLine("Kokoro cuDNN convolution families, costliest first (plans built by the warm-up, probes and both passes):");
        table.AppendLine("```");
        table.Append(families);
        table.AppendLine("```");
        if (profile)
        {
            AppendStages(table, first, repeat);
        }
        GpuBenchSupport.Emit(_out.WriteLine, table, OutEnvVar);
        if (!string.IsNullOrEmpty(outDir))
        {
            WriteArtifacts(outDir, kokoro.Config.SampleRate, first, repeat);
        }
    }

    /// <summary>Speaks the 20 sentences once, in order or in reverse; the result is indexed by sentence either way. Gap
    /// lengths follow the speaking position, so both orders sleep the same sequence.</summary>
    private static Row[] RunPass(CudaBackend backend, KokoroPipeline kokoro, EnglishG2P g2p, bool gaps, bool reverse,
        Cleanup cleanup, ConcurrentQueue<string> captured)
    {
        Row[] rows = new Row[Sentences.Length];
        for (int position = 0; position < Sentences.Length; position++)
        {
            int index = reverse ? Sentences.Length - 1 - position : position;
            if (gaps)
            {
                Thread.Sleep(TimeSpan.FromSeconds(GapSeconds(position)));
            }
            rows[index] = Measure(backend, kokoro, g2p, Sentences[index], cleanup, captured) with { Position = position + 1 };
        }
        return rows;
    }

    /// <summary>One synthesis as the synthesizer lease runs it (G2P, then the pipeline), timed end to end, with the
    /// backend's per-shape setup read around it. The cleanup runs after the clock stops, where the voice session's GPU
    /// worker runs it.</summary>
    private static Row Measure(CudaBackend backend, KokoroPipeline kokoro, EnglishG2P g2p, string text, Cleanup cleanup,
        ConcurrentQueue<string> captured)
    {
        captured.Clear();
        CudnnConvPlanStats convBefore = backend.CudnnConvPlanStats;
        (int _, long ltMissesBefore, long _, double ltMsBefore) = backend.LtGemmPlanStats;
        (long reservedBefore, long _) = backend.GetMemPoolUsage();
        long syncsBefore = backend.GetD2hSyncCount();
        DateTime startUtc = DateTime.UtcNow;
        long start = Stopwatch.GetTimestamp();
        string ipa = g2p.ToIpa(text);
        double g2pMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        float[] wave = kokoro.Synthesize(backend, ipa, Voice, 1f);
        double wallMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        DateTime endUtc = DateTime.UtcNow;
        long syncs = backend.GetD2hSyncCount() - syncsBefore;
        CudnnConvPlanStats convAfter = backend.CudnnConvPlanStats;
        (int _, long ltMissesAfter, long _, double ltMsAfter) = backend.LtGemmPlanStats;
        (long reservedAfter, long _) = backend.GetMemPoolUsage();
        switch (cleanup)
        {
            case Cleanup.Session:
                backend.FreeActivations();
                break;
            case Cleanup.FreeKeepPool:
                backend.FreeActivations(trimPool: false);
                break;
        }
        string stages = string.Join(" || ", captured.Where(l => l.Contains("[Kokoro]", StringComparison.Ordinal)));
        return new Row(text, 0, startUtc, endUtc, wallMs, g2pMs, kokoro.CountTokens(ipa), wave.Length / SamplesPerFrame, syncs,
            convAfter.PlanBuilds - convBefore.PlanBuilds, convAfter.BucketPlanBuilds - convBefore.BucketPlanBuilds,
            convAfter.ReferenceBuilds - convBefore.ReferenceBuilds, convAfter.BucketFallbacks - convBefore.BucketFallbacks,
            convAfter.BuildMs - convBefore.BuildMs, convAfter.HeuristicMs - convBefore.HeuristicMs,
            convAfter.FinalizeMs - convBefore.FinalizeMs, ltMissesAfter - ltMissesBefore, ltMsAfter - ltMsBefore,
            reservedBefore, reservedAfter, wave, AudioParityMetrics.Sha256Hex(wave)[..12], stages, "", double.NaN);
    }

    /// <summary>Each first-pass sentence against the same-named <c>.f32</c> in <paramref name="refDir"/>.</summary>
    private static Comparison[] CompareWithReference(Row[] first, string refDir)
    {
        Comparison[] result = new Comparison[first.Length];
        for (int i = 0; i < first.Length; i++)
        {
            string path = Path.Combine(refDir, Name(i) + ".f32");
            if (!File.Exists(path))
            {
                result[i] = new Comparison(false, false, double.NaN, double.NaN);
                continue;
            }
            float[] reference = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();
            float[] wave = first[i].Wave;
            if (reference.Length != wave.Length)
            {
                result[i] = new Comparison(true, false, double.NaN, double.NaN);
                continue;
            }
            bool identical = reference.AsSpan().SequenceEqual(wave);
            (double maxAbs, double _) = AudioParityMetrics.Compare(reference, wave);
            result[i] = new Comparison(true, identical, identical ? 1.0 : AudioParityMetrics.LogSpectralCorrelation(reference, wave), maxAbs);
        }
        return result;
    }

    private static void AppendSummary(StringBuilder table, Row[] first, Row[] repeat, Comparison[]? versusReference)
    {
        table.AppendLine("| Pass | p50 ms | p95 ms | min ms | max ms | mean conv plans (from bucket) | mean bucket references "
            + "| mean conv build ms | mean Lt plans | mean pool growth MB | recall min / mean |");
        table.AppendLine("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|");
        foreach ((string name, Row[] rows) in new[] { ("first (new text)", first), ("repeat (same text)", repeat) })
        {
            List<double> sorted = rows.Select(r => r.WallMs).OrderBy(v => v).ToList();
            string recall = rows.All(r => double.IsNaN(r.Recall)) ? "—"
                : $"{rows.Min(r => r.Recall):P0} / {rows.Average(r => r.Recall):P0}";
            table.AppendLine($"| {name} | {Ms(GpuBenchSupport.Percentile(sorted, 0.5))} | {Ms(GpuBenchSupport.Percentile(sorted, 0.95))} "
                + $"| {Ms(sorted[0])} | {Ms(sorted[^1])} | {rows.Average(r => r.ConvPlans):F1} ({rows.Average(r => r.ConvBucketPlans):F1}) "
                + $"| {rows.Average(r => r.ConvReferences):F1} | {rows.Average(r => r.ConvBuildMs):F1} | {rows.Average(r => r.LtPlans):F1} "
                + $"| {rows.Average(r => (r.ReservedAfter - r.ReservedBefore) / 1048576.0):F1} | {recall} |");
        }
        int same = first.Zip(repeat).Count(p => p.First.Digest == p.Second.Digest);
        table.AppendLine();
        table.AppendLine($"First and repeat pass give byte-identical audio for {same} of {first.Length} sentences; bucket "
            + $"fallbacks (a bucket's choice that did not finalize for a length) over both passes: "
            + $"{first.Sum(r => r.ConvFallbacks) + repeat.Sum(r => r.ConvFallbacks)}.");
        if (versusReference is not null)
        {
            Comparison[] compared = versusReference.Where(c => c.Found).ToArray();
            Comparison[] differing = compared.Where(c => !c.Identical).ToArray();
            string floor = differing.Length == 0 ? "" : differing.Any(c => double.IsNaN(c.LogSpectral))
                ? "; at least one differs in length"
                : $"; the others: log-spectral correlation ≥ {differing.Min(c => c.LogSpectral):F6}, max-abs ≤ {differing.Max(c => c.MaxAbs):E2}";
            table.AppendLine($"Against the reference audio: {compared.Count(c => c.Identical)} of {compared.Length} sentences "
                + $"byte-identical{floor}.");
        }
        table.AppendLine();
    }

    private static void AppendSentences(StringBuilder table, Row[] first, Row[] repeat, Comparison[]? versusReference)
    {
        table.AppendLine("| # | spoken at | tokens | frames | first ms | repeat ms | G2P ms | D2H syncs | conv plans (from bucket) "
            + "| bucket references | conv build ms (heuristic / finalize) | Lt plans (ms) | pool MB before → after | recall "
            + "| vs reference | heard |");
        table.AppendLine("|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---:|---|---|");
        for (int i = 0; i < first.Length; i++)
        {
            Row r = first[i];
            string vsRef = "—";
            if (versusReference is { } comparisons && comparisons[i].Found)
            {
                Comparison c = comparisons[i];
                vsRef = c.Identical ? "identical"
                    : double.IsNaN(c.LogSpectral) ? "length differs"
                    : $"log-spec {c.LogSpectral:F6}, max-abs {c.MaxAbs:E2}";
            }
            table.AppendLine($"| {i + 1} | {r.Position} | {r.Tokens} | {r.Frames} | {Ms(r.WallMs)} | {Ms(repeat[i].WallMs)} | {r.G2pMs:F2} "
                + $"| {r.Syncs} | {r.ConvPlans} ({r.ConvBucketPlans}) | {r.ConvReferences} "
                + $"| {r.ConvBuildMs:F1} ({r.ConvHeuristicMs:F1} / {r.ConvFinalizeMs:F1}) | {r.LtPlans} ({r.LtBuildMs:F1}) "
                + $"| {r.ReservedBefore / 1048576.0:F0} → {r.ReservedAfter / 1048576.0:F0} | {r.Recall:P0} | {vsRef} "
                + $"| {AudioParityMetrics.Cell(r.Heard)} |");
        }
    }

    private static void AppendStages(StringBuilder table, Row[] first, Row[] repeat)
    {
        table.AppendLine();
        table.AppendLine("Stage timer per sentence (stage=ms/D2H syncs; each mark drains the stream):");
        table.AppendLine("```");
        for (int i = 0; i < first.Length; i++)
        {
            table.AppendLine($"#{i + 1} first : {first[i].Stages}");
            table.AppendLine($"#{i + 1} repeat: {repeat[i].Stages}");
        }
        table.AppendLine("```");
    }

    private static void WriteArtifacts(string outDir, int sampleRate, Row[] first, Row[] repeat)
    {
        StringBuilder csv = new StringBuilder();
        csv.AppendLine("pass,index,position,utc_start,utc_end,wall_ms,g2p_ms,tokens,frames,d2h_syncs,conv_plans,conv_bucket_plans,"
            + "conv_references,conv_fallbacks,conv_build_ms,conv_heuristic_ms,conv_finalize_ms,lt_plans,lt_build_ms,"
            + "pool_reserved_before,pool_reserved_after,sha12");
        foreach ((string pass, Row[] rows) in new[] { ("first", first), ("repeat", repeat) })
        {
            for (int i = 0; i < rows.Length; i++)
            {
                Row r = rows[i];
                csv.AppendLine(string.Join(',', pass, (i + 1).ToString(CultureInfo.InvariantCulture), r.Position,
                    r.StartUtc.ToString("O", CultureInfo.InvariantCulture), r.EndUtc.ToString("O", CultureInfo.InvariantCulture),
                    F(r.WallMs), F(r.G2pMs), r.Tokens, r.Frames, r.Syncs, r.ConvPlans, r.ConvBucketPlans, r.ConvReferences,
                    r.ConvFallbacks, F(r.ConvBuildMs), F(r.ConvHeuristicMs), F(r.ConvFinalizeMs), r.LtPlans, F(r.LtBuildMs),
                    r.ReservedBefore, r.ReservedAfter, r.Digest));
            }
        }
        File.WriteAllText(Path.Combine(outDir, "kokoro_first_sentences.csv"), csv.ToString());
        for (int i = 0; i < first.Length; i++)
        {
            File.WriteAllBytes(Path.Combine(outDir, Name(i) + ".f32"), MemoryMarshal.AsBytes<float>(first[i].Wave).ToArray());
            WavFile.WriteMono16(Path.Combine(outDir, Name(i) + ".wav"), first[i].Wave, sampleRate);
        }
    }

    /// <summary>Whether an engine settings document changes anything: <see cref="KnobFile.Apply"/> reads only its
    /// <c>settings</c> object and <c>profile</c> name and ignores every other property.</summary>
    private static bool AppliesSettings(string json)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;
            return root.ValueKind == JsonValueKind.Object
                && ((root.TryGetProperty("settings", out JsonElement settings) && settings.ValueKind == JsonValueKind.Object)
                    || (root.TryGetProperty("profile", out JsonElement profile) && profile.ValueKind == JsonValueKind.String));
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>The fixed idle gap before speaking position <paramref name="position"/> in gap mode: 2-5 s, spread by the
    /// golden ratio so consecutive gaps differ and every run sleeps the same sequence.</summary>
    private static double GapSeconds(int position) => 2.0 + 3.0 * ((position + 1) * 0.6180339887498949 % 1.0);

    private static string Name(int index) => $"kokoro_first_{index + 1:D2}";

    private static string Ms(double ms) => ms.ToString("F1", CultureInfo.InvariantCulture);

    private static string F(double value) => value.ToString("F3", CultureInfo.InvariantCulture);

    /// <summary>What runs after each synthesis.</summary>
    private enum Cleanup
    {
        /// <summary><c>FreeActivations()</c>, which trims the pool too.</summary>
        Session,

        /// <summary><c>FreeActivations(trimPool: false)</c>: activations go back to the pool, which keeps them.</summary>
        FreeKeepPool,

        /// <summary>Nothing.</summary>
        None,
    }

    /// <summary>One first-pass sentence against its reference: whether a reference was found, byte identity, and when the
    /// bytes differ, the log-spectral correlation (NaN when the lengths differ) and max-abs difference.</summary>
    private readonly record struct Comparison(bool Found, bool Identical, double LogSpectral, double MaxAbs);

    private sealed record Row(string Text, int Position, DateTime StartUtc, DateTime EndUtc, double WallMs, double G2pMs, int Tokens,
        int Frames, long Syncs, long ConvPlans, long ConvBucketPlans, long ConvReferences, long ConvFallbacks, double ConvBuildMs,
        double ConvHeuristicMs, double ConvFinalizeMs, long LtPlans, double LtBuildMs, long ReservedBefore, long ReservedAfter,
        float[] Wave, string Digest, string Stages, string Heard, double Recall);
}
