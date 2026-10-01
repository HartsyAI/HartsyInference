using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Scores the arms <c>TtsConvBucketSpotCheckTests</c> wrote (one directory per model under
/// <c>HARTSY_TTS_SPOT_OUT_DIR</c>, <c>&lt;arm&gt;_&lt;nn&gt;.f32</c> plus <c>arms.csv</c>): per sentence, byte identity,
/// log-spectral correlation and max-abs between the conv engine chosen per length bucket (on) and per exact length (off),
/// and both against the no-TF32 arm (f32) when there is one; then Whisper small.en on the CPU for every arm, with
/// content-word recall against the text and whether the transcripts match, and each arm's first and repeat synthesis
/// time (the repeat is the steady state, every plan cached). Where on and off differ beyond rounding
/// (log-spectral correlation under 0.999) it says what moved: the lag that best aligns them, where the difference energy
/// sits, and when the two waveforms first part. Opt-in with <c>HARTSY_TTS_SPOT_COMPARE=1</c>; the table goes to the test
/// log and, with <c>HARTSY_TTS_SPOT_REPORT</c>, to that file.</summary>
public sealed class TtsConvBucketSpotCheckCompareTests
{
    private const string GateEnvVar = "HARTSY_TTS_SPOT_COMPARE";
    private const string OutDirEnvVar = "HARTSY_TTS_SPOT_OUT_DIR";
    private const string ReportEnvVar = "HARTSY_TTS_SPOT_REPORT";
    private const string WhisperRepo = "openai/whisper-small.en";
    private const int WhisperRate = 16_000;

    private readonly ITestOutputHelper _out;

    public TtsConvBucketSpotCheckCompareTests(ITestOutputHelper output) => _out = output;

    [Fact]
    [Trait("Category", "Integration")]
    [Trait("Category", "RealWeights")]
    public async Task CompareArms()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to score the length-bucket spot-check arms.");
            return;
        }
        string root = Environment.GetEnvironmentVariable(OutDirEnvVar)
            ?? throw new InvalidOperationException($"{OutDirEnvVar} is not set");
        if (!RealWeightGate.Require(_out.WriteLine, GpuBenchSupport.WhisperFiles(WhisperRepo)))
        {
            return;
        }
        using IBackend cpu = new CpuBackend();
        using WhisperPipeline whisper = await WhisperPipeline.LoadAsync(WhisperRepo);
        WhisperOptions options = new WhisperOptions { Language = "en" };

        StringBuilder table = new StringBuilder();
        table.AppendLine("### cuDNN conv length buckets — spot check, on (per bucket) against off (per length) and f32 (no TF32)");
        table.AppendLine();
        table.AppendLine("| Model | # | samples off / on | first ms off / on | repeat ms off / on | on vs off | off vs f32 | on vs f32 "
            + "| recall off / on (f32) | transcripts | what moved |");
        table.AppendLine("|---|---:|---|---|---|---|---|---|---|---|---|");
        foreach (string dir in Directory.GetDirectories(root).OrderBy(d => d, StringComparer.Ordinal))
        {
            string model = Path.GetFileName(dir);
            Dictionary<(string Arm, int Index), ArmRow> rows = ReadArms(Path.Combine(dir, "arms.csv"));
            foreach (int index in rows.Keys.Select(k => k.Index).Distinct().Order())
            {
                if (!rows.TryGetValue(("off", index), out ArmRow? offRow) || !rows.TryGetValue(("on", index), out ArmRow? onRow))
                {
                    continue;
                }
                (int rate, string text) = (offRow.Rate, offRow.Text);
                float[]? off = Load(dir, "off", index), on = Load(dir, "on", index), f32 = Load(dir, "f32", index);
                if (off is null || on is null)
                {
                    continue;
                }
                string Heard(float[] wave) =>
                    whisper.TranscribeAudio(cpu, Resampler.Create(rate, WhisperRate).Resample(wave), WhisperRate, options);
                string heardOff = Heard(off), heardOn = Heard(on);
                string? heardF32 = f32 is null ? null : Heard(f32);
                double recallOff = AudioParityMetrics.ContentWordRecall(text, heardOff);
                double recallOn = AudioParityMetrics.ContentWordRecall(text, heardOn);
                string recall = $"{recallOff:P0} / {recallOn:P0}"
                    + (heardF32 is null ? "" : $" ({AudioParityMetrics.ContentWordRecall(text, heardF32):P0})");
                bool sameWords = AudioParityMetrics.Words(heardOff).SequenceEqual(AudioParityMetrics.Words(heardOn));
                double logSpec = off.Length == on.Length ? AudioParityMetrics.LogSpectralCorrelation(off, on) : double.NaN;
                string moved = !double.IsNaN(logSpec) && logSpec < 0.999 ? WhatMoved(off, on, rate) : "—";
                table.AppendLine($"| {model} | {index} | {off.Length} / {on.Length} | {offRow.FirstMs:F1} / {onRow.FirstMs:F1} "
                    + $"| {offRow.RepeatMs:F1} / {onRow.RepeatMs:F1} | {Versus(off, on)} "
                    + $"| {(f32 is null ? "—" : Versus(f32, off))} | {(f32 is null ? "—" : Versus(f32, on))} | {recall} "
                    + $"| {(sameWords ? "same" : $"off '{AudioParityMetrics.Cell(heardOff)}' / on '{AudioParityMetrics.Cell(heardOn)}'")} "
                    + $"| {moved} |");
            }
        }
        string rendered = table.ToString();
        _out.WriteLine(rendered);
        string? report = Environment.GetEnvironmentVariable(ReportEnvVar);
        if (!string.IsNullOrEmpty(report))
        {
            File.WriteAllText(report, rendered);
        }
    }

    /// <summary>Byte identity, else log-spectral correlation and max-abs (or the length change).</summary>
    private static string Versus(float[] a, float[] b)
    {
        if (a.Length != b.Length)
        {
            return $"length {a.Length} → {b.Length}";
        }
        if (a.AsSpan().SequenceEqual(b))
        {
            return "identical";
        }
        (double maxAbs, double _) = AudioParityMetrics.Compare(a, b);
        return $"log-spec {AudioParityMetrics.LogSpectralCorrelation(a, b):F6}, max-abs {maxAbs:E2}";
    }

    /// <summary>Where two same-length waveforms differ: the lag (±50 ms) that best aligns them and its correlation against
    /// lag 0, where the largest 10 ms difference window sits, the share of difference energy in the top 5 % of windows (a
    /// transient concentrates it, a drift spreads it), and when the waveforms first part by more than 1e-3.</summary>
    internal static string WhatMoved(float[] a, float[] b, int rate)
    {
        int maxLag = rate / 20;
        (int bestLag, double bestCorr) = (0, Correlation(a, b, 0));
        double zeroCorr = bestCorr;
        for (int lag = -maxLag; lag <= maxLag; lag++)
        {
            double c = Correlation(a, b, lag);
            if (c > bestCorr)
            {
                (bestLag, bestCorr) = (lag, c);
            }
        }
        int window = Math.Max(1, rate / 100);
        int windows = a.Length / window;
        double[] energy = new double[Math.Max(1, windows)];
        double total = 0;
        for (int w = 0; w < windows; w++)
        {
            double e = 0;
            for (int i = w * window; i < (w + 1) * window; i++)
            {
                double d = a[i] - b[i];
                e += d * d;
            }
            energy[w] = e;
            total += e;
        }
        int peak = Array.IndexOf(energy, energy.Max());
        double topShare = total <= 0 ? 0 : energy.OrderByDescending(e => e).Take(Math.Max(1, windows / 20)).Sum() / total;
        int firstApart = Enumerable.Range(0, a.Length).FirstOrDefault(i => Math.Abs(a[i] - b[i]) > 1e-3f, -1);
        return FormattableString.Invariant(
                $"best lag {bestLag} samples ({bestLag * 1000.0 / rate:F1} ms, corr {bestCorr:F4} vs {zeroCorr:F4} at 0); ")
            + FormattableString.Invariant($"peak 10 ms difference at {peak * window / (double)rate:F2} s of {a.Length / (double)rate:F2} s; ")
            + FormattableString.Invariant($"top 5 % of windows hold {topShare:P0} of the difference energy; first parted at ")
            + (firstApart < 0 ? "never" : FormattableString.Invariant($"{firstApart / (double)rate:F2} s"));
    }

    private static double Correlation(float[] a, float[] b, int lag)
    {
        int start = Math.Max(0, -lag), end = Math.Min(a.Length, b.Length - lag);
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        int n = 0;
        for (int i = start; i < end; i++)
        {
            double x = a[i], y = b[i + lag];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            n++;
        }
        if (n == 0)
        {
            return 0;
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        return va > 0 && vb > 0 ? cov / Math.Sqrt(va * vb) : 0;
    }

    private static float[]? Load(string dir, string arm, int index)
    {
        string path = Path.Combine(dir, $"{arm}_{index:D2}.f32");
        return File.Exists(path) ? MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray() : null;
    }

    /// <summary>One arm's sentence from <c>arms.csv</c>: arm,index,rate,samples,first ms,repeat ms,plans,bucket,refs,"text".</summary>
    private sealed record ArmRow(int Rate, double FirstMs, double RepeatMs, string Text);

    private static Dictionary<(string Arm, int Index), ArmRow> ReadArms(string csv)
    {
        Dictionary<(string, int), ArmRow> result = new();
        if (!File.Exists(csv))
        {
            return result;
        }
        foreach (string line in File.ReadAllLines(csv))
        {
            string[] f = line.Split(',', 10);
            if (f.Length < 10)
            {
                continue;
            }
            result[(f[0], int.Parse(f[1], CultureInfo.InvariantCulture))] = new ArmRow(int.Parse(f[2], CultureInfo.InvariantCulture),
                double.Parse(f[4], CultureInfo.InvariantCulture), double.Parse(f[5], CultureInfo.InvariantCulture), f[9].Trim('"'));
        }
        return result;
    }

    [Fact]
    public void WhatMoved_FindsAShift()
    {
        float[] a = new float[4800];
        for (int i = 0; i < a.Length; i++) a[i] = MathF.Sin(i * 0.05f) * MathF.Exp(-((i - 2400) * (i - 2400)) / 2e5f);
        float[] b = new float[a.Length];
        Array.Copy(a, 0, b, 12, a.Length - 12);
        string moved = WhatMoved(a, b, 24_000);
        Assert.Contains("best lag 12 samples", moved, StringComparison.Ordinal);
    }
}
