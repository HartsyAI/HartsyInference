using System.Diagnostics;
using System.Globalization;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>The voice agent's front-end budget: RNNoise over one 20 ms frame of 16 kHz audio plus one Silero VAD
/// chunk, serial on one core, must fit in 2 ms.
///
/// <para>Conservative on purpose: Silero consumes 512 samples, so in the real stream it runs on 0.625 of frames, and
/// here it runs on every one. The work is what the audio thread does per frame, scale conversions included: ±1 in,
/// ×32768 for RNNoise, which keeps upstream's absolute thresholds, and back to ±1 for Silero. The thread pins itself
/// to one CPU and enters <see cref="CpuParallel.EnterInline"/>, so no kernel can fan out. The clip is jfk.wav with
/// white noise mixed in, so RNNoise never meets a frame quiet enough to skip its network.</para>
///
/// <para>Asserts wall-clock p99 ≤ 2 ms. Beside it, the thread's own CPU time per frame (Linux) separates the work
/// from time the scheduler gave to something else — the voice host runs this thread SCHED_FIFO on a reserved core,
/// a desktop test run cannot — along with allocations, GC count and process CPU time against wall time. Opt in with
/// <c>HARTSY_VOICE_FRONTEND_BENCH=1</c>. The core is the one whose hyperthread pair was idlest while the weights
/// loaded, unless <c>HARTSY_VOICE_FRONTEND_BENCH_CPU</c> names one. Needs the RNNoise and Silero weights under the
/// wake model root, or <c>HARTSYINFERENCE_RNNOISE_WEIGHTS</c> and <c>HARTSYINFERENCE_SILERO_WEIGHTS</c>. Run it
/// alone: any other benchmark or test run on the box moves it.</para>
///
/// <para>The p99 budget is an open gate that this box does not meet yet: at alpha.227, three runs gave p50
/// 1.55–1.56 ms and p99 3.81–3.93 ms (see CHANGELOG). Until the GRU weight precision or the SCHED_FIFO measurement
/// settles it, a p99 failure here is that open gate. A regression shows up instead as a higher p50, or as any
/// allocation or GC while timed.</para></summary>
public sealed partial class VoiceFrontendBenchTests(ITestOutputHelper log)
{
    private const int Rate = 16_000;
    private const int FrameSamples = Rate / 50;
    // Long enough for tiered compilation to settle every method on the path before timing starts.
    private const int WarmupFrames = 500;
    private const int TimedFrames = 2_000;
    private const double BudgetMs = 2.0;
    private const float Int16Scale = 32768f;
    private const int ClockThreadCpuTime = 3;

    [Fact]
    [Trait("Category", "Integration")]
    public void SileroPlusRnnoise_PerTwentyMsFrame_OnOneCore_FitsTwoMilliseconds()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_VOICE_FRONTEND_BENCH") != "1")
        {
            log.WriteLine("SKIPPED: set HARTSY_VOICE_FRONTEND_BENCH=1 to run this benchmark");
            return;
        }
        string rnnoisePath = RnnoiseRealSpeechTests.WeightsPath();
        string sileroPath = Environment.GetEnvironmentVariable("HARTSYINFERENCE_SILERO_WEIGHTS")
            ?? Path.Combine(TestPaths.ModelsDir, "audio", "wake", "vad", "silero_vad_16k.safetensors");
        string clipPath = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(log.WriteLine, rnnoisePath, sileroPath, clipPath)) return;

        // Loading takes long enough to be the idle sample window; nothing sleeps for it.
        long[][]? before = OperatingSystem.IsLinux() ? ReadCpuTimes() : null;
        float[] audio = LoopWithNoise(WavFile.Read(clipPath).ToMono(), (WarmupFrames + TimedFrames) * FrameSamples);
        using RnnoiseWeights weights = LoadRnnoise(rnnoisePath);
        int cpu = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_VOICE_FRONTEND_BENCH_CPU"), out int c) ? c
            : before is null ? Environment.ProcessorCount - 1 : IdlestCpu(before, ReadCpuTimes());

        Result? result = null;
        Exception? failure = null;
        Thread thread = new(() =>
        {
            try
            {
                result = Measure(weights, sileroPath, audio, cpu);
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        });
        thread.Start();
        thread.Join();
        if (failure is not null) throw new InvalidOperationException("front-end benchmark thread failed", failure);

        Result r = result!;
        (double p50, double p99, double max) = Percentiles(r.FrameNs);
        log.WriteLine($"{TimedFrames} frames of 20 ms, wall: p50 {p50:F3} ms, p99 {p99:F3} ms, max {max:F3} ms");
        if (r.CpuNs is not null)
        {
            (double c50, double c99, double cMax) = Percentiles(r.CpuNs);
            log.WriteLine($"  thread CPU time: p50 {c50:F3} ms, p99 {c99:F3} ms, max {cMax:F3} ms");
        }
        log.WriteLine($"pinned to CPU {cpu}: {(r.Pinned ? "yes" : "no, " + r.PinReason)}; RNNoise network ran on "
            + $"{r.NetworkFrames}/{TimedFrames} frames; {r.AllocatedBytes} bytes allocated, {r.Gen0Collections} gen-0 GCs "
            + $"while timed; process CPU / wall {r.CpuOverWall:F2}; load average {File.ReadAllText("/proc/loadavg").Trim()}");
        foreach ((string label, double ms) in r.Stages)
            log.WriteLine($"  {label}: {ms:F3} ms/frame mean");

        // A frame whose gains saturate can repeat the previous speech probability exactly, so this is a floor
        // against the silence-floor passthrough rather than an exact count.
        Assert.True(r.NetworkFrames >= TimedFrames * 0.95,
            $"RNNoise ran its network on only {r.NetworkFrames}/{TimedFrames} frames");
        Assert.True(p99 <= BudgetMs,
            $"p99 {p99:F3} ms exceeds the {BudgetMs} ms front-end budget (p50 {p50:F3}, max {max:F3})");
    }

    private static Result Measure(RnnoiseWeights weights, string sileroPath, float[] audio, int cpu)
    {
        bool pinned = RealtimeScheduling.TryPinToCpu(cpu, out string pinReason);
        bool threadClock = OperatingSystem.IsLinux();
        using CpuParallel.InlineScope inline = CpuParallel.EnterInline();
        using CpuBackend backend = new();
        using RnnoiseStream denoiser = new(weights, Rate);
        using SileroVad vad = LoadSilero(sileroPath);
        SileroVadStream endpointer = new(vad);

        float[] scaled = new float[FrameSamples];
        float[] denoised = new float[FrameSamples + denoiser.FrameSize];
        float[] window = new float[SileroVad.WindowSamples];
        long[] frameNs = new long[TimedFrames];
        long[]? cpuNs = threadClock ? new long[TimedFrames] : null;
        long[] stageTicks = new long[3];
        int networkFrames = 0;
        float lastProbability = float.NaN;
        long allocatedBefore = 0, wallStart = 0;
        int gen0Before = 0;
        TimeSpan cpuStart = TimeSpan.Zero;

        for (int f = 0; f < WarmupFrames + TimedFrames; f++)
        {
            int timed = f - WarmupFrames;
            if (timed == 0)
            {
                cpuStart = Process.GetCurrentProcess().TotalProcessorTime;
                gen0Before = GC.CollectionCount(0);
                wallStart = Stopwatch.GetTimestamp();
                allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
            }
            ReadOnlySpan<float> frame = audio.AsSpan(f * FrameSamples, FrameSamples);
            long c0 = threadClock ? ThreadCpuNs() : 0;
            long t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < FrameSamples; i++) scaled[i] = frame[i] * Int16Scale;
            int written = denoiser.Process(backend, scaled, denoised);
            long t1 = Stopwatch.GetTimestamp();
            // Newest 512 denoised samples, back at ±1: Silero's scale.
            window.AsSpan(written).CopyTo(window);
            for (int i = 0; i < written; i++) window[SileroVad.WindowSamples - written + i] = denoised[i] / Int16Scale;
            long t2 = Stopwatch.GetTimestamp();
            endpointer.Push(backend, window, out _);
            long t3 = Stopwatch.GetTimestamp();
            long c1 = threadClock ? ThreadCpuNs() : 0;
            if (timed < 0) continue;
            frameNs[timed] = (long)((t3 - t0) * (1e9 / Stopwatch.Frequency));
            if (cpuNs is not null) cpuNs[timed] = c1 - c0;
            stageTicks[0] += t1 - t0;
            stageTicks[1] += t2 - t1;
            stageTicks[2] += t3 - t2;
            if (denoiser.SpeechProbability != lastProbability) networkFrames++;
            lastProbability = denoiser.SpeechProbability;
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
        long wallTicks = Stopwatch.GetTimestamp() - wallStart;
        int collections = GC.CollectionCount(0) - gen0Before;
        double cpuOverWall = (Process.GetCurrentProcess().TotalProcessorTime - cpuStart).TotalSeconds
            / (wallTicks / (double)Stopwatch.Frequency);
        (string, double)[] stages =
        [
            ("scale + RNNoise (2 x 10 ms, 16k <-> 48k)", StageMs(stageTicks[0])),
            ("rescale into Silero window", StageMs(stageTicks[1])),
            ("Silero 512-sample chunk + endpointing", StageMs(stageTicks[2])),
        ];
        return new Result(frameNs, cpuNs, stages, networkFrames, pinned, pinReason, allocated, collections, cpuOverWall);
    }

    /// <summary>The CPU whose hyperthread pair spent the largest share of the interval idle: a busy sibling shares
    /// the core's execution units and caches, so the pair, not the logical CPU, is what has to be quiet.</summary>
    private static int IdlestCpu(long[][] before, long[][] after)
    {
        int count = Math.Min(before.Length, after.Length);
        double[] idle = new double[count];
        for (int i = 0; i < count; i++)
        {
            long total = after[i][1] - before[i][1];
            idle[i] = total > 0 ? (after[i][0] - before[i][0]) / (double)total : 0;
        }
        int best = count - 1;
        double bestScore = double.MinValue;
        for (int i = 0; i < count; i++)
        {
            double score = idle[i];
            string siblings = $"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list";
            if (File.Exists(siblings))
            {
                foreach (string part in File.ReadAllText(siblings).Trim().Split(','))
                {
                    if (int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out int s) && s != i
                        && s < count)
                        score = Math.Min(score, idle[s]);
                }
            }
            if (score > bestScore)
            {
                bestScore = score;
                best = i;
            }
        }
        return best;
    }

    /// <summary>Per-CPU (idle, total) jiffies from <c>/proc/stat</c>, indexed by CPU number.</summary>
    private static long[][] ReadCpuTimes()
    {
        List<long[]> cpus = [];
        foreach (string line in File.ReadLines("/proc/stat"))
        {
            if (!line.StartsWith("cpu", StringComparison.Ordinal) || line.Length < 4 || !char.IsDigit(line[3])) continue;
            string[] fields = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            long total = 0;
            for (int i = 1; i < fields.Length; i++) total += long.Parse(fields[i], CultureInfo.InvariantCulture);
            // idle + iowait
            long idle = long.Parse(fields[4], CultureInfo.InvariantCulture)
                + long.Parse(fields[5], CultureInfo.InvariantCulture);
            cpus.Add([idle, total]);
        }
        return [.. cpus];
    }

    private static long ThreadCpuNs()
    {
        ClockGetTime(ClockThreadCpuTime, out Timespec now);
        return now.Seconds * 1_000_000_000L + now.Nanoseconds;
    }

    [LibraryImport("libc", EntryPoint = "clock_gettime")]
    private static partial int ClockGetTime(int clockId, out Timespec time);

    private static (double P50, double P99, double Max) Percentiles(long[] ns)
    {
        long[] sorted = [.. ns];
        Array.Sort(sorted);
        return (sorted[sorted.Length / 2] / 1e6, sorted[(int)(sorted.Length * 0.99)] / 1e6, sorted[^1] / 1e6);
    }

    /// <summary>The clip repeated to <paramref name="length"/> samples with white noise ~30 dB under the speech, so
    /// every frame carries energy.</summary>
    private static float[] LoopWithNoise(float[] clip, int length)
    {
        Random rng = new(99);
        float[] audio = new float[length];
        for (int i = 0; i < length; i++)
            audio[i] = clip[i % clip.Length] + (float)((rng.NextDouble() * 2 - 1) * 0.01);
        return audio;
    }

    private static RnnoiseWeights LoadRnnoise(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        RnnoiseWeights weights = new();
        weights.Load(tensors);
        foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        return weights;
    }

    private static SileroVad LoadSilero(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        SileroVad vad = new();
        vad.LoadWeights(loader.GetAllTensors());
        return vad;
    }

    private static double StageMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency / TimedFrames;

    [StructLayout(LayoutKind.Sequential)]
    private struct Timespec
    {
        public long Seconds;
        public long Nanoseconds;
    }

    private sealed record Result(long[] FrameNs, long[]? CpuNs, (string Label, double Ms)[] Stages, int NetworkFrames,
        bool Pinned, string PinReason, long AllocatedBytes, int Gen0Collections, double CpuOverWall);
}
