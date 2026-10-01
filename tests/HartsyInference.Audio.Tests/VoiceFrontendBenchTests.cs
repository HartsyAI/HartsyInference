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
using Microsoft.Win32.SafeHandles;
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
/// <para>Two opt-in conditions show what the budget meets on a busy host. They are characterization runs: the log
/// reports what they measure, and the 2 ms gate is asserted only without them.
/// <list type="bullet">
/// <item><c>HARTSY_VOICE_FRONTEND_BENCH_LOAD_THREADS=N</c> runs N background threads, each pinned to its own CPU
/// outside the bench core's hyperthread pair, one per physical core first and then on siblings. By default each runs
/// a STREAM triad over buffers several times the L3, which costs memory bandwidth and evicts what the bench keeps in
/// L3.</item>
/// <item><c>HARTSY_VOICE_FRONTEND_BENCH_LOAD=l1</c> runs the same loop over buffers that fit in L1 instead. That is
/// the same instructions on the same load threads, minus the memory traffic, which separates turbo and power effects
/// from cache and bandwidth ones.</item>
/// <item><c>HARTSY_VOICE_FRONTEND_BENCH_PACED=1</c> releases one frame every 20 ms on a fixed clock, as the live stream
/// does, so other cores get the gap to evict the bench's cache lines. A frame that overruns delays the next ones, which
/// then run back to back until the clock is caught up, as queued audio would; the log counts the frames that started
/// late. Frame times run from when a frame starts, so they hold its processing only, not a late frame's wait behind
/// earlier overruns. It spins between frames rather than sleeping, which keeps the core's clock ramp and C-state exits
/// out of the figure.</item>
/// </list>
/// The log also reports the load's CPUs, its bandwidth over the timed frames, and the bench core's clock sampled from
/// cpufreq over the same frames.</para>
///
/// <para>The budget is an open gate that this box does not meet yet. At alpha.227, back to back, three runs gave p50
/// 1.57–1.70 ms and p99 3.72–3.92 ms. Paced, p50 is 2.61 ms, and under streaming load the frame time grows
/// with the weight bytes each frame reads (see CHANGELOG). Until the weights' precision and cache residency are
/// settled, a p99 failure in the default mode is that open gate. A regression shows up instead as a higher p50, or as
/// any allocation or GC while timed.</para></summary>
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

        int requestedLoad = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_VOICE_FRONTEND_BENCH_LOAD_THREADS"),
            out int n) ? Math.Max(0, n) : 0;
        bool l1Load = Environment.GetEnvironmentVariable("HARTSY_VOICE_FRONTEND_BENCH_LOAD") == "l1";
        bool paced = Environment.GetEnvironmentVariable("HARTSY_VOICE_FRONTEND_BENCH_PACED") == "1";
        List<int> others = OtherCpus(cpu);
        int[] loadCpus = [.. others.Take(requestedLoad)];
        int samplerCpu = others.Count > loadCpus.Length ? others[^1] : -1;

        // Disposed on the way out whatever happens, so a failed bench still stops and joins the load.
        using BackgroundLoad load = new(loadCpus, l1Load);
        using ClockSampler clock = new(cpu, samplerCpu);
        Result? result = null;
        Exception? failure = null;
        Thread thread = new(() =>
        {
            try
            {
                result = Measure(weights, sileroPath, audio, cpu, paced, clock, load);
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        });
        thread.Start();
        thread.Join();
        load.Stop();
        clock.Stop();
        if (failure is not null) throw new InvalidOperationException("front-end benchmark thread failed", failure);

        Result r = result!;
        (double p50, double p99, double max) = Percentiles(r.FrameNs);
        log.WriteLine($"{TimedFrames} frames of 20 ms, wall: p50 {p50:F3} ms, p99 {p99:F3} ms, max {max:F3} ms");
        if (r.CpuNs is not null)
        {
            (double c50, double c99, double cMax) = Percentiles(r.CpuNs);
            log.WriteLine($"  thread CPU time: p50 {c50:F3} ms, p99 {c99:F3} ms, max {cMax:F3} ms");
        }
        foreach ((string label, double ms) in r.Stages)
            log.WriteLine($"  {label}: {ms:F3} ms/frame mean");
        string loadText = loadCpus.Length == 0 ? "none"
            : $"{loadCpus.Length} × {(l1Load ? "L1-resident" : "streaming")} triad on CPUs {string.Join(',', loadCpus)}, "
                + $"{load.GigabytesPerSecond:F1} GB/s over the timed frames";
        if (loadCpus.Length < requestedLoad) loadText += $" ({requestedLoad} requested, {others.Count} CPUs free)";
        string pacing = paced
            ? $"paced on a 20 ms clock, spinning between, {r.LateFrames} of {TimedFrames} started late"
            : "back to back";
        log.WriteLine($"background load: {loadText}; frames {pacing}");
        string clockText = clock.Samples == 0 ? "core clock not sampled"
            : $"core clock while timed {clock.MeanMhz:F0} MHz mean, {clock.MinMhz:F0} MHz min over {clock.Samples} "
                + $"samples from {(clock.Pinned ? $"CPU {samplerCpu}" : "an unpinned thread")}";
        log.WriteLine($"pinned to CPU {cpu}: {(r.Pinned ? "yes" : "no, " + r.PinReason)}; {clockText}; RNNoise network "
            + $"ran on {r.NetworkFrames}/{TimedFrames} frames; {r.AllocatedBytes} bytes allocated, {r.Gen0Collections} "
            + "gen-0 GCs while timed; "
            + (loadCpus.Length == 0 && !paced ? $"process CPU / wall {r.CpuOverWall:F2}; " : "")
            + $"load average {File.ReadAllText("/proc/loadavg").Trim()}");

        // A frame whose gains saturate can repeat the previous speech probability exactly, so this is a floor
        // against the silence-floor passthrough rather than an exact count.
        Assert.True(r.NetworkFrames >= TimedFrames * 0.95,
            $"RNNoise ran its network on only {r.NetworkFrames}/{TimedFrames} frames");
        if (loadCpus.Length > 0 || paced)
        {
            log.WriteLine("characterization run (load or pacing set): the 2 ms gate is asserted only without them");
            return;
        }
        Assert.True(p99 <= BudgetMs,
            $"p99 {p99:F3} ms exceeds the {BudgetMs} ms front-end budget (p50 {p50:F3}, max {max:F3})");
    }

    private static Result Measure(RnnoiseWeights weights, string sileroPath, float[] audio, int cpu, bool paced,
        ClockSampler clock, BackgroundLoad load)
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
        int lateFrames = 0;
        float lastProbability = float.NaN;
        long allocatedBefore = 0, wallStart = 0;
        int gen0Before = 0;
        TimeSpan cpuStart = TimeSpan.Zero;
        long period = Stopwatch.Frequency / 50;
        long nextFrame = Stopwatch.GetTimestamp();

        for (int f = 0; f < WarmupFrames + TimedFrames; f++)
        {
            int timed = f - WarmupFrames;
            if (timed == 0)
            {
                cpuStart = Process.GetCurrentProcess().TotalProcessorTime;
                gen0Before = GC.CollectionCount(0);
                clock.Begin();
                load.Begin();
                wallStart = Stopwatch.GetTimestamp();
                allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
            }
            if (paced)
            {
                // A fixed clock, not a gap after each frame: an overrun delays the next frames, which then run back
                // to back until the clock is caught up, as queued audio would.
                nextFrame += period;
                if (timed >= 0 && Stopwatch.GetTimestamp() >= nextFrame) lateFrames++;
                while (Stopwatch.GetTimestamp() < nextFrame) Thread.SpinWait(16);
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
        load.End();
        clock.End();
        int collections = GC.CollectionCount(0) - gen0Before;
        double cpuOverWall = (Process.GetCurrentProcess().TotalProcessorTime - cpuStart).TotalSeconds
            / (wallTicks / (double)Stopwatch.Frequency);
        (string, double)[] stages =
        [
            ("scale + RNNoise (2 x 10 ms, 16k <-> 48k)", StageMs(stageTicks[0])),
            ("rescale into Silero window", StageMs(stageTicks[1])),
            ("Silero 512-sample chunk + endpointing", StageMs(stageTicks[2])),
        ];
        return new Result(frameNs, cpuNs, stages, networkFrames, lateFrames, pinned, pinReason, allocated, collections,
            cpuOverWall);
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
            foreach (int s in Siblings(i))
            {
                if (s < count) score = Math.Min(score, idle[s]);
            }
            if (score > bestScore)
            {
                bestScore = score;
                best = i;
            }
        }
        return best;
    }

    /// <summary>Every CPU outside <paramref name="benchCpu"/>'s hyperthread pair: one per physical core first, then
    /// their siblings, so N load threads land on N distinct cores while there are enough of them.</summary>
    private static List<int> OtherCpus(int benchCpu)
    {
        int[] benchPair = Siblings(benchCpu);
        List<int> firsts = [];
        List<int> seconds = [];
        for (int i = 0; i < Environment.ProcessorCount; i++)
        {
            if (benchPair.Contains(i)) continue;
            (Siblings(i).Min() == i ? firsts : seconds).Add(i);
        }
        return [.. firsts, .. seconds];
    }

    /// <summary>The logical CPUs sharing <paramref name="cpu"/>'s physical core, itself included.</summary>
    private static int[] Siblings(int cpu)
    {
        List<int> siblings = [cpu];
        string path = $"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list";
        if (!File.Exists(path)) return [.. siblings];
        foreach (string part in File.ReadAllText(path).Trim().Split(','))
        {
            // Some topologies list a pair as a range, "0-1".
            string[] bounds = part.Split('-');
            if (!int.TryParse(bounds[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int first)) continue;
            int last = bounds.Length > 1
                && int.TryParse(bounds[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int end) ? end : first;
            for (int s = first; s <= last; s++)
            {
                if (!siblings.Contains(s)) siblings.Add(s);
            }
        }
        return [.. siblings];
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
        int LateFrames, bool Pinned, string PinReason, long AllocatedBytes, int Gen0Collections, double CpuOverWall);

    /// <summary>Background threads standing in for other work on the host during a call, each pinned to its own CPU.
    /// Each runs a STREAM triad, <c>c = a + s·b</c>. Over three 32 MB arrays it costs memory bandwidth and, on any L3
    /// smaller than 96 MB, evicts what the bench keeps there; over three 4 KB arrays it stays in L1 and costs only the
    /// busy core. The buffers are native, so the load allocates nothing managed while the bench is timed.</summary>
    private sealed unsafe class BackgroundLoad : IDisposable
    {
        private const int StreamFloats = 8 << 20;
        private const int L1Floats = 1 << 10;
        // Each thread publishes its byte count in steps this size: fine enough that a seconds-long timed window reads
        // it to well under 1 %, coarse enough that eight L1-speed threads do not contend on the counter.
        private const long PublishBytes = 64L << 20;
        private readonly Thread[] _threads;
        private volatile bool _stop;
        private long _bytes;
        private long _beginBytes;
        private long _beginTicks;
        private long _windowBytes;
        private long _windowTicks;

        public BackgroundLoad(int[] cpus, bool l1)
        {
            int floats = l1 ? L1Floats : StreamFloats;
            _threads = new Thread[cpus.Length];
            for (int i = 0; i < cpus.Length; i++)
            {
                int cpu = cpus[i];
                _threads[i] = new Thread(() => Triad(cpu, floats)) { IsBackground = true };
                _threads[i].Start();
            }
        }

        /// <summary>Aggregate bandwidth between <see cref="Begin"/> and <see cref="End"/>, by the STREAM convention of
        /// 12 bytes per element. With write-allocate, the DRAM traffic of the streaming load is about a third more.</summary>
        public double GigabytesPerSecond =>
            _windowTicks > 0 ? _windowBytes / (_windowTicks / (double)Stopwatch.Frequency) / 1e9 : 0;

        /// <summary>Opens the window the bandwidth is reported over: the timed frames, not buffer setup or warm-up.</summary>
        public void Begin()
        {
            _beginBytes = Interlocked.Read(ref _bytes);
            _beginTicks = Stopwatch.GetTimestamp();
        }

        public void End()
        {
            _windowBytes = Interlocked.Read(ref _bytes) - _beginBytes;
            _windowTicks = Stopwatch.GetTimestamp() - _beginTicks;
        }

        public void Stop()
        {
            if (_stop) return;
            _stop = true;
            foreach (Thread thread in _threads) thread.Join();
        }

        public void Dispose() => Stop();

        private void Triad(int cpu, int floats)
        {
            RealtimeScheduling.TryPinToCpu(cpu, out _);
            nuint bytes = (nuint)floats * sizeof(float);
            float* a = (float*)NativeMemory.AlignedAlloc(bytes, 64);
            float* b = (float*)NativeMemory.AlignedAlloc(bytes, 64);
            float* c = (float*)NativeMemory.AlignedAlloc(bytes, 64);
            long pending = 0;
            try
            {
                for (int i = 0; i < floats; i++)
                {
                    a[i] = 1f;
                    b[i] = 2f;
                    c[i] = 0f;
                }
                Vector256<float> scale = Vector256.Create(0.5f);
                while (!_stop)
                {
                    for (int i = 0; i < floats; i += Vector256<float>.Count)
                        Vector256.StoreAligned(Vector256.LoadAligned(a + i) + scale * Vector256.LoadAligned(b + i), c + i);
                    pending += 3L * floats * sizeof(float);
                    if (pending < PublishBytes) continue;
                    Interlocked.Add(ref _bytes, pending);
                    pending = 0;
                }
            }
            finally
            {
                Interlocked.Add(ref _bytes, pending);
                NativeMemory.AlignedFree(a);
                NativeMemory.AlignedFree(b);
                NativeMemory.AlignedFree(c);
            }
        }
    }

    /// <summary>Samples one CPU's clock from cpufreq every 50 ms, between <see cref="Begin"/> and <see cref="End"/>, on
    /// a thread pinned off the bench pair and the load. Where the cpufreq driver derives the figure from APERF/MPERF
    /// over the last tick, as intel_pstate and intel_cpufreq do, it is the clock the core ran at rather than the
    /// governor's request; other drivers may report the request. It reads into a reused buffer, so it allocates nothing
    /// managed while the bench is timed.</summary>
    private sealed class ClockSampler : IDisposable
    {
        private readonly Thread? _thread;
        private volatile bool _recording;
        private volatile bool _stop;
        private long _sumKhz;
        private long _minKhz = long.MaxValue;

        public ClockSampler(int cpu, int samplerCpu)
        {
            string path = $"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq";
            if (!OperatingSystem.IsLinux() || !File.Exists(path)) return;
            _thread = new Thread(() => Run(path, samplerCpu)) { IsBackground = true };
            _thread.Start();
        }

        public int Samples { get; private set; }

        /// <summary>Whether the sampling thread got its own CPU; with none free it runs unpinned, waking every 50 ms
        /// wherever the scheduler puts it, possibly beside the bench.</summary>
        public bool Pinned { get; private set; }

        public double MeanMhz => Samples > 0 ? _sumKhz / (double)Samples / 1000 : double.NaN;

        public double MinMhz => Samples > 0 ? _minKhz / 1000.0 : double.NaN;

        public void Begin() => _recording = true;

        public void End() => _recording = false;

        public void Stop()
        {
            _stop = true;
            _thread?.Join();
        }

        public void Dispose() => Stop();

        private void Run(string path, int samplerCpu)
        {
            Pinned = samplerCpu >= 0 && RealtimeScheduling.TryPinToCpu(samplerCpu, out _);
            using SafeFileHandle handle = File.OpenHandle(path);
            byte[] buffer = new byte[32];
            while (!_stop)
            {
                if (_recording)
                {
                    int read = RandomAccess.Read(handle, buffer, 0);
                    long khz = 0;
                    for (int i = 0; i < read && buffer[i] >= (byte)'0' && buffer[i] <= (byte)'9'; i++)
                        khz = khz * 10 + (buffer[i] - '0');
                    if (khz > 0)
                    {
                        _sumKhz += khz;
                        _minKhz = Math.Min(_minKhz, khz);
                        Samples++;
                    }
                }
                Thread.Sleep(50);
            }
        }
    }
}
