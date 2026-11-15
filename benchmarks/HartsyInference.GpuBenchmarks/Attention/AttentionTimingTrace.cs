using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;

namespace HartsyInference.GpuBenchmarks.Attention;

/// <summary>Diagnostic timing trace for the public boundary. Two phases, each in the same process after the caller's warmups:
/// <list type="bullet">
/// <item><description>Single phase: one public operation per trial, decomposed into host dispatch (the call's return),
/// completion wait (<c>Sync</c> after it) and device span (CUDA events on the backend stream from before the call to after the
/// wait). The device span includes device idle gaps that wait for host launches, so it bounds busy device time from above.</description></item>
/// <item><description>Burst phase: 32 back-to-back operations between two events and one wait, divided by 32. This is the
/// device-throughput measure with host launches overlapped, the same quantity the resident boundary reports.</description></item>
/// </list>
/// Each trial also records .NET GC and JIT counters, and per-thread scheduler and fault counters from <c>/proc/thread-self</c>,
/// plus system CPU pressure. Counters are read outside the timed intervals. The output is labelled
/// <c>measurement_mode=diagnostic-trace</c> and is never eligible for acceptance. The trace runs only when <c>--trace-out</c>
/// is given.</summary>
internal static class AttentionTimingTrace
{
    private const double ClockTicksPerSecond = 100.0;   // USER_HZ on Linux; recorded as an assumption in the header.
    private const int BurstCalls = 32;

    public static double[] Run(IBackend backend, Action operation, Func<string?> route, string expectedRoute, int trials,
        string tracePath, string boundary, IReadOnlyDictionary<string, string> header)
    {
        if (backend is not CudaBackend) throw new NotSupportedException("The timing trace is CUDA-only in this revision.");
        if (trials <= 0) throw new ArgumentOutOfRangeException(nameof(trials));
        nint stream = StreamHandle(backend);
        CudaDriverApi.cuEventCreate(out nint startEvent, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint endEvent, 0).ThrowOnError();
        try
        {
            List<Sample> samples = new(trials);
            double[] totals = new double[trials];
            for (int i = 0; i < trials; i++)
            {
                backend.Sync();
                Counters before = Counters.Read();
                CudaDriverApi.cuEventRecord(startEvent, stream).ThrowOnError();
                long t0 = Stopwatch.GetTimestamp();
                operation();
                long t1 = Stopwatch.GetTimestamp();
                backend.Sync();
                long t2 = Stopwatch.GetTimestamp();
                CudaDriverApi.cuEventRecord(endEvent, stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(endEvent).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float deviceMs, startEvent, endEvent).ThrowOnError();
                Counters after = Counters.Read();
                if (route() != expectedRoute) throw new InvalidOperationException("Trace instrumentation changed the executed branch.");
                double total = Stopwatch.GetElapsedTime(t0, t2).TotalMilliseconds;
                totals[i] = total;
                samples.Add(new Sample(i, Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds,
                    Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds, total, deviceMs, Delta(before, after), before.Core, after.Core));
            }
            List<Burst> bursts = new(trials);
            for (int i = 0; i < trials; i++)
            {
                backend.Sync();
                Counters before = Counters.Read();
                CudaDriverApi.cuEventRecord(startEvent, stream).ThrowOnError();
                for (int c = 0; c < BurstCalls; c++) operation();
                CudaDriverApi.cuEventRecord(endEvent, stream).ThrowOnError();
                backend.Sync();
                CudaDriverApi.cuEventSynchronize(endEvent).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float burstMs, startEvent, endEvent).ThrowOnError();
                Counters after = Counters.Read();
                if (route() != expectedRoute) throw new InvalidOperationException("Trace instrumentation changed the executed branch.");
                bursts.Add(new Burst(i, burstMs / BurstCalls, Delta(before, after)));
            }
            WriteTrace(tracePath, boundary, trials, header, samples, bursts);
            return totals;
        }
        finally
        {
            CudaDriverApi.cuEventDestroy(endEvent);
            CudaDriverApi.cuEventDestroy(startEvent);
        }
    }

    private static nint StreamHandle(IBackend backend)
    {
        object stream = backend.GetType().GetField("_stream", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(backend)!;
        return (nint)stream.GetType().GetProperty("Handle")!.GetValue(stream)!;
    }

    private static Deltas Delta(Counters before, Counters after) => new(
        after.Gen0 - before.Gen0, after.Gen1 - before.Gen1, after.Gen2 - before.Gen2,
        after.GcPauseMs - before.GcPauseMs, after.JitMs - before.JitMs, after.JitMethods - before.JitMethods,
        after.ThreadVoluntarySwitches - before.ThreadVoluntarySwitches, after.ThreadInvoluntarySwitches - before.ThreadInvoluntarySwitches,
        after.ThreadMinorFaults - before.ThreadMinorFaults, after.ThreadMajorFaults - before.ThreadMajorFaults,
        after.ThreadRunMs - before.ThreadRunMs, after.ThreadRunQueueWaitMs - before.ThreadRunQueueWaitMs,
        after.PressureCpuSomeUs - before.PressureCpuSomeUs);

    private static void WriteTrace(string path, string boundary, int trials, IReadOnlyDictionary<string, string> header,
        List<Sample> samples, List<Burst> bursts)
    {
        using FileStream file = new(path, FileMode.CreateNew);
        using Utf8JsonWriter writer = new(file, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();
        writer.WriteNumber("schema", 2);
        writer.WriteString("measurement_mode", "diagnostic-trace");
        writer.WriteBoolean("timing_eligible", false);
        writer.WriteString("boundary", boundary);
        writer.WriteNumber("trials", trials);
        writer.WriteNumber("burst_calls", BurstCalls);
        writer.WriteNumber("clock_ticks_per_second_assumed", ClockTicksPerSecond);
        writer.WriteStartObject("header");
        foreach ((string key, string value) in header) writer.WriteString(key, value);
        writer.WriteEndObject();
        writer.WriteStartArray("samples");
        foreach (Sample s in samples)
        {
            writer.WriteStartObject();
            writer.WriteNumber("trial", s.Trial);
            writer.WriteNumber("dispatch_ms", s.DispatchMs);
            writer.WriteNumber("wait_ms", s.WaitMs);
            writer.WriteNumber("total_ms", s.TotalMs);
            writer.WriteNumber("device_span_ms", s.DeviceSpanMs);
            WriteDeltas(writer, s.Deltas);
            writer.WriteNumber("core_before", s.CoreBefore);
            writer.WriteNumber("core_after", s.CoreAfter);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        writer.WriteStartArray("burst_samples");
        foreach (Burst b in bursts)
        {
            writer.WriteStartObject();
            writer.WriteNumber("trial", b.Trial);
            writer.WriteNumber("per_call_ms", b.PerCallMs);
            WriteDeltas(writer, b.Deltas);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    private static void WriteDeltas(Utf8JsonWriter writer, Deltas d)
    {
        writer.WriteNumber("gc_gen0", d.Gen0);
        writer.WriteNumber("gc_gen1", d.Gen1);
        writer.WriteNumber("gc_gen2", d.Gen2);
        writer.WriteNumber("gc_pause_ms", d.GcPauseMs);
        writer.WriteNumber("jit_ms", d.JitMs);
        writer.WriteNumber("jit_methods", d.JitMethods);
        writer.WriteNumber("thread_voluntary_switches", d.VoluntarySwitches);
        writer.WriteNumber("thread_involuntary_switches", d.InvoluntarySwitches);
        writer.WriteNumber("thread_minor_faults", d.MinorFaults);
        writer.WriteNumber("thread_major_faults", d.MajorFaults);
        writer.WriteNumber("thread_run_ms", d.RunMs);
        writer.WriteNumber("thread_runqueue_wait_ms", d.RunQueueWaitMs);
        writer.WriteNumber("cpu_pressure_some_us", d.PressureCpuSomeUs);
    }

    private sealed record Sample(int Trial, double DispatchMs, double WaitMs, double TotalMs, double DeviceSpanMs,
        Deltas Deltas, int CoreBefore, int CoreAfter);

    private sealed record Burst(int Trial, double PerCallMs, Deltas Deltas);

    /// <summary>Counter changes across one timed interval.</summary>
    private readonly record struct Deltas(int Gen0, int Gen1, int Gen2, double GcPauseMs, double JitMs, long JitMethods,
        long VoluntarySwitches, long InvoluntarySwitches, long MinorFaults, long MajorFaults, double RunMs, double RunQueueWaitMs,
        double PressureCpuSomeUs);

    /// <summary>Counter snapshot. Thread-level and file-based counters are read first and the runtime counters last, so the
    /// snapshot's own allocations fall before the runtime counters are read and do not appear in the deltas.</summary>
    private readonly record struct Counters(int Gen0, int Gen1, int Gen2, double GcPauseMs, double JitMs, long JitMethods,
        long ThreadVoluntarySwitches, long ThreadInvoluntarySwitches, long ThreadMinorFaults, long ThreadMajorFaults,
        double ThreadRunMs, double ThreadRunQueueWaitMs, double PressureCpuSomeUs, int Core)
    {
        public static Counters Read()
        {
            (long voluntary, long involuntary) = ReadThreadSwitches();
            (long minor, long major, int core) = ReadThreadStat();
            (double runMs, double waitMs) = ReadThreadSchedstat();
            double pressure = ReadPressure();
            return new Counters(GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2),
                GC.GetTotalPauseDuration().TotalMilliseconds, JitInfo.GetCompilationTime().TotalMilliseconds,
                JitInfo.GetCompiledMethodCount(), voluntary, involuntary, minor, major, runMs, waitMs, pressure, core);
        }

        private static (long Voluntary, long Involuntary) ReadThreadSwitches()
        {
            long voluntary = -1, involuntary = -1;
            foreach (string line in File.ReadLines("/proc/thread-self/status"))
            {
                if (line.StartsWith("voluntary_ctxt_switches:", StringComparison.Ordinal))
                    voluntary = long.Parse(line.AsSpan("voluntary_ctxt_switches:".Length).Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture);
                else if (line.StartsWith("nonvoluntary_ctxt_switches:", StringComparison.Ordinal))
                    involuntary = long.Parse(line.AsSpan("nonvoluntary_ctxt_switches:".Length).Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture);
            }
            return (voluntary, involuntary);
        }

        private static (long Minor, long Major, int Core) ReadThreadStat()
        {
            string text = File.ReadAllText("/proc/thread-self/stat");
            int close = text.LastIndexOf(')');
            string[] fields = text[(close + 2)..].Split(' ');
            // After the command name the fields start at field 3: minflt 10, majflt 12, processor 39.
            return (long.Parse(fields[7], CultureInfo.InvariantCulture), long.Parse(fields[9], CultureInfo.InvariantCulture),
                int.Parse(fields[36], CultureInfo.InvariantCulture));
        }

        private static (double RunMs, double WaitMs) ReadThreadSchedstat()
        {
            string[] fields = File.ReadAllText("/proc/thread-self/schedstat").Split(' ', StringSplitOptions.RemoveEmptyEntries);
            // run_time_ns (time on CPU), run_queue_wait_ns (time runnable but not running), timeslices.
            return (double.Parse(fields[0], CultureInfo.InvariantCulture) / 1e6, double.Parse(fields[1], CultureInfo.InvariantCulture) / 1e6);
        }

        private static double ReadPressure()
        {
            const string path = "/proc/pressure/cpu";
            if (!File.Exists(path)) return double.NaN;
            foreach (string line in File.ReadLines(path))
            {
                if (!line.StartsWith("some ", StringComparison.Ordinal)) continue;
                int at = line.IndexOf("total=", StringComparison.Ordinal);
                if (at >= 0) return double.Parse(line.AsSpan(at + "total=".Length), NumberStyles.Integer, CultureInfo.InvariantCulture);
            }
            return double.NaN;
        }
    }

    /// <summary>Environment and scheduler facts for the header. Read once, before the trials.</summary>
    public static Dictionary<string, string> Header(string route, string fixtureSha, string arm)
    {
        Dictionary<string, string> header = new(StringComparer.Ordinal)
        {
            ["route"] = route,
            ["fixture_sha256"] = fixtureSha,
            ["arm"] = arm,
            ["gc_is_server"] = GCSettings.IsServerGC.ToString(CultureInfo.InvariantCulture),
            ["gc_latency_mode"] = GCSettings.LatencyMode.ToString(),
            ["process_id"] = Environment.ProcessId.ToString(CultureInfo.InvariantCulture),
            ["loadavg_at_start"] = File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim() : "",
            ["cpu_model"] = CpuModel(),
            ["cpu_governor_core0"] = ReadTrimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            ["cpu_pressure_available"] = File.Exists("/proc/pressure/cpu").ToString(CultureInfo.InvariantCulture),
            ["processor_count"] = Environment.ProcessorCount.ToString(CultureInfo.InvariantCulture),
        };
        foreach (System.Collections.DictionaryEntry entry in Environment.GetEnvironmentVariables())
        {
            string name = (string)entry.Key;
            if (name.StartsWith("DOTNET_", StringComparison.Ordinal) || name.StartsWith("COMPlus_", StringComparison.Ordinal))
                header["env." + name] = (string?)entry.Value ?? "";
        }
        return header;
    }

    private static string CpuModel()
    {
        if (!File.Exists("/proc/cpuinfo")) return "";
        foreach (string line in File.ReadLines("/proc/cpuinfo"))
            if (line.StartsWith("model name", StringComparison.Ordinal)) return line[(line.IndexOf(':') + 1)..].Trim();
        return "";
    }

    private static string ReadTrimmed(string path) => File.Exists(path) ? File.ReadAllText(path).Trim() : "";
}
