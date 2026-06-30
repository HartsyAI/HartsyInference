using System.Runtime.InteropServices;

namespace HartsyInference.Cuda.Profiling;

/// <summary>RAII-style NVTX range that shows up in Nsight Systems traces. Use with
/// <c>using</c>:
///
/// <code>
/// using NvtxRange range = NvtxRange.Push("Linear");
/// // ... work ...
/// </code>
///
/// <para>If NVTX isn't available (e.g. running on a non-CUDA machine or in a container without the
/// CUDA toolkit installed), the wrapper detects this on first call and disables itself for the rest
/// of the process lifetime — no exceptions thrown, ranges become no-ops. The first failed call's
/// exception type is logged to stderr once as a debugging hint.</para></summary>
public readonly ref struct NvtxRange
{
    private readonly bool _active;
    private readonly long _startTicks;
    private readonly string? _profName;

    private static int _disabled;
    private static readonly bool _isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    // ── Env-gated CPU wall-time profiler (HARTSY_PROFILE=1). Accumulates per-label (calls, ticks) across all
    // NvtxRange-wrapped ops, regardless of whether NVTX itself is available. Zero overhead when off. ──
    internal static readonly bool ProfileEnabled =
        Environment.GetEnvironmentVariable("HARTSY_PROFILE") == "1";
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long[]> _profStats = new();

    /// <summary>Writes the accumulated per-op wall-time table (sorted by total) to <paramref name="path"/>.</summary>
    public static void DumpProfile(string path)
    {
        if (_profStats.IsEmpty) return;
        double freq = System.Diagnostics.Stopwatch.Frequency;
        System.Text.StringBuilder sb = new();
        sb.AppendLine($"{"op",-26} {"calls",10} {"total_ms",12} {"avg_ms",10}");
        foreach (System.Collections.Generic.KeyValuePair<string, long[]> kv in _profStats.OrderByDescending(e => e.Value[1]))
        {
            double totalMs = kv.Value[1] / freq * 1000.0;
            double avgMs = totalMs / Math.Max(1, kv.Value[0]);
            sb.AppendLine($"{kv.Key,-26} {kv.Value[0],10} {totalMs,12:F1} {avgMs,10:F3}");
        }
        System.IO.File.WriteAllText(path, sb.ToString());
    }

    private NvtxRange(bool active)
    {
        _active = active;
        _startTicks = 0;
        _profName = null;
    }

    private NvtxRange(bool active, long startTicks, string? profName)
    {
        _active = active;
        _startTicks = startTicks;
        _profName = profName;
    }

    /// <summary>Returns true once any NVTX call has thrown a DllNotFoundException — subsequent
    /// pushes are skipped. Test hook for unit tests.</summary>
    public static bool IsDisabled => Volatile.Read(ref _disabled) != 0;

    /// <summary>Pushes an NVTX range with the given label. The range pops on Dispose.
    /// <paramref name="message"/> must be non-null.</summary>
    public static NvtxRange Push(string message)
    {
        ArgumentNullException.ThrowIfNull(message);
        long profTicks = ProfileEnabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        string? profName = ProfileEnabled ? message : null;
        if (Volatile.Read(ref _disabled) != 0)
        {
            return new NvtxRange(active: false, profTicks, profName);
        }

        try
        {
            int rc = _isWindows
                ? NvtxApi.RangePushA_Windows(message)
                : NvtxApi.RangePushA_Linux(message);
            // NVTX returns the depth of the range stack on success, or a negative value on failure.
            // Either way, treat it as success for the wrapper.
            _ = rc;
            return new NvtxRange(active: true, profTicks, profName);
        }
        catch (DllNotFoundException ex)
        {
            // libnvToolsExt isn't on the loader path. Disable for the rest of the process so we
            // don't pay the exception cost on every range.
            if (Interlocked.Exchange(ref _disabled, 1) == 0)
            {
                Console.Error.WriteLine(
                    $"[HartsyInference.Cuda.Profiling] NVTX disabled: {ex.GetType().Name}: {ex.Message}");
            }
            return new NvtxRange(active: false, profTicks, profName);
        }
    }

    /// <summary>Pops the range. Safe to call multiple times.</summary>
    public void Dispose()
    {
        if (_profName is not null)
        {
            long elapsed = System.Diagnostics.Stopwatch.GetTimestamp() - _startTicks;
            long[] slot = _profStats.GetOrAdd(_profName, static _ => new long[2]);
            Interlocked.Increment(ref slot[0]);
            Interlocked.Add(ref slot[1], elapsed);
        }
        if (!_active) return;
        try
        {
            int rc = _isWindows
                ? NvtxApi.RangePop_Windows()
                : NvtxApi.RangePop_Linux();
            _ = rc;
        }
        catch
        {
            // Already disabled-or-failing; swallow so a Dispose path never throws.
        }
    }
}
