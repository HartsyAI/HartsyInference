using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Core.Logging;

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

    /// <summary>HARTSY_PROFILE_SYNC=1: sync the compute stream on each range Dispose so per-op timing = GPU time.</summary>
    internal static readonly bool ProfileSync =
        Environment.GetEnvironmentVariable("HARTSY_PROFILE_SYNC") == "1";

    /// <summary>HARTSY_PROFILE_FINE=1: enable sub-op ranges (see <see cref="PushFine"/>), off by default.</summary>
    internal static readonly bool ProfileFine =
        Environment.GetEnvironmentVariable("HARTSY_PROFILE_FINE") == "1";

    /// <summary>HARTSY_PROFILE_SHAPES=1: split selected op labels by shape, so one label's total can be attributed
    /// to the call regimes inside it. Off by default — it multiplies the label count.</summary>
    internal static readonly bool ProfileShapes =
        Environment.GetEnvironmentVariable("HARTSY_PROFILE_SHAPES") == "1";

    private static readonly ConcurrentDictionary<string, long[]> _profStats = new();

    /// <summary>Writes the accumulated per-op wall-time table (sorted by total) to <paramref name="path"/>.</summary>
    public static void DumpProfile(string path)
    {
        if (_profStats.IsEmpty) return;
        double freq = Stopwatch.Frequency;
        StringBuilder sb = new();
        sb.AppendLine($"{"op",-26} {"calls",10} {"total_ms",12} {"avg_ms",10}");
        foreach (KeyValuePair<string, long[]> kv in _profStats.OrderByDescending(e => e.Value[1]))
        {
            double totalMs = kv.Value[1] / freq * 1000.0;
            double avgMs = totalMs / Math.Max(1, kv.Value[0]);
            sb.AppendLine($"{kv.Key,-26} {kv.Value[0],10} {totalMs,12:F1} {avgMs,10:F3}");
        }
        File.WriteAllText(path, sb.ToString());
    }

    /// <summary>Drops everything accumulated so far, so the next <see cref="DumpProfile"/> covers only later work.</summary>
    public static void ResetProfile() => _profStats.Clear();

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

    /// <summary>Like <see cref="Push"/> but for sub-op ranges nested inside an op that already pushes one —
    /// a no-op unless <c>HARTSY_PROFILE_FINE=1</c>.</summary>
    /// <remarks>A single op can hide several launches (the Sage attention prologue hides four), and attributing
    /// them needs a range per launch. Those extra pushes are pure overhead in a production run, and this model
    /// launches enough of them per step to be measurable — so they stay off unless explicitly asked for.</remarks>
    public static NvtxRange PushFine(string message) =>
        ProfileFine ? Push(message) : new NvtxRange(active: false);

    /// <summary>Pushes an NVTX range with the given label. The range pops on Dispose.
    /// <paramref name="message"/> must be non-null.</summary>
    public static NvtxRange Push(string message)
    {
        ArgumentNullException.ThrowIfNull(message);
        long profTicks = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
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
            // HARTSY_PROFILE_SYNC=1: drain the compute stream before timestamping so each op's recorded time is its
            // TRUE GPU execution time (not just the async launch cost). Serializes execution — profiling only — but
            // it's the only way, without Nsight, to attribute where GPU time actually goes across async ops.
            // Resolved via the ambient backend State (not a static field): a static stream handle here was
            // exactly the multi-backend poison this file's caches were fixed for elsewhere — with two live
            // backends, one's Dispose clearing (or simply the last-constructed backend's ctor overwriting) a
            // shared field would sync the WRONG backend's stream while attributing the time to this op's caller.
            if (ProfileSync)
            {
                nint stream = GpuTransferHelper.ResolvedStreamHandle;
                if (stream != 0)
                {
                    try { CudaDriverApi.cuStreamSynchronize(stream); }
                    catch (Exception ex) { Logs.Error("[NvtxRange] profile-sync cuStreamSynchronize failed.", ex); }
                }
            }
            long elapsed = Stopwatch.GetTimestamp() - _startTicks;
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
        catch (Exception ex)
        {
            // Already disabled-or-failing; log and swallow so a Dispose path never throws.
            Logs.Error("[NvtxRange] nvtxRangePop failed.", ex);
        }
    }
}
