using System.Runtime.InteropServices;

namespace SharpInference.Cuda.Profiling;

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

    private static int _disabled;
    private static readonly bool _isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    private NvtxRange(bool active)
    {
        _active = active;
    }

    /// <summary>Returns true once any NVTX call has thrown a DllNotFoundException — subsequent
    /// pushes are skipped. Test hook for unit tests.</summary>
    public static bool IsDisabled => Volatile.Read(ref _disabled) != 0;

    /// <summary>Pushes an NVTX range with the given label. The range pops on Dispose.
    /// <paramref name="message"/> must be non-null.</summary>
    public static NvtxRange Push(string message)
    {
        ArgumentNullException.ThrowIfNull(message);
        if (Volatile.Read(ref _disabled) != 0)
        {
            return new NvtxRange(active: false);
        }

        try
        {
            int rc = _isWindows
                ? NvtxApi.RangePushA_Windows(message)
                : NvtxApi.RangePushA_Linux(message);
            // NVTX returns the depth of the range stack on success, or a negative value on failure.
            // Either way, treat it as success for the wrapper.
            _ = rc;
            return new NvtxRange(active: true);
        }
        catch (DllNotFoundException ex)
        {
            // libnvToolsExt isn't on the loader path. Disable for the rest of the process so we
            // don't pay the exception cost on every range.
            if (Interlocked.Exchange(ref _disabled, 1) == 0)
            {
                Console.Error.WriteLine(
                    $"[SharpInference.Cuda.Profiling] NVTX disabled: {ex.GetType().Name}: {ex.Message}");
            }
            return new NvtxRange(active: false);
        }
    }

    /// <summary>Pops the range. Safe to call multiple times.</summary>
    public void Dispose()
    {
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
