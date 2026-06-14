using System.Runtime.InteropServices;

namespace HartsyInference.Cuda.Profiling;

/// <summary>P/Invoke into the NVTX shared library (<c>libnvToolsExt.so.1</c> on Linux,
/// <c>nvToolsExt64_1.dll</c> on Windows). Provides scope ranges that show up as colored bars in
/// Nsight Systems timelines.
///
/// <para>NVTX is shipped with the CUDA toolkit; no additional install is required on a system that
/// already has CUDA. If the library is missing the calls fall through silently — the
/// <see cref="NvtxRange"/> wrapper traps the load failure and turns the range into a no-op for the
/// rest of the process lifetime.</para></summary>
internal static partial class NvtxApi
{
    private const string LinuxLib = "libnvToolsExt.so.1";
    private const string WindowsLib = "nvToolsExt64_1";

    [LibraryImport(LinuxLib, EntryPoint = "nvtxRangePushA", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int RangePushA_Linux(string message);

    [LibraryImport(LinuxLib, EntryPoint = "nvtxRangePop")]
    internal static partial int RangePop_Linux();

    [LibraryImport(WindowsLib, EntryPoint = "nvtxRangePushA", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int RangePushA_Windows(string message);

    [LibraryImport(WindowsLib, EntryPoint = "nvtxRangePop")]
    internal static partial int RangePop_Windows();
}
