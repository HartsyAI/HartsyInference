namespace HartsyInference.Core.Runtime;

/// <summary>Best-effort available HOST (not GPU) RAM query, cross-backend diagnostic.</summary>
/// <remarks>Just as relevant to diagnosing a CPU-backend OOM as a cuDNN host-allocation failure on the CUDA
/// backend. Never throws; returns <c>null</c> on any unsupported platform or read failure rather than let a
/// diagnostic query itself become a new failure mode.</remarks>
public static partial class HostMemoryInfo
{
    /// <summary>Available host RAM in bytes, or <c>null</c> if it couldn't be determined.</summary>
    /// <remarks>Linux: parses <c>/proc/meminfo</c>'s <c>MemAvailable</c> (accounts for reclaimable cache/buffers,
    /// not just raw free — the same field this engine's test suite already probes for its own memory-gated
    /// tests). Windows: <c>GlobalMemoryStatusEx</c>. Anything else: unsupported, returns <c>null</c>.</remarks>
    public static long? AvailableBytes()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return AvailableBytesWindows();
            }
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return AvailableBytesLinux();
            }
        }
        catch
        {
            // Best-effort — a failed diagnostic probe should never itself become a new failure.
        }
        return null;
    }

    private static long? AvailableBytesLinux()
    {
        string meminfo = File.ReadAllText("/proc/meminfo");
        foreach (string line in meminfo.Split('\n'))
        {
            if (line.StartsWith("MemAvailable:", StringComparison.Ordinal))
            {
                string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2 && long.TryParse(parts[1], out long kb))
                {
                    return kb * 1024;
                }
                return null;
            }
        }
        return null;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MEMORYSTATUSEX
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX lpBuffer);

    private static long? AvailableBytesWindows()
    {
        MEMORYSTATUSEX status = new() { dwLength = (uint)Marshal.SizeOf<MEMORYSTATUSEX>() };
        return GlobalMemoryStatusEx(ref status) ? (long)status.ullAvailPhys : null;
    }
}
