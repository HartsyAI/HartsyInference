namespace HartsyInference.Core.Memory.Host;

/// <summary>Best-effort available HOST (not GPU) RAM query, cross-backend diagnostic.</summary>
/// <remarks>Just as relevant to diagnosing a CPU-backend OOM as a cuDNN host-allocation failure on the CUDA
/// backend. Never throws; returns <c>null</c> on any unsupported platform or read failure rather than let a
/// diagnostic query itself become a new failure mode.</remarks>
public static partial class HostMemoryInfo
{
    /// <summary>Available host RAM in bytes, or <c>null</c> if it couldn't be determined.</summary>
    /// <remarks>Linux: the smaller of <c>/proc/meminfo</c>'s <c>MemAvailable</c> (accounts for reclaimable cache/buffers,
    /// not just raw free) and what the process's memory cgroup still allows (<see cref="CgroupAvailableBytes"/>). Inside a
    /// container <c>/proc/meminfo</c> describes the whole machine: a pod capped at 46 GB on a 251 GB host reads about 150 GB
    /// there, so a check against it passes and the container is OOM-killed anyway. Windows: <c>GlobalMemoryStatusEx</c>.
    /// Anything else: unsupported, returns <c>null</c>.</remarks>
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
        long? machine = MemAvailableBytes(File.ReadAllText("/proc/meminfo"));
        long? cgroup = null;
        try
        {
            cgroup = CgroupAvailableBytes("/sys/fs/cgroup", File.ReadAllText("/proc/self/cgroup"));
        }
        catch (Exception)
        {
            // Unreadable cgroup files: the machine figure stands alone rather than losing both.
        }
        return Smaller(machine, cgroup);
    }

    /// <summary><c>MemAvailable</c> from the text of <c>/proc/meminfo</c>, in bytes, or <c>null</c> when it is missing.</summary>
    internal static long? MemAvailableBytes(string meminfo)
    {
        foreach (string line in meminfo.Split('\n'))
        {
            if (line.StartsWith("MemAvailable:", StringComparison.Ordinal))
            {
                string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                return parts.Length >= 2 && long.TryParse(parts[1], out long kb) ? kb * 1024 : null;
            }
        }
        return null;
    }

    /// <summary>What the memory cgroup lets the process still take, in bytes, or <c>null</c> when no limit applies.</summary>
    /// <remarks>cgroup v2: the process's own group (the <c>0::</c> line of <c>/proc/self/cgroup</c>) and each ancestor up to
    /// <paramref name="root"/> that sets <c>memory.max</c>; the tightest wins. cgroup v1: <c>memory/memory.limit_in_bytes</c> under
    /// <paramref name="root"/>. Each level counts its limit minus its working set, the usage less the inactive file cache
    /// (<c>inactive_file</c>, <c>total_inactive_file</c> on v1), which the kernel reclaims before it OOM-kills; that is the
    /// working set the kubelet evicts on. A container with its own cgroup namespace sees its group as the root.</remarks>
    /// <param name="root">The cgroup mount, <c>/sys/fs/cgroup</c> on Linux.</param>
    /// <param name="selfCgroup">The text of <c>/proc/self/cgroup</c>.</param>
    internal static long? CgroupAvailableBytes(string root, string selfCgroup)
    {
        long? tightest = null;
        string? relative = null;
        foreach (string line in selfCgroup.Split('\n'))
        {
            if (line.StartsWith("0::", StringComparison.Ordinal))
            {
                relative = line[3..].Trim().Trim('/');
                break;
            }
        }
        if (relative is not null)
        {
            // cgroup v2: walk from the process's group up to the root; a group the mount does not show is skipped.
            string dir = relative.Length == 0 ? root : Path.Combine(root, relative);
            while (true)
            {
                tightest = Smaller(tightest, LevelAvailable(Path.Combine(dir, "memory.max"), Path.Combine(dir, "memory.current"),
                    Path.Combine(dir, "memory.stat"), "inactive_file"));
                if (string.Equals(Path.GetFullPath(dir).TrimEnd(Path.DirectorySeparatorChar), Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar),
                    StringComparison.Ordinal))
                {
                    break;
                }
                dir = Path.GetDirectoryName(dir) ?? root;
            }
            if (tightest is not null) return tightest;
        }
        string v1 = Path.Combine(root, "memory");
        return LevelAvailable(Path.Combine(v1, "memory.limit_in_bytes"), Path.Combine(v1, "memory.usage_in_bytes"),
            Path.Combine(v1, "memory.stat"), "total_inactive_file");
    }

    // A limit at or above 2^60 bytes is v1's "no limit" (PAGE_COUNTER_MAX in pages); v2 writes "max".
    private const long NoLimitBytes = 1L << 60;

    private static long? LevelAvailable(string limitFile, string usageFile, string statFile, string inactiveKey)
    {
        if (!File.Exists(limitFile) || !File.Exists(usageFile)) return null;
        string limitText = File.ReadAllText(limitFile).Trim();
        if (!long.TryParse(limitText, out long limit) || limit >= NoLimitBytes) return null;
        if (!long.TryParse(File.ReadAllText(usageFile).Trim(), out long usage)) return null;
        long inactive = File.Exists(statFile) ? StatValue(File.ReadAllText(statFile), inactiveKey) : 0;
        long workingSet = Math.Max(0, usage - Math.Min(inactive, usage));
        return Math.Max(0, limit - workingSet);
    }

    private static long StatValue(string stat, string key)
    {
        foreach (string line in stat.Split('\n'))
        {
            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 2 && parts[0] == key && long.TryParse(parts[1], out long value)) return value;
        }
        return 0;
    }

    private static long? Smaller(long? a, long? b) => a is null ? b : b is null ? a : Math.Min(a.Value, b.Value);

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
