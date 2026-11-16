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
        // Each read stands alone: a sandboxed /proc that refuses one file must not throw away a figure another file gave.
        long? machine = null, cgroup = null;
        string? mountInfo = null;
        try { machine = MemAvailableBytes(File.ReadAllText("/proc/meminfo")); }
        catch (Exception) { /* the cgroup figure, if any, stands alone */ }
        try { mountInfo = File.ReadAllText("/proc/self/mountinfo"); }
        catch (Exception) { /* each mount is then taken as its tree's root at the usual place */ }
        try { cgroup = CgroupAvailableBytes("/sys/fs/cgroup", File.ReadAllText("/proc/self/cgroup"), mountInfo); }
        catch (Exception) { /* the machine figure stands alone */ }
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
    /// <remarks>The process's group and each ancestor up to the mount are read, and the tightest level wins; a group the mount does not
    /// show (a container whose own group is mounted as the root) is skipped. cgroup v2: the <c>0::</c> line of <c>/proc/self/cgroup</c>
    /// and <c>memory.max</c> under the <c>cgroup2</c> mount. cgroup v1, also on a hybrid host whose v2 tree sets no limit: the
    /// <c>memory</c> controller's line and <c>memory.limit_in_bytes</c> under the <c>cgroup</c> mount whose options name <c>memory</c>
    /// (<c>/sys/fs/cgroup/memory</c>, or a co-mount such as <c>cpu,memory</c>), capped by <c>memory.stat</c>'s
    /// <c>hierarchical_memory_limit</c>. Each level counts its limit minus its working set, the usage less the inactive file cache
    /// (<c>inactive_file</c>, <c>total_inactive_file</c> on v1), which the kernel reclaims before it OOM-kills; that is the working set
    /// the kubelet evicts on. A level that cannot be read is skipped, never the whole answer. The mounts come from
    /// <c>/proc/self/mountinfo</c>, the last matching line (the visible one when mounts stack), and the membership path is made relative
    /// to the tree's root as mounted (that line's fourth field): a mount of the subtree <c>/docker/abc</c> shows the group
    /// <c>/docker/abc/worker</c> at <c>worker</c>. A group outside the mounted subtree, or a mount root outside the cgroup namespace,
    /// reads as the mount itself, which may be an ancestor with a looser limit: the figure is then an upper bound.</remarks>
    /// <param name="root">Where the cgroup tree is mounted when <paramref name="mountInfo"/> lists no cgroup mount: <c>/sys/fs/cgroup</c>
    /// for v2 and its <c>memory</c> directory for v1.</param>
    /// <param name="selfCgroup">The text of <c>/proc/self/cgroup</c>.</param>
    /// <param name="mountInfo">The text of <c>/proc/self/mountinfo</c>, or <c>null</c> to use <paramref name="root"/> as each tree's root.</param>
    internal static long? CgroupAvailableBytes(string root, string selfCgroup, string? mountInfo = null)
    {
        string? v2Group = null, v1Group = null;
        foreach (string line in selfCgroup.Split('\n'))
        {
            string[] parts = line.Trim().Split(':', 3);
            if (parts.Length != 3) continue;
            if (parts[0] == "0" && parts[1].Length == 0) v2Group = parts[2];
            else if (parts[1].Split(',').Contains("memory")) v1Group = parts[2];
        }
        (string Point, string Root) v2Mount = FindMount(mountInfo, "cgroup2", null) ?? (root, "/");
        if (v2Group is not null
            && Walk(v2Mount.Point, MountRelative(v2Group, v2Mount.Root), "memory.max", "memory.current", "inactive_file", null) is { } v2)
        {
            return v2;
        }
        (string Point, string Root) v1Mount = FindMount(mountInfo, "cgroup", "memory") ?? (Path.Combine(root, "memory"), "/");
        return Walk(v1Mount.Point, MountRelative(v1Group ?? "/", v1Mount.Root), "memory.limit_in_bytes", "memory.usage_in_bytes",
            "total_inactive_file", "hierarchical_memory_limit");
    }

    // A limit at or above 2^60 bytes is v1's "no limit" (PAGE_COUNTER_MAX in pages); v2 writes "max".
    private const long NoLimitBytes = 1L << 60;

    /// <summary>The last <c>mountinfo</c> line of filesystem type <paramref name="fsType"/> whose super options name
    /// <paramref name="controller"/> (any, when null): its mount point and the tree path mounted there, or <c>null</c> when none is listed.</summary>
    /// <remarks>A line reads <c>id parent dev root point options [optional...] - fstype source superoptions</c>; paths write a space as
    /// <c>\040</c>.</remarks>
    private static (string Point, string Root)? FindMount(string? mountInfo, string fsType, string? controller)
    {
        if (mountInfo is null) return null;
        (string, string)? found = null;
        foreach (string line in mountInfo.Split('\n'))
        {
            string[] fields = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int dash = Array.IndexOf(fields, "-");
            if (dash < 6 || fields.Length < dash + 4 || fields[dash + 1] != fsType) continue;
            if (controller is not null && !fields[dash + 3].Split(',').Contains(controller)) continue;
            found = (fields[4].Replace("\\040", " "), fields[3].Replace("\\040", " "));
        }
        return found;
    }

    /// <summary><paramref name="group"/> relative to the mounted tree's root <paramref name="mountRoot"/>; a group outside it (one the
    /// mount does not show) reads as the mount itself.</summary>
    private static string MountRelative(string group, string mountRoot)
    {
        string g = "/" + group.Trim().Trim('/'), r = "/" + mountRoot.Trim().Trim('/');
        if (r == "/") return g;
        if (g == r) return "/";
        return g.StartsWith(r + "/", StringComparison.Ordinal) ? g[r.Length..] : "/";
    }

    /// <summary>The tightest level from the group at <paramref name="group"/> (relative to <paramref name="mount"/>) up to the mount. A group
    /// outside the cgroup namespace's root shows as a path with <c>..</c>; it is read as the mount itself.</summary>
    private static long? Walk(string mount, string group, string limitName, string usageName, string inactiveKey, string? hierarchicalKey)
    {
        string relative = group.Trim().Trim('/');
        if (relative.Split('/').Contains("..")) relative = "";
        string top = Path.GetFullPath(mount).TrimEnd(Path.DirectorySeparatorChar);
        string dir = relative.Length == 0 ? top : Path.GetFullPath(Path.Combine(top, relative));
        long? tightest = null;
        while (true)
        {
            tightest = Smaller(tightest, LevelAvailable(dir, limitName, usageName, inactiveKey, hierarchicalKey));
            if (dir.Length <= top.Length) return tightest;
            dir = Path.GetDirectoryName(dir) ?? top;
        }
    }

    private static long? LevelAvailable(string dir, string limitName, string usageName, string inactiveKey, string? hierarchicalKey)
    {
        try
        {
            string limitFile = Path.Combine(dir, limitName), usageFile = Path.Combine(dir, usageName), statFile = Path.Combine(dir, "memory.stat");
            if (!File.Exists(limitFile) || !File.Exists(usageFile)) return null;
            string? stat = File.Exists(statFile) ? File.ReadAllText(statFile) : null;
            long limit = long.TryParse(File.ReadAllText(limitFile).Trim(), out long parsed) ? parsed : long.MaxValue;   // "max": no limit
            if (hierarchicalKey is not null && stat is not null && StatValue(stat, hierarchicalKey) is > 0 and var inherited)
                limit = Math.Min(limit, inherited);
            if (limit >= NoLimitBytes || !long.TryParse(File.ReadAllText(usageFile).Trim(), out long usage)) return null;
            long inactive = stat is null ? 0 : StatValue(stat, inactiveKey);
            long workingSet = Math.Max(0, usage - Math.Min(inactive, usage));
            return Math.Max(0, limit - workingSet);
        }
        catch (Exception)
        {
            // One unreadable level (EACCES on an ancestor, say) must not discard the levels already read.
            return null;
        }
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
