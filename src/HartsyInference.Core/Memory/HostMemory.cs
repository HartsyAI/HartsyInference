using System.Runtime.InteropServices;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.Memory;

/// <summary>Host (glibc) allocator control. Tensor storage is <see cref="NativeBuffer"/>, i.e. plain <c>aligned_alloc</c>, so a freed weight buffer returns to a glibc arena — not to the OS.</summary>
/// <remarks>Freeing is not the same as returning. glibc only shrinks an arena's <em>topmost</em> heap, so a
/// fully-free 64 MB heap in the middle of an arena's chain keeps its pages resident forever. Rotating model
/// families allocates and frees hundreds of 12-31 MB weight buffers, which strands one such heap after another:
/// measured on a synthetic replay of that pattern, 6.54 G of anon RSS survived freeing every block, and a single
/// <c>malloc_trim(0)</c> took it back to 0.05 G. Under a 52 G cgroup cap that difference is the OOM-kill.
/// <c>malloc_trim</c> walks every arena and MADV_DONTNEEDs the free page ranges, which is exactly the case
/// glibc's own automatic trimming misses.</remarks>
public static class HostMemory
{
    [DllImport("libc", EntryPoint = "malloc_trim", SetLastError = true)]
    private static extern int MallocTrim(nuint pad);

    /// <summary>Returns free arena pages to the OS. Reports the anon-RSS delta, or null if the platform has no <c>malloc_trim</c> (non-glibc) — callers treat that as "nothing to do", never as an error.</summary>
    public static long? Trim()
    {
        long before = AnonymousRssBytes();
        try
        {
            MallocTrim(0);
        }
        catch (EntryPointNotFoundException) { return null; }
        catch (DllNotFoundException) { return null; }
        long after = AnonymousRssBytes();
        return before >= 0 && after >= 0 ? before - after : null;
    }

    /// <summary>Trims and logs the reclaim when it is worth reading. Safe to call from cleanup paths — it never throws.</summary>
    public static void TrimAndLog(string context)
    {
        try
        {
            long? reclaimed = Trim();
            if (reclaimed is > (256L << 20))
                Logs.Info($"[HostMemory] {context}: returned {reclaimed.Value >> 20} MB of arena pages to the OS.");
        }
        catch (Exception error)
        {
            Logs.Debug($"[HostMemory] Trim after {context} failed: {error.Message}");
        }
    }

    /// <summary>Anonymous resident bytes for this process, or -1 when /proc is unreadable.</summary>
    public static long AnonymousRssBytes()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/self/smaps_rollup"))
            {
                if (!line.StartsWith("Anonymous:", StringComparison.Ordinal)) continue;
                string value = line.AsSpan("Anonymous:".Length).Trim().ToString();
                int space = value.IndexOf(' ');
                if (space > 0 && long.TryParse(value[..space], out long kb)) return kb << 10;
                return -1;
            }
        }
        catch (Exception) { }
        return -1;
    }
}
