using System.Runtime.Versioning;

namespace HartsyInference.Core.Runtime;

/// <summary>Puts the calling thread on a real-time scheduling class or a fixed CPU, reporting why when it cannot.</summary>
/// <remarks>Linux ignores <see cref="Thread.Priority"/> entirely: every managed thread runs under <c>SCHED_OTHER</c>
/// and the CFS scheduler, so a media thread that must wake on time has to ask for <c>SCHED_FIFO</c> explicitly. Both
/// calls act on the calling thread only (Linux's <c>sched_setscheduler</c> and <c>sched_setaffinity</c> take a thread
/// id, and 0 means the caller), never throw, and on failure return a reason a person can act on: the usual one is the
/// <c>RLIMIT_RTPRIO</c> ceiling, which is lifted with <c>LimitRTPRIO=</c> in the systemd unit or an <c>rtprio</c>
/// line under <c>/etc/security/limits.d/</c>. Outside Linux both return false.</remarks>
public static partial class RealtimeScheduling
{
    /// <summary>Highest <c>SCHED_FIFO</c> priority Linux accepts.</summary>
    public const int MaxFifoPriority = 99;

    private const int SchedFifo = 1;
    private const int RlimitRtprio = 14;
    private const ulong RlimInfinity = ulong.MaxValue;
    private const int Eperm = 1;
    private const int Einval = 22;
    // A 1024-bit cpu_set_t, the size glibc's CPU_SETSIZE describes.
    private const int CpuSetBytes = 128;

    /// <summary>Highest CPU index <see cref="TryPinToCpu"/> can address.</summary>
    public const int MaxCpu = CpuSetBytes * 8 - 1;

    /// <summary>Switches the calling thread to <c>SCHED_FIFO</c> at <paramref name="priority"/> (1..99).</summary>
    /// <returns>True on success; otherwise false with <paramref name="reason"/> naming what to configure.</returns>
    public static bool TryEnterFifo(int priority, out string reason)
    {
        if (priority < 1 || priority > MaxFifoPriority)
        {
            reason = $"SCHED_FIFO priority must be between 1 and {MaxFifoPriority}; {priority} was requested.";
            return false;
        }
        if (!OperatingSystem.IsLinux())
        {
            reason = "SCHED_FIFO is only available on Linux; the thread stays on the default scheduler.";
            return false;
        }
        return TryEnterFifoLinux(priority, out reason);
    }

    /// <summary>Restricts the calling thread to CPU <paramref name="cpu"/>.</summary>
    /// <returns>True on success; otherwise false with <paramref name="reason"/>.</returns>
    public static bool TryPinToCpu(int cpu, out string reason)
    {
        if (cpu < 0 || cpu > MaxCpu)
        {
            reason = $"CPU index must be between 0 and {MaxCpu}; {cpu} was requested.";
            return false;
        }
        if (!OperatingSystem.IsLinux())
        {
            reason = "CPU pinning is only available on Linux; the thread keeps the process affinity.";
            return false;
        }
        return TryPinToCpuLinux(cpu, out reason);
    }

    [SupportedOSPlatform("linux")]
    private static bool TryEnterFifoLinux(int priority, out string reason)
    {
        Rlimit limit = default;
        if (GetRlimit(RlimitRtprio, ref limit) != 0)
        {
            reason = $"getrlimit(RLIMIT_RTPRIO) failed with errno {Marshal.GetLastPInvokeError()}.";
            return false;
        }
        // Root (CAP_SYS_NICE) is not bound by the limit, so only an unprivileged caller is refused up front.
        bool limited = limit.Current != RlimInfinity && limit.Current < (ulong)priority;
        if (limited && GetEuid() != 0)
        {
            reason = RtprioAdvice(priority, limit.Current);
            return false;
        }
        SchedParam param = new() { Priority = priority };
        if (SchedSetScheduler(0, SchedFifo, ref param) != 0)
        {
            int errno = Marshal.GetLastPInvokeError();
            reason = errno == Eperm
                ? RtprioAdvice(priority, limit.Current)
                : $"sched_setscheduler(SCHED_FIFO, {priority}) failed with errno {errno}.";
            return false;
        }
        reason = "";
        return true;
    }

    [SupportedOSPlatform("linux")]
    private static bool TryPinToCpuLinux(int cpu, out string reason)
    {
        Span<byte> mask = stackalloc byte[CpuSetBytes];
        mask.Clear();
        mask[cpu >> 3] = (byte)(1 << (cpu & 7));
        if (SchedSetAffinity(0, (nuint)CpuSetBytes, ref MemoryMarshal.GetReference(mask)) != 0)
        {
            int errno = Marshal.GetLastPInvokeError();
            reason = errno == Einval
                ? $"CPU {cpu} is not online or lies outside this process's allowed set (AllowedCPUs / cpuset)."
                : $"sched_setaffinity(cpu {cpu}) failed with errno {errno}.";
            return false;
        }
        reason = "";
        return true;
    }

    private static string RtprioAdvice(int priority, ulong current) =>
        $"SCHED_FIFO {priority} refused: RLIMIT_RTPRIO is {current} (ulimit -r). Grant it with LimitRTPRIO={priority} "
        + $"in the systemd unit, or a '<user> - rtprio {priority}' line under /etc/security/limits.d/ followed by a "
        + "new login session.";

    [LibraryImport("libc", EntryPoint = "getrlimit", SetLastError = true)]
    [SupportedOSPlatform("linux")]
    private static partial int GetRlimit(int resource, ref Rlimit limit);

    [LibraryImport("libc", EntryPoint = "sched_setscheduler", SetLastError = true)]
    [SupportedOSPlatform("linux")]
    private static partial int SchedSetScheduler(int pid, int policy, ref SchedParam param);

    [LibraryImport("libc", EntryPoint = "sched_setaffinity", SetLastError = true)]
    [SupportedOSPlatform("linux")]
    private static partial int SchedSetAffinity(int pid, nuint cpuSetSize, ref byte mask);

    [LibraryImport("libc", EntryPoint = "geteuid")]
    [SupportedOSPlatform("linux")]
    private static partial uint GetEuid();

    /// <summary><c>struct rlimit</c> on LP64 Linux: two <c>unsigned long</c> fields.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct Rlimit
    {
        public ulong Current;
        public ulong Max;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SchedParam
    {
        public int Priority;
    }
}
