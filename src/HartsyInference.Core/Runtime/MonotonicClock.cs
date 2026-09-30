using System.Diagnostics;
using System.Runtime.Versioning;

namespace HartsyInference.Core.Runtime;

/// <summary>Monotonic nanosecond clock with an absolute-deadline sleep, for threads that keep a fixed cadence.</summary>
/// <remarks>On Linux both calls go straight to <c>CLOCK_MONOTONIC</c>: <see cref="NowNs"/> is <c>clock_gettime</c> and
/// <see cref="SleepUntil"/> is <c>clock_nanosleep(TIMER_ABSTIME)</c>, so a tick thread sleeps to the deadline it
/// computed rather than for a relative duration measured after the fact, and lateness never accumulates into the
/// next period. Neither call allocates. <c>clock_nanosleep</c> reports failure as its return value (an errno, not
/// -1), and a signal interrupts it with <c>EINTR</c>, which is simply retried against the same absolute deadline.
/// <para>On every other platform the clock is <see cref="Stopwatch"/> and the sleep is <see cref="Thread.Sleep"/>
/// to within a millisecond of the deadline followed by a yield loop. That fallback is coarse and is the only place
/// this library sleeps a thread; the real-time callers are written for the Linux path.</para></remarks>
public static partial class MonotonicClock
{
    private const int ClockMonotonic = 1;
    private const int TimerAbstime = 1;
    private const int Eintr = 4;
    private const long NsPerSecond = 1_000_000_000L;
    private const long NsPerMillisecond = 1_000_000L;

    private static readonly double _stopwatchNsPerTick = NsPerSecond / (double)Stopwatch.Frequency;

    /// <summary>Nanoseconds on a clock that never jumps backwards; only differences between readings are meaningful.</summary>
    public static long NowNs()
    {
        if (OperatingSystem.IsLinux())
        {
            return NowNsLinux();
        }
        return (long)(Stopwatch.GetTimestamp() * _stopwatchNsPerTick);
    }

    /// <summary>Blocks the calling thread until <see cref="NowNs"/> reaches <paramref name="deadlineNs"/>; returns at once if it already has.</summary>
    public static void SleepUntil(long deadlineNs)
    {
        if (OperatingSystem.IsLinux())
        {
            SleepUntilLinux(deadlineNs);
            return;
        }
        SleepUntilFallback(deadlineNs);
    }

    [SupportedOSPlatform("linux")]
    private static long NowNsLinux()
    {
        Timespec now = default;
        if (ClockGetTime(ClockMonotonic, ref now) != 0)
        {
            throw new InvalidOperationException($"clock_gettime(CLOCK_MONOTONIC) failed with errno {Marshal.GetLastPInvokeError()}.");
        }
        return now.Seconds * NsPerSecond + now.Nanoseconds;
    }

    [SupportedOSPlatform("linux")]
    private static void SleepUntilLinux(long deadlineNs)
    {
        if (deadlineNs <= NowNsLinux())
        {
            return;
        }
        Timespec deadline = new()
        {
            Seconds = deadlineNs / NsPerSecond,
            Nanoseconds = deadlineNs % NsPerSecond,
        };
        int status;
        do
        {
            status = ClockNanosleep(ClockMonotonic, TimerAbstime, ref deadline, IntPtr.Zero);
        }
        while (status == Eintr);
        if (status != 0)
        {
            throw new InvalidOperationException($"clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME) failed with errno {status}.");
        }
    }

    /// <summary>Non-Linux fallback: sleeps in whole milliseconds while more than one remains, then yields to the deadline.</summary>
    private static void SleepUntilFallback(long deadlineNs)
    {
        long remaining = deadlineNs - NowNs();
        while (remaining > 0)
        {
            if (remaining > 2 * NsPerMillisecond)
            {
                Thread.Sleep((int)((remaining - NsPerMillisecond) / NsPerMillisecond));
            }
            else
            {
                Thread.Yield();
            }
            remaining = deadlineNs - NowNs();
        }
    }

    [LibraryImport("libc", EntryPoint = "clock_gettime", SetLastError = true)]
    [SupportedOSPlatform("linux")]
    private static partial int ClockGetTime(int clockId, ref Timespec time);

    [LibraryImport("libc", EntryPoint = "clock_nanosleep")]
    [SupportedOSPlatform("linux")]
    private static partial int ClockNanosleep(int clockId, int flags, ref Timespec request, IntPtr remain);

    /// <summary><c>struct timespec</c> on LP64 Linux (x64, arm64): two 64-bit fields.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct Timespec
    {
        public long Seconds;
        public long Nanoseconds;
    }
}
