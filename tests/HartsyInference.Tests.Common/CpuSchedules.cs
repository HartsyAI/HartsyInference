using HartsyInference.Core.Configuration;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Tests.Common;

/// <summary>Runs CPU work under the schedules <see cref="CpuParallel"/> can give it, for tests that pin a result to be
/// the same whichever one ran it, or a nested fan-out to finish under each.</summary>
/// <remarks>The <c>numerics.cpuThreads</c> cap is process-wide, so a test class that changes it through here must not
/// run alongside one that asserts on how work is spread across threads; whatever cap was set before is restored.</remarks>
public static class CpuSchedules
{
    /// <summary>How long <see cref="RunBounded"/> waits before calling a run deadlocked.</summary>
    public static readonly TimeSpan DeadlockBound = TimeSpan.FromSeconds(60);

    /// <summary>Runs <paramref name="run"/> under the default fan-out over every core, a <c>numerics.cpuThreads</c> cap
    /// of 1, and inside <see cref="CpuParallel.EnterInline"/> (what a real-time audio thread runs under).</summary>
    public static (T Parallel, T Capped, T Inline) UnderEverySchedule<T>(Func<T> run)
    {
        ArgumentNullException.ThrowIfNull(run);
        T parallel = WithCpuThreads(0, run);
        T capped = WithCpuThreads(1, run);
        T inline = WithCpuThreads(0, () =>
        {
            using CpuParallel.InlineScope scope = CpuParallel.EnterInline();
            return run();
        });
        return (parallel, capped, inline);
    }

    /// <summary>Runs <paramref name="run"/> with the cap set to <paramref name="cap"/>, or cleared when it is 0, then
    /// restores whatever was set before.</summary>
    public static T WithCpuThreads<T>(int cap, Func<T> run)
    {
        ArgumentNullException.ThrowIfNull(run);
        bool hadOverride = KnobStore.HasOverride(EngineKnobs.CpuThreads);
        int previous = EngineKnobs.CpuThreads.Value;
        try
        {
            if (cap > 0) KnobStore.Set(EngineKnobs.CpuThreads, cap);
            else KnobStore.Clear(EngineKnobs.CpuThreads);
            return run();
        }
        finally
        {
            if (hadOverride) KnobStore.Set(EngineKnobs.CpuThreads, previous);
            else KnobStore.Clear(EngineKnobs.CpuThreads);
        }
    }

    /// <summary>As <see cref="WithCpuThreads{T}(int, Func{T})"/> for work that returns nothing.</summary>
    public static void WithCpuThreads(int cap, Action run)
    {
        ArgumentNullException.ThrowIfNull(run);
        WithCpuThreads(cap, () =>
        {
            run();
            return 0;
        });
    }

    /// <summary>Runs each action on its own background thread and throws, instead of hanging, if any is still running
    /// after <see cref="DeadlockBound"/>; rethrows the first failure wrapped.</summary>
    public static void RunBounded(params Action[] actions)
    {
        ArgumentNullException.ThrowIfNull(actions);
        Exception?[] failures = new Exception?[actions.Length];
        Thread[] threads = new Thread[actions.Length];
        for (int i = 0; i < actions.Length; i++)
        {
            int index = i;
            threads[i] = new Thread(() =>
            {
                try
                {
                    actions[index]();
                }
                catch (Exception ex)
                {
                    failures[index] = ex;
                }
            }) { IsBackground = true };
            threads[i].Start();
        }
        foreach (Thread thread in threads)
        {
            if (!thread.Join(DeadlockBound))
            {
                throw new TimeoutException($"a nested parallel run was still going after {DeadlockBound.TotalSeconds:F0} s; it deadlocked");
            }
        }
        foreach (Exception? failure in failures)
        {
            if (failure is not null) throw new InvalidOperationException("a nested parallel run failed", failure);
        }
    }
}
