using System.Collections.Concurrent;
using HartsyInference.Core.Numerics;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Numerics;

/// <summary>A thread inside <see cref="CpuParallel.InlineScope"/> must never hand work to the pool — it is the
/// audio thread with a deadline — while the same call outside the scope still fans out, or every kernel would
/// silently go single-threaded.</summary>
public sealed class CpuParallelInlineScopeTests
{
    private const int Items = 64;
    private const long WorkAboveThreshold = CpuParallel.MinWorkForParallel * 64;

    private static readonly Action<int, int[]> Store = static (i, results) => results[i] = i * 2;

    private readonly ITestOutputHelper _output;
    public CpuParallelInlineScopeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void InsideTheScope_EveryItemRunsOnTheCallingThread()
    {
        int caller = Environment.CurrentManagedThreadId;
        ConcurrentBag<int> threads = [];
        Assert.False(CpuParallel.IsInline);
        using (CpuParallel.EnterInline())
        {
            Assert.True(CpuParallel.IsInline);
            CpuParallel.For(Items, WorkAboveThreshold, i =>
            {
                Thread.SpinWait(50_000);
                threads.Add(Environment.CurrentManagedThreadId);
            });
        }
        Assert.False(CpuParallel.IsInline);
        Assert.Equal(Items, threads.Count);
        Assert.All(threads, t => Assert.Equal(caller, t));
    }

    [Fact]
    public void OutsideTheScope_TheSameCallFansOut()
    {
        if (CpuParallel.MaxThreads <= 1)
        {
            _output.WriteLine("SKIPPED: one worker only; nothing to fan out to.");
            return;
        }
        ConcurrentBag<int> threads = [];
        CpuParallel.For(Items, WorkAboveThreshold, i =>
        {
            Thread.SpinWait(200_000);
            threads.Add(Environment.CurrentManagedThreadId);
        });
        Assert.Equal(Items, threads.Count);
        Assert.True(threads.Distinct().Count() > 1, "the work never left the calling thread");
    }

    /// <summary>The stateful overload is for kernels on the audio thread: with its body in a static field, a call
    /// that runs inline must not allocate — not even the fan-out closure, which the compiler would otherwise build
    /// on entry.</summary>
    [Fact]
    public void StatefulFor_Inline_VisitsEveryIndex_AndAllocatesNothing()
    {
        int[] results = new int[Items];
        using CpuParallel.InlineScope scope = CpuParallel.EnterInline();
        CpuParallel.For(Items, WorkAboveThreshold, results, Store);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int call = 0; call < 100; call++) CpuParallel.For(Items, WorkAboveThreshold, results, Store);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        for (int i = 0; i < Items; i++) Assert.Equal(i * 2, results[i]);
    }

    [Fact]
    public void StatefulFor_Inline_ThrowsTheBodysExceptionAsItself()
    {
        int[] ran = new int[Items];
        using CpuParallel.InlineScope scope = CpuParallel.EnterInline();
        ArgumentException thrown = Assert.Throws<ArgumentException>(() =>
            CpuParallel.For(Items, WorkAboveThreshold, ran, static (i, r) =>
            {
                if (i == 7) throw new ArgumentException("item 7");
                r[i] = 1;
            }));
        Assert.Equal("item 7", thrown.Message);
        // Serial: the items before the throw ran, none after it did.
        Assert.Equal(7, ran.Sum());
    }

    [Fact]
    public void StatefulFor_FansOut_AndRethrowsAWorkerExceptionUnwrapped()
    {
        ConcurrentBag<int> seen = [];
        CpuParallel.For(Items, WorkAboveThreshold, seen, static (i, bag) =>
        {
            Thread.SpinWait(20_000);
            bag.Add(i);
        });
        Assert.Equal(Enumerable.Range(0, Items), seen.Order());
        Assert.Throws<ArgumentException>(() => CpuParallel.For(Items, WorkAboveThreshold, 0, static (i, _) =>
        {
            if (i == 7) throw new ArgumentException("item 7");
        }));
    }

    [Fact]
    public void Scopes_NestAndRestoreThePreviousState()
    {
        using (CpuParallel.EnterInline())
        {
            using (CpuParallel.EnterInline())
            {
                Assert.True(CpuParallel.IsInline);
            }
            Assert.True(CpuParallel.IsInline);
        }
        Assert.False(CpuParallel.IsInline);
    }

    [Fact]
    public void ADefaultScope_DisposesToNothing()
    {
        using (CpuParallel.EnterInline())
        {
            default(CpuParallel.InlineScope).Dispose();
            Assert.True(CpuParallel.IsInline, "a default instance must not clear an enclosing scope");
        }
        Assert.False(CpuParallel.IsInline);
    }

    [Fact]
    public void TheScope_IsPerThread()
    {
        using CpuParallel.InlineScope scope = CpuParallel.EnterInline();
        bool seenOnOtherThread = true;
        Thread other = new(() => seenOnOtherThread = CpuParallel.IsInline);
        other.Start();
        other.Join();
        Assert.False(seenOnOtherThread);
    }
}
