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
