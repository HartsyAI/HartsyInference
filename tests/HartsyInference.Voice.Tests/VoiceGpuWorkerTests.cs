using HartsyInference.Cpu;
using HartsyInference.Voice.Gpu;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The GPU thread's job contract on a CPU device (where the gate and the memory calls are no-ops): jobs run on
/// the worker thread in order, a cancelled job is skipped, a failing job faults only its caller, a released model is
/// reopened once and the job retried, and shutdown runs its final work on the thread and refuses later jobs.</summary>
public sealed class VoiceGpuWorkerTests
{
    [Fact]
    public async Task JobsRunOnTheWorkerThreadInArrivalOrder()
    {
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device);
        List<int> order = [];
        List<Task<int>> jobs = [];
        for (int i = 0; i < 20; i++)
        {
            int index = i;
            jobs.Add(worker.RunAsync(VoiceGpuJobKind.Synthesize, () =>
            {
                order.Add(index);
                return Environment.CurrentManagedThreadId;
            }, CancellationToken.None));
        }
        int[] threads = await Task.WhenAll(jobs).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.All(threads, thread => Assert.Equal(worker.ManagedThreadId, thread));
        Assert.Equal(Enumerable.Range(0, 20), order);
    }

    [Fact]
    public async Task ACancelledJobIsSkippedAndItsCallerReleased()
    {
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device);
        using ManualResetEventSlim release = new(false);
        Task<bool> blocking = worker.RunAsync(VoiceGpuJobKind.Transcribe, () => release.Wait(TimeSpan.FromSeconds(10)), CancellationToken.None);
        bool ran = false;
        using CancellationTokenSource cancel = new();
        Task<bool> skipped = worker.RunAsync(VoiceGpuJobKind.Synthesize, () => ran = true, cancel.Token);
        cancel.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => skipped.WaitAsync(TimeSpan.FromSeconds(5)));
        release.Set();
        Assert.True(await blocking.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(42, await worker.RunAsync(VoiceGpuJobKind.Synthesize, () => 42, CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.False(ran);
    }

    [Fact]
    public async Task AFailingJobFaultsOnlyItsCaller()
    {
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device);
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            worker.RunAsync<int>(VoiceGpuJobKind.Transcribe, () => throw new InvalidOperationException("model failed"), CancellationToken.None));
        Assert.Equal(7, await worker.RunAsync(VoiceGpuJobKind.Transcribe, () => 7, CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(5)));
    }

    [Fact]
    public async Task AReleasedModelIsReopenedOnceAndTheJobRetried()
    {
        using CpuBackend device = new();
        int reopens = 0;
        int calls = 0;
        bool revoked = true;
        int reopenThread = 0;
        using VoiceGpuWorker worker = new(device, () =>
        {
            reopens++;
            reopenThread = Environment.CurrentManagedThreadId;
            revoked = false;
        });
        int result = await worker.RunAsync(VoiceGpuJobKind.Synthesize, () =>
        {
            calls++;
            return revoked ? throw new ObjectDisposedException("lease") : 5;
        }, CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(5, result);
        Assert.Equal(2, calls);
        Assert.Equal(1, reopens);
        Assert.Equal(1, worker.Reopens);
        Assert.Equal(worker.ManagedThreadId, reopenThread);
    }

    [Fact]
    public async Task AFailedReopenFailsTheJobButNotTheThread()
    {
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device, () => throw new InvalidOperationException("engine gone"));
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            worker.RunAsync<int>(VoiceGpuJobKind.Synthesize, () => throw new ObjectDisposedException("lease"), CancellationToken.None));
        Assert.Equal(3, await worker.RunAsync(VoiceGpuJobKind.Synthesize, () => 3, CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(5)));
    }

    [Fact]
    public async Task ATrimRequestRunsAsSoonAsTheQueueIsIdle()
    {
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device);
        Assert.Equal(1, await worker.RunAsync(VoiceGpuJobKind.Synthesize, () => 1, CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(5)));

        // No job follows the request: the trim must not wait for one.
        worker.RequestTrim();
        DateTime deadline = DateTime.UtcNow.AddSeconds(5);
        while (worker.Trims == 0 && DateTime.UtcNow < deadline)
        {
            await Task.Delay(5);
        }
        Assert.Equal(1, worker.Trims);
    }

    [Fact]
    public async Task ATrimRequestIsSkippedWhenTheNextTurnsWorkIsQueuedBehindIt()
    {
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device);
        using ManualResetEventSlim release = new(false);
        Task<bool> blocking = worker.RunAsync(VoiceGpuJobKind.Synthesize, () => release.Wait(TimeSpan.FromSeconds(10)), CancellationToken.None);
        worker.RequestTrim();
        Task<int> next = worker.RunAsync(VoiceGpuJobKind.Transcribe, () => 2, CancellationToken.None);
        release.Set();

        Assert.True(await blocking.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(2, await next.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(0, worker.Trims);
        await Assert.ThrowsAsync<ArgumentException>(() => worker.RunAsync(VoiceGpuJobKind.Trim, () => 0, CancellationToken.None));
    }

    [Fact]
    public async Task StopRunsTheFinalWorkOnTheThreadAndRefusesLaterJobs()
    {
        using CpuBackend device = new();
        using VoiceGpuWorker worker = new(device);
        int finalThread = 0;
        await worker.StopAsync(() => finalThread = Environment.CurrentManagedThreadId).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(worker.ManagedThreadId, finalThread);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => worker.RunAsync(VoiceGpuJobKind.Synthesize, () => 1, CancellationToken.None));
    }
}
