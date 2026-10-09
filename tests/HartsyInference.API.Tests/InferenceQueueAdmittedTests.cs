using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The shared queue's two entry points: the bounded one refuses a request that arrives at a full queue, while a round the scheduler has already admitted waits
/// its turn.</summary>
public sealed class InferenceQueueAdmittedTests
{
    [Fact]
    public async Task An_Admitted_Round_Waits_Where_A_New_Request_Is_Refused()
    {
        using InferenceQueue queue = new(maxConcurrency: 1, maxQueueDepth: 0);
        TaskCompletionSource hold = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task holder = queue.EnqueueAsync(async () => { await hold.Task; return 0; }, CancellationToken.None);

        await Assert.ThrowsAsync<QueueFullException>(() => queue.EnqueueAsync(() => Task.FromResult(0), CancellationToken.None));

        Task<int> round = queue.EnqueueAdmittedAsync(() => Task.FromResult(1), CancellationToken.None);
        Assert.False(round.IsCompleted); // queued behind the holder, not refused

        hold.SetResult();
        await holder;
        Assert.Equal(1, await round);
    }
}
