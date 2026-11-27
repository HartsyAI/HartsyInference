using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

public sealed class WorldExecutionLeaseTests
{
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(5);

    [Fact]
    public async Task SecondAcquire_WaitsUntilTheFirstIsReleased()
    {
        WorldExecutionLease lease = new WorldExecutionLease();
        IDisposable first = await lease.AcquireAsync(CancellationToken.None);
        Task<IDisposable> second = lease.AcquireAsync(CancellationToken.None);

        await Task.Delay(TimeSpan.FromMilliseconds(50));
        Assert.False(second.IsCompleted);
        Assert.Equal(1, lease.Queued);
        Assert.True(lease.IsHeld);

        first.Dispose();
        using IDisposable held = await second.WaitAsync(Patience);
        Assert.Equal(0, lease.Queued);
    }

    [Fact]
    public async Task CancelledWaiter_LeavesTheQueueCleanly()
    {
        WorldExecutionLease lease = new WorldExecutionLease();
        IDisposable first = await lease.AcquireAsync(CancellationToken.None);
        using CancellationTokenSource stop = new CancellationTokenSource();
        Task<IDisposable> waiter = lease.AcquireAsync(stop.Token);
        Assert.Equal(1, lease.Queued);

        stop.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiter);
        Assert.Equal(0, lease.Queued);

        // The cancelled waiter must not have taken or double-counted the slot: the next request gets it as soon as the holder leaves.
        first.Dispose();
        using IDisposable next = await lease.AcquireAsync(CancellationToken.None).WaitAsync(Patience);
        Assert.True(lease.IsHeld);
    }

    [Fact]
    public async Task Release_IsIdempotent_AndDoesNotGrantTheLeaseTwice()
    {
        WorldExecutionLease lease = new WorldExecutionLease();
        IDisposable first = await lease.AcquireAsync(CancellationToken.None);
        first.Dispose();
        first.Dispose();

        IDisposable second = await lease.AcquireAsync(CancellationToken.None);
        Task<IDisposable> third = lease.AcquireAsync(CancellationToken.None);
        await Task.Delay(TimeSpan.FromMilliseconds(50));

        // A double release would have let the third request in alongside the second.
        Assert.False(third.IsCompleted);
        second.Dispose();
        using IDisposable thirdHeld = await third.WaitAsync(Patience);
        Assert.True(lease.IsHeld);
    }
}
