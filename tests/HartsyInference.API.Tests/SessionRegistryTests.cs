namespace HartsyInference.API.Tests;

/// <summary>The generic session registry behind the world and realtime routes.</summary>
public sealed class SessionRegistryTests
{
    private sealed class Probe
    {
        public int Closed;
    }

    private static SessionRegistry<Probe> Registry(TimeSpan? idle = null) =>
        new(idle ?? TimeSpan.FromMinutes(10), p =>
        {
            Interlocked.Increment(ref p.Closed);
            return ValueTask.CompletedTask;
        });

    [Fact]
    public async Task RegisteredSessionsAreFoundUntilClosedOnce()
    {
        using SessionRegistry<Probe> registry = Registry();
        Probe probe = new();
        string id = registry.Register(probe);
        Assert.Same(probe, registry.Get(id));
        Assert.True(await registry.CloseAsync(id));
        Assert.False(await registry.CloseAsync(id));
        Assert.Null(registry.Get(id));
        Assert.Equal(1, probe.Closed);
    }

    [Fact]
    public void TheSweepClosesOnlyIdleSessions()
    {
        using SessionRegistry<Probe> registry = Registry(TimeSpan.FromMinutes(1));
        Probe stale = new(), fresh = new();
        string staleId = registry.Register(stale);
        string freshId = registry.Register(fresh);

        registry.Sweep(DateTime.UtcNow + TimeSpan.FromSeconds(30));
        Assert.Equal(2, registry.Count);

        registry.Sweep(DateTime.UtcNow + TimeSpan.FromMinutes(2));
        Assert.Equal(0, registry.Count);
        Assert.Equal((1, 1), (stale.Closed, fresh.Closed));
        Assert.Null(registry.Get(staleId));
        Assert.Null(registry.Get(freshId));
    }

    [Fact]
    public void DisposeClosesEverySessionStillOpen()
    {
        SessionRegistry<Probe> registry = Registry();
        Probe a = new(), b = new();
        registry.Register(a);
        registry.Register(b);
        registry.Dispose();
        Assert.Equal((1, 1), (a.Closed, b.Closed));
        Assert.Throws<ObjectDisposedException>(() => registry.Register(new Probe()));
    }

    [Fact]
    public void AFailingCloseDoesNotStopTheSweep()
    {
        Probe ok = new();
        using SessionRegistry<Probe> registry = new(TimeSpan.FromMinutes(1), p =>
        {
            if (p != ok) throw new InvalidOperationException("boom");
            p.Closed++;
            return ValueTask.CompletedTask;
        });
        registry.Register(new Probe());
        registry.Register(ok);
        registry.Sweep(DateTime.UtcNow + TimeSpan.FromMinutes(2));
        Assert.Equal(0, registry.Count);
        Assert.Equal(1, ok.Closed);
    }
}
