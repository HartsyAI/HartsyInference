using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

/// <summary>Counts the acquisitions and releases of the fake components a phase owns.</summary>
internal sealed class ResourceTally
{
    /// <summary>Components constructed.</summary>
    public int Acquired { get; private set; }

    /// <summary>Components disposed for the first time.</summary>
    public int Released { get; private set; }

    /// <summary>Records one acquisition.</summary>
    public void RecordAcquire() => Acquired++;

    /// <summary>Records one release.</summary>
    public void RecordRelease() => Released++;
}

/// <summary>A component that counts its own acquisition, and records each dispose so a double release shows up as a count of two.</summary>
internal sealed class CountingComponent : IDisposable
{
    private readonly ResourceTally _tally;
    private readonly string _name;
    private readonly List<string>? _log;
    private int _disposeCalls;

    /// <summary>Acquires one component against <paramref name="tally"/>; <paramref name="log"/> receives its name on dispose.</summary>
    public CountingComponent(ResourceTally tally, string name, List<string>? log = null)
    {
        ArgumentNullException.ThrowIfNull(tally);
        _tally = tally;
        _name = name;
        _log = log;
        _tally.RecordAcquire();
    }

    /// <summary>How many times <see cref="Dispose"/> was called.</summary>
    public int DisposeCalls => _disposeCalls;

    /// <inheritdoc/>
    public void Dispose()
    {
        _disposeCalls++;
        if (_disposeCalls == 1)
        {
            _tally.RecordRelease();
            _log?.Add(_name);
        }
    }
}

public sealed class WorldPhaseTests
{
    [Fact]
    public void Release_DisposesEachResourceOnceNewestFirst()
    {
        ResourceTally tally = new ResourceTally();
        List<string> log = [];
        WorldPhase phase = new WorldPhase("test");
        CountingComponent first = phase.Own(new CountingComponent(tally, "first", log));
        phase.OwnRelease("mapping", () => log.Add("mapping"));
        CountingComponent last = phase.Own(new CountingComponent(tally, "last", log));

        phase.Release();
        phase.Release();

        Assert.Equal(new[] { "last", "mapping", "first" }, log);
        Assert.Equal(1, first.DisposeCalls);
        Assert.Equal(1, last.DisposeCalls);
        Assert.Equal(2, tally.Released);
        Assert.Equal(3, phase.AcquiredCount);
        Assert.Equal(3, phase.ReleasedCount);
    }

    [Fact]
    public void Ledger_CountsEveryAcquireAndEndsAtZeroOnceReleased()
    {
        WorldPhaseLedger ledger = new WorldPhaseLedger();
        ResourceTally tally = new ResourceTally();
        WorldPhase phase = new WorldPhase("test", ledger);
        phase.Own(new CountingComponent(tally, "a"));
        phase.OwnRelease("b", () => { });
        Assert.Equal(2, ledger.Outstanding);

        phase.Release();

        Assert.Equal(0, ledger.Outstanding);
        Assert.Equal(2, ledger.Released);
    }

    [Fact]
    public void OwningAfterRelease_DisposesTheLateResourceAndRefuses()
    {
        ResourceTally tally = new ResourceTally();
        WorldPhase phase = new WorldPhase("test");
        phase.Release();

        Assert.Throws<ObjectDisposedException>(() => phase.Own(new CountingComponent(tally, "late")));

        Assert.Equal(1, tally.Acquired);
        Assert.Equal(tally.Acquired, tally.Released);
    }

    [Fact]
    public void AFailingRelease_IsLoggedAndTheRestStillRun()
    {
        ResourceTally tally = new ResourceTally();
        List<string> log = [];
        WorldPhase phase = new WorldPhase("test");
        phase.Own(new CountingComponent(tally, "kept", log));
        phase.OwnRelease("broken", () => throw new InvalidOperationException("dispose failed"));
        phase.Own(new CountingComponent(tally, "newest", log));

        phase.Release();

        Assert.Equal(new[] { "newest", "kept" }, log);
        Assert.Equal(2, tally.Released);
        Assert.Equal(phase.AcquiredCount, phase.ReleasedCount);
    }

    [Fact]
    public void SuccessPath_ReleasesEveryAcquiredComponentExactlyOnce()
    {
        ResourceTally tally = new ResourceTally();
        WorldPhaseLedger ledger = new WorldPhaseLedger();
        List<CountingComponent> components = [];
        using (WorldPhase phase = new WorldPhase("success", ledger))
        {
            for (int i = 0; i < 3; i++)
            {
                components.Add(phase.Own(new CountingComponent(tally, $"c{i}")));
            }
        }

        AssertReleasedExactlyOnce(tally, ledger, components);
    }

    [Fact]
    public void ThrownException_ReleasesEveryAcquiredComponentExactlyOnce()
    {
        ResourceTally tally = new ResourceTally();
        WorldPhaseLedger ledger = new WorldPhaseLedger();
        List<CountingComponent> components = [];
        Action failing = () =>
        {
            using WorldPhase phase = new WorldPhase("throws", ledger);
            for (int i = 0; i < 3; i++)
            {
                components.Add(phase.Own(new CountingComponent(tally, $"c{i}")));
            }
            throw new InvalidOperationException("the clip failed");
        };
        Assert.Throws<InvalidOperationException>(failing);

        AssertReleasedExactlyOnce(tally, ledger, components);
    }

    [Fact]
    public void Cancellation_ReleasesEveryAcquiredComponentExactlyOnce()
    {
        using CancellationTokenSource cancel = new CancellationTokenSource();
        cancel.Cancel();
        ResourceTally tally = new ResourceTally();
        WorldPhaseLedger ledger = new WorldPhaseLedger();
        List<CountingComponent> components = [];
        Action cancelled = () =>
        {
            using WorldPhase phase = new WorldPhase("cancelled", ledger);
            for (int i = 0; i < 3; i++)
            {
                components.Add(phase.Own(new CountingComponent(tally, $"c{i}")));
            }
            cancel.Token.ThrowIfCancellationRequested();
        };
        Assert.Throws<OperationCanceledException>(cancelled);

        AssertReleasedExactlyOnce(tally, ledger, components);
    }

    [Fact]
    public void LoadFailure_ReleasesEverythingTheLoadAcquired()
    {
        ResourceTally tally = new ResourceTally();
        ModelSpec spec = WorldTestFixtures.Spec(null, "fake-world");

        Assert.Throws<InvalidOperationException>(() => WorldService.LoadInPhase<object>("fake", spec, phase =>
        {
            phase.Own(new CountingComponent(tally, "dit"));
            phase.Own(new CountingComponent(tally, "vae"));
            throw new InvalidOperationException("the VAE checkpoint is truncated");
        }));

        Assert.Equal(2, tally.Acquired);
        Assert.Equal(2, tally.Released);
    }

    [Fact]
    public void LoadSuccess_KeepsItsPhaseOwnedForTheCacheEntry()
    {
        ResourceTally tally = new ResourceTally();
        ModelSpec spec = WorldTestFixtures.Spec(null, "fake-world");

        WorldService.LoadInPhase<object>("fake", spec, phase =>
        {
            phase.Own(new CountingComponent(tally, "dit"));
            return new object();
        });

        Assert.Equal(1, tally.Acquired);
        Assert.Equal(0, tally.Released);
    }

    private static void AssertReleasedExactlyOnce(ResourceTally tally, WorldPhaseLedger ledger, List<CountingComponent> components)
    {
        Assert.Equal(3, components.Count);
        Assert.Equal(components.Count, tally.Acquired);
        Assert.Equal(tally.Acquired, tally.Released);
        Assert.All(components, component => Assert.Equal(1, component.DisposeCalls));
        Assert.Equal(0, ledger.Outstanding);
    }
}
