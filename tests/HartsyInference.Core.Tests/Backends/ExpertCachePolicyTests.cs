using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Residency policy of <see cref="ExpertCacheBase"/> against a recording fake.</summary>
public sealed class ExpertCachePolicyTests
{
    private const int MatrixElements = 256;
    private const long ExpertBytes = 3 * MatrixElements * 4;

    private static ExpertBank MakeBank(int layer, int count = 16)
    {
        return new ExpertBank(layer, count, key => new ExpertWeights(
            key,
            new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)),
            new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)),
            new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32))));
    }

    private static FakeExpertCache MakeCache(int experts, params int[] layers)
    {
        FakeExpertCache cache = new(experts * ExpertBytes);
        foreach (int layer in layers) cache.RegisterBank(MakeBank(layer));
        return cache;
    }

    private static ExpertKey K(int layer, int expert) => new(layer, expert);

    [Fact]
    public void Acquire_DedupsRepeatedKeysAndCountsMissesThenHits()
    {
        using FakeExpertCache cache = MakeCache(4, 0);
        ExpertKey[] keys = [K(0, 1), K(0, 1), K(0, 2)];
        using ExpertLease first = cache.Acquire(keys);
        Assert.Equal(2, first.Weights.Count);
        Assert.Equal(2, cache.Events.Count(e => e.StartsWith("upload")));
        ExpertLease second = cache.Acquire([K(0, 1)]);
        Assert.Same(first.Get(K(0, 1)), second.Get(K(0, 1)));
        ExpertCacheStats stats = cache.Stats;
        Assert.Equal(2, stats.Misses);
        Assert.Equal(1, stats.Hits);
        Assert.Equal(2, stats.ResidentExperts);
        second.Dispose();
    }

    [Fact]
    public void Acquire_PinnedExpertsAreNeverEvicted()
    {
        using FakeExpertCache cache = MakeCache(2, 0, 1);
        using ExpertLease pinned = cache.Acquire([K(0, 0), K(0, 1)]);
        Assert.Throws<OutOfVramException>(() => cache.Acquire([K(1, 0)]));
        Assert.DoesNotContain(cache.Events, e => e.StartsWith("evict"));
        Assert.Equal(2, cache.Stats.PinnedExperts);
        Assert.Equal(2 * ExpertBytes, cache.Stats.PinnedBytes);
    }

    [Fact]
    public void Acquire_FailureLeavesNothingPinnedOrHalfResident()
    {
        using FakeExpertCache cache = MakeCache(4, 0);
        cache.FailUploadOf = K(0, 3);
        Assert.Throws<InvalidOperationException>(() => cache.Acquire([K(0, 1), K(0, 3)]));
        Assert.Equal(0, cache.Stats.ResidentExperts);
        Assert.Equal(0, cache.Stats.PinnedExperts);
        Assert.Equal(0, cache.Stats.ResidentBytes);
    }

    [Fact]
    public void Eviction_UsesProbationBeforeProtectedAndOldestFirst()
    {
        using FakeExpertCache cache = MakeCache(3, 0);
        cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(0, 1)]).Dispose();
        cache.Acquire([K(0, 2)]).Dispose();
        cache.Events.Clear();
        cache.Acquire([K(0, 3)]).Dispose();
        Assert.Contains("evict L0.E1", cache.Events);
        Assert.DoesNotContain("evict L0.E0", cache.Events);
    }

    [Fact]
    public void Eviction_PrefersColdLayerOverHotLayer()
    {
        using FakeExpertCache cache = MakeCache(3, 0, 1, 2);
        for (int i = 0; i < 5; i++) cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(1, 0)]).Dispose();
        cache.Acquire([K(1, 1)]).Dispose();
        cache.Events.Clear();
        cache.Acquire([K(2, 0)]).Dispose();
        Assert.Contains("evict L1.E0", cache.Events);
        Assert.DoesNotContain("evict L0.E0", cache.Events);
    }

    [Fact]
    public void Eviction_WaitsOnTheLastReadersFenceBeforeFreeing()
    {
        using FakeExpertCache cache = MakeCache(1, 0, 1);
        cache.FencesComplete = false;
        cache.Acquire([K(0, 0)]).Dispose();
        cache.Events.Clear();
        cache.Acquire([K(1, 0)]).Dispose();
        int wait = cache.Events.IndexOf("waitfence");
        int evict = cache.Events.IndexOf("evict L0.E0");
        Assert.True(wait >= 0 && wait < evict, string.Join(",", cache.Events));
    }

    [Fact]
    public void Prefetch_IsBestEffortAndDoesNotEvictPinnedOrCurrentLayer()
    {
        using FakeExpertCache cache = MakeCache(2, 0, 1);
        using ExpertLease pinned = cache.Acquire([K(0, 0)]);
        int started = cache.Prefetch([K(1, 0), K(1, 1), K(1, 2)]);
        Assert.Equal(1, started);
        Assert.Equal(1, cache.Stats.Prefetches);
        Assert.DoesNotContain(cache.Events, e => e.StartsWith("evict"));
    }

    [Fact]
    public void Prefetch_ThenAcquireAwaitsTheSharedUploadOnce()
    {
        using FakeExpertCache cache = MakeCache(4, 0);
        cache.Prefetch([K(0, 5)]);
        using ExpertLease lease = cache.Acquire([K(0, 5)]);
        Assert.Single(cache.Events, e => e == "upload L0.E5");
        Assert.Single(cache.Events, e => e == "await L0.E5");
        Assert.Equal(1, cache.Stats.InFlightHits);
        Assert.Equal(0, cache.Stats.Misses);
    }

    [Fact]
    public void Trim_EvictsUnpinnedDownToTarget()
    {
        using FakeExpertCache cache = MakeCache(4, 0);
        cache.Acquire([K(0, 0), K(0, 1), K(0, 2)]).Dispose();
        using ExpertLease keep = cache.Acquire([K(0, 3)]);
        long freed = cache.Trim(ExpertBytes);
        Assert.Equal(3 * ExpertBytes, freed);
        Assert.Equal(ExpertBytes, cache.Stats.ResidentBytes);
        Assert.Equal(1, cache.Stats.PinnedExperts);
    }

    [Fact]
    public void Dispose_ClosesLeasesEvictsEverythingAndDrains()
    {
        FakeExpertCache cache = MakeCache(4, 0);
        ExpertLease open = cache.Acquire([K(0, 0), K(0, 1)]);
        cache.Prefetch([K(0, 2)]);
        cache.Dispose();
        Assert.True(open.IsReleased);
        Assert.True(cache.Drained);
        Assert.Equal(3, cache.Events.Count(e => e.StartsWith("evict")));
        Assert.Equal(0, cache.LiveFences);
        Assert.Throws<ObjectDisposedException>(() => cache.Acquire([K(0, 0)]));
        cache.Dispose();
    }

    [Fact]
    public void ThousandAcquireCancelCycles_LeaveNoResidentBytesOrFences()
    {
        FakeExpertCache cache = MakeCache(3, 0, 1);
        for (int i = 0; i < 1000; i++)
        {
            ExpertLease lease = cache.Acquire([K(i % 2, i % 16), K(i % 2, (i * 7 + 1) % 16)]);
            if (i % 3 == 0) cache.Prefetch([K((i + 1) % 2, (i + 5) % 16)]);
            lease.Dispose();
        }
        cache.Dispose();
        Assert.Equal(0, cache.LiveFences);
        Assert.True(cache.Drained);
        Assert.Equal(cache.Events.Count(e => e.StartsWith("upload")), cache.Events.Count(e => e.StartsWith("evict")));
    }
}
