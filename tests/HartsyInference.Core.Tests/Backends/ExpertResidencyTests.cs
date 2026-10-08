using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Residency queries and no-upload acquisition on <see cref="ExpertCacheBase"/>, against the recording fake.</summary>
public sealed class ExpertResidencyTests
{
    private const int MatrixElements = 256;
    private const long ExpertBytes = 3 * MatrixElements * 4;

    private static ExpertWeights Weights(ExpertKey key) => new(
        key,
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)),
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)),
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)));

    private static FakeExpertCache CacheWithExpertsOneAndTwoResident(out FakeExpertCache cache)
    {
        cache = new FakeExpertCache(8 * ExpertBytes);
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 8, bank: 0));
        using ExpertLease warm = cache.Acquire([new ExpertKey(0, 1), new ExpertKey(0, 2)]);
        warm.Dispose();
        return cache;
    }

    [Fact]
    public void LookupResident_ReportsPresenceWithoutChangingCacheState()
    {
        using FakeExpertCache cache = CacheWithExpertsOneAndTwoResident(out _);
        ExpertCacheStats before = cache.Stats;
        bool[] mask = new bool[3];

        int resident = cache.LookupResident([new ExpertKey(0, 1), new ExpertKey(0, 5), new ExpertKey(0, 2)], mask);

        Assert.Equal(2, resident);
        Assert.Equal([true, false, true], mask);
        Assert.Equal(before.Hits, cache.Stats.Hits);
        Assert.Equal(before.Misses, cache.Stats.Misses);
        Assert.Throws<ArgumentException>(() => cache.LookupResident([new ExpertKey(0, 1)], new bool[0]));
    }

    [Fact]
    public void AcquireResident_PinsResidentExpertsAndReturnsMissesWithoutUploading()
    {
        using FakeExpertCache cache = CacheWithExpertsOneAndTwoResident(out _);
        int uploadsBefore = cache.Events.Count(e => e.StartsWith("upload"));
        List<ExpertKey> misses = [];

        using ExpertLease lease = cache.AcquireResident([new ExpertKey(0, 1), new ExpertKey(0, 5), new ExpertKey(0, 2)], misses);

        Assert.Equal([new ExpertKey(0, 5)], misses);
        Assert.Equal(2, lease.Weights.Count);
        Assert.Equal(uploadsBefore, cache.Events.Count(e => e.StartsWith("upload")));
        Assert.Equal(2, cache.Stats.PinnedExperts);
    }

    [Fact]
    public void AcquireResident_WithNothingResident_ReturnsAnEmptyLeaseAndAllMisses()
    {
        using FakeExpertCache cache = new(4 * ExpertBytes);
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 8, bank: 0));
        List<ExpertKey> misses = [];

        using ExpertLease lease = cache.AcquireResident([new ExpertKey(0, 3)], misses);

        Assert.Empty(lease.Weights);
        Assert.Equal([new ExpertKey(0, 3)], misses);
        Assert.DoesNotContain(cache.Events, e => e.StartsWith("upload"));
    }

    [Fact]
    public void AcquireResident_PinnedExpertsSurviveTrim()
    {
        using FakeExpertCache cache = CacheWithExpertsOneAndTwoResident(out _);
        List<ExpertKey> misses = [];
        using ExpertLease lease = cache.AcquireResident([new ExpertKey(0, 1)], misses);

        cache.Trim(0);

        Assert.Equal(1, cache.Stats.ResidentExperts);
        Assert.DoesNotContain(cache.Events, e => e == "evict " + new ExpertKey(0, 1));
    }

    [Fact]
    public void AcquireResident_OnADisposedCache_ThrowsAndLeavesMissesUntouched()
    {
        FakeExpertCache cache = new(4 * ExpertBytes);
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 8, bank: 0));
        cache.Dispose();
        List<ExpertKey> misses = [];

        Assert.ThrowsAny<Exception>(() => cache.AcquireResident([new ExpertKey(0, 4)], misses));
        Assert.Empty(misses);
    }

    [Fact]
    public void ResidencyPath_DoesNotAllocatePerCallWithAReusedLease()
    {
        using FakeExpertCache cache = CacheWithExpertsOneAndTwoResident(out _);
        ExpertKey[] keys = [new ExpertKey(0, 1), new ExpertKey(0, 2)];
        bool[] mask = new bool[keys.Length];
        List<ExpertKey> misses = new(keys.Length);
        ExpertLease lease = new();

        // Warm up so scratch capacity and the lease's slots have settled before measuring.
        for (int i = 0; i < 4; i++)
        {
            cache.LookupResident(keys, mask);
            misses.Clear();
            cache.AcquireResident(keys, misses, lease);
            lease.Dispose();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) cache.LookupResident(keys, mask);
        long lookupBytes = GC.GetAllocatedBytesForCurrentThread() - before;

        long acquireBytes = 0;
        for (int i = 0; i < 100; i++)
        {
            misses.Clear();
            before = GC.GetAllocatedBytesForCurrentThread();
            cache.AcquireResident(keys, misses, lease);
            acquireBytes += GC.GetAllocatedBytesForCurrentThread() - before;
            lease.Dispose();
        }

        Assert.Equal(0, lookupBytes);
        Assert.Equal(0, acquireBytes);
    }

    [Fact]
    public void AcquireResident_OntoAPinnedLease_Throws_AndRebindingAfterReleaseReplacesTheExperts()
    {
        using FakeExpertCache cache = CacheWithExpertsOneAndTwoResident(out _);
        List<ExpertKey> misses = [];
        ExpertLease lease = new();
        cache.AcquireResident([new ExpertKey(0, 1), new ExpertKey(0, 2)], misses, lease);

        Assert.Throws<InvalidOperationException>(() => cache.AcquireResident([new ExpertKey(0, 1)], misses, lease));
        Assert.Equal(2, lease.Count);

        lease.Dispose();
        cache.AcquireResident([new ExpertKey(0, 2)], misses, lease);

        Assert.Single(lease);
        Assert.Equal(new ExpertKey(0, 2), lease[0].Key);
        Assert.Throws<KeyNotFoundException>(() => lease.Get(new ExpertKey(0, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => lease[1]);
        lease.Dispose();
        Assert.Equal(0, cache.Stats.PinnedExperts);
    }

    [Fact]
    public void RepeatedReleaseOfTheSameExperts_ReclaimsCompletedFences_InsteadOfGrowingThem()
    {
        using FakeExpertCache cache = CacheWithExpertsOneAndTwoResident(out _);
        ExpertKey[] keys = [new ExpertKey(0, 1), new ExpertKey(0, 2)];
        List<ExpertKey> misses = [];
        ExpertLease lease = new();

        for (int i = 0; i < 1000; i++)
        {
            misses.Clear();
            cache.AcquireResident(keys, misses, lease);
            lease.Dispose();
        }

        // Each release records one device fence; completed ones must be reclaimed on the release path, not kept until eviction.
        Assert.True(cache.LiveFences <= 2, $"Expected at most two live fences, found {cache.LiveFences}.");
    }

    [Fact]
    public void ExternalImplementer_WrittenAgainstTheOldInterface_StillCompilesAndWorks()
    {
        using FakeExpertCache inner = CacheWithExpertsOneAndTwoResident(out _);
        IResidencyAwareExpertCache external = new LegacyResidencyCache(inner);
        bool[] mask = new bool[2];

        Assert.Equal(2, external.LookupResident([new ExpertKey(0, 1), new ExpertKey(0, 2)], mask));
        Assert.Equal([true, true], mask);

        List<ExpertKey> misses = [];
        using (ExpertLease lease = external.AcquireResident([new ExpertKey(0, 1), new ExpertKey(0, 5)], misses))
        {
            Assert.Equal([new ExpertKey(0, 5)], misses);
            Assert.Single(lease.Weights);
        }

        using (ExpertLease acquired = external.Acquire([new ExpertKey(0, 2)]))
        {
            Assert.Single(acquired.Weights);
        }

        // The caller-owned overload is not implemented by this implementer, so it reports that instead of failing silently.
        Assert.Throws<NotSupportedException>(() => external.AcquireResident([new ExpertKey(0, 1)], misses, new ExpertLease()));
    }

    /// <summary>An implementer compiled against the published interface: it has no caller-owned lease overload.</summary>
    private sealed class LegacyResidencyCache(ExpertCacheBase inner) : IResidencyAwareExpertCache
    {
        public long BudgetBytes => inner.BudgetBytes;
        public ExpertCacheStats Stats => inner.Stats;
        public void RegisterBank(ExpertBank bank) => inner.RegisterBank(bank);
        public ExpertLease Acquire(ReadOnlySpan<ExpertKey> keys) => inner.Acquire(keys);
        public int Prefetch(ReadOnlySpan<ExpertKey> keys) => inner.Prefetch(keys);
        public void Release(ExpertLease lease) => inner.Release(lease);
        public long Trim(long targetResidentBytes) => inner.Trim(targetResidentBytes);
        public int LookupResident(ReadOnlySpan<ExpertKey> keys, Span<bool> resident) => inner.LookupResident(keys, resident);
        public ExpertLease AcquireResident(ReadOnlySpan<ExpertKey> keys, List<ExpertKey> misses) => inner.AcquireResident(keys, misses);
        public void Dispose() => inner.Dispose();
    }
}
