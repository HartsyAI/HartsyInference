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
}
