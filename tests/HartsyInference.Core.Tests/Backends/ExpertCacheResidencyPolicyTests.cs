using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe.Residency;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Adaptive residency policy attached to <see cref="ExpertCacheBase"/>: victim choice, invalid-victim fallback, budget bound.</summary>
public sealed class ExpertCacheResidencyPolicyTests
{
    private const int MatrixElements = 256;
    private const long ExpertBytes = 3 * MatrixElements * 4;
    private const int Seed = 20261009;
    private const int ExpertsPerLayer = 16;

    private static ExpertBank MakeBank(int layer)
    {
        return new ExpertBank(layer, ExpertsPerLayer, key => new ExpertWeights(
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

    private static string Evicted(ExpertKey key) => "evict " + key;

    [Fact]
    public void LfuPolicy_ChoosesLowestCountVictim_WhereBuiltInOrderWouldEvictAnotherExpert()
    {
        using FakeExpertCache cache = MakeCache(2, 0);
        cache.AttachResidencyPolicy(new LfuPolicy());

        cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(0, 1)]).Dispose();
        cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(0, 1)]).Dispose();

        // Counts are 0 -> 3 and 1 -> 2, so LFU evicts 1. The built-in order would evict 0, the unprotected one.
        cache.Acquire([K(0, 2)]).Dispose();

        Assert.Contains(Evicted(K(0, 1)), cache.Events);
        Assert.DoesNotContain(Evicted(K(0, 0)), cache.Events);
        Assert.Equal(1, cache.Stats.Evictions);
    }

    [Fact]
    public void SegmentedLruPolicy_EvictsProbationaryLeastRecentVictim()
    {
        using FakeExpertCache cache = MakeCache(2, 0);
        cache.AttachResidencyPolicy(new SegmentedLruPolicy(protectedCapacity: 1));

        cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(0, 1)]).Dispose();
        cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(0, 1)]).Dispose();

        // Promoting 1 demotes 0 back to probation, so 0 is the only probationary resident and the policy evicts it.
        cache.Acquire([K(0, 2)]).Dispose();

        Assert.Contains(Evicted(K(0, 0)), cache.Events);
        Assert.DoesNotContain(Evicted(K(0, 1)), cache.Events);
    }

    [Fact]
    public void Policy_NamingLeasedExpert_FallsBackAndNeverEvictsLeasedEntry()
    {
        using FakeExpertCache cache = MakeCache(2, 0);
        cache.AttachResidencyPolicy(new FixedVictimPolicy(K(0, 0)));

        using ExpertLease held = cache.Acquire([K(0, 0)]);
        cache.Acquire([K(0, 1)]).Dispose();

        using ExpertLease incoming = cache.Acquire([K(0, 2)]);

        Assert.DoesNotContain(Evicted(K(0, 0)), cache.Events);
        Assert.Contains(Evicted(K(0, 1)), cache.Events);
        Assert.Equal(2, cache.Stats.PinnedExperts);
    }

    [Fact]
    public void Policy_NamingAbsentExpert_FallsBackToBuiltInOrder()
    {
        using FakeExpertCache cache = MakeCache(2, 0, 7);
        cache.AttachResidencyPolicy(new FixedVictimPolicy(K(7, 7)));

        cache.Acquire([K(0, 0)]).Dispose();
        cache.Acquire([K(0, 1)]).Dispose();
        cache.Acquire([K(0, 2)]).Dispose();

        Assert.Contains(Evicted(K(0, 0)), cache.Events);
        Assert.DoesNotContain(Evicted(K(7, 7)), cache.Events);
        Assert.Equal(1, cache.Stats.Evictions);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Policy_ChurnWithLeases_BudgetIsNeverExceeded(bool useLfu)
    {
        const int layers = 4;
        using FakeExpertCache cache = MakeCache(6, 0, 1, 2, 3);
        cache.AttachResidencyPolicy(useLfu ? new LfuPolicy() : new SegmentedLruPolicy(protectedCapacity: 3));
        Random random = new(Seed);
        for (int cycle = 0; cycle < 1500; cycle++)
        {
            ExpertKey[] keys = new ExpertKey[1 + random.Next(3)];
            for (int i = 0; i < keys.Length; i++) keys[i] = K(random.Next(layers), random.Next(ExpertsPerLayer));

            if (random.Next(8) == 0) cache.Prefetch(keys);
            else cache.Acquire(keys).Dispose();
            if (random.Next(50) == 0) cache.Trim(random.Next(6) * ExpertBytes);

            ExpertCacheStats stats = cache.Stats;
            Assert.True(stats.ResidentBytes <= stats.BudgetBytes, $"Resident {stats.ResidentBytes} exceeds budget {stats.BudgetBytes}.");
            Assert.Equal((long)stats.ResidentExperts * ExpertBytes, stats.ResidentBytes);
        }
        Assert.True(cache.Stats.Evictions > 0);
    }

    /// <summary>Always names the same expert, whether or not it is a valid candidate.</summary>
    private sealed class FixedVictimPolicy : IAdaptiveResidencyPolicy
    {
        private readonly ExpertKey named;

        public FixedVictimPolicy(ExpertKey named) => this.named = named;

        public string Name => "Fixed";

        public void NoteAccess(ExpertKey key)
        {
        }

        public void NoteInsert(ExpertKey key)
        {
        }

        public void NoteEvict(ExpertKey key)
        {
        }

        public ExpertKey ChooseVictim(ReadOnlySpan<ExpertKey> candidates) => named;

        public void Reset()
        {
        }
    }
}
