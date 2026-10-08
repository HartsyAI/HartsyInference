using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tests.Backends;
using Xunit;

namespace HartsyInference.Core.Tests.Moe;

/// <summary>Placement planning: residency decides where experts run, every routed pair is counted, and planning changes nothing.</summary>
public sealed class ExpertSchedulerTests
{
    private const int MatrixElements = 256;
    private const long ExpertBytes = 3 * MatrixElements * 4;

    private static ExpertWeights Weights(ExpertKey key) => new(
        key,
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)),
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)),
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)));

    private static FakeExpertCache CacheWithResident(params (ushort Bank, int Expert)[] resident)
    {
        FakeExpertCache cache = new(16 * ExpertBytes);
        foreach (ushort bank in resident.Select(static r => r.Bank).Distinct())
            cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 8, bank));
        foreach ((ushort bank, int expert) in resident)
        {
            using ExpertLease warm = cache.Acquire([new ExpertKey(0, expert, bank)]);
        }
        return cache;
    }

    private static List<ExpertAssignment> Plan(FakeExpertCache cache, int[] ids, IMissExecutionPolicy policy, ushort bank = 0)
    {
        int[] counts = new int[8];
        bool[] resident = new bool[8];
        ExpertKey[] keys = new ExpertKey[8];
        ExpertAssignment[] output = new ExpertAssignment[8];
        int written = ExpertScheduler.Plan(cache, ids, layer: 0, bank, expertCount: 8, policy, counts, resident, keys, output);
        return output.Take(written).ToList();
    }

    [Fact]
    public void ResidentExpertsRunOnTheGpu_AndMissesRunOnTheCpu()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1), (0, 2));

        List<ExpertAssignment> plan = Plan(cache, [1, 1, 3, 2, 0], ResidentFirstPolicy.Instance);

        Assert.Equal(
            [
                new ExpertAssignment(new ExpertKey(0, 0, 0), ExpertPlacement.Cpu, 1),
                new ExpertAssignment(new ExpertKey(0, 1, 0), ExpertPlacement.Gpu, 2),
                new ExpertAssignment(new ExpertKey(0, 2, 0), ExpertPlacement.Gpu, 1),
                new ExpertAssignment(new ExpertKey(0, 3, 0), ExpertPlacement.Cpu, 1),
            ],
            plan);
    }

    [Fact]
    public void EveryRoutedPair_IsCountedExactlyOnce()
    {
        using FakeExpertCache cache = CacheWithResident((0, 4));
        int[] ids = [4, 4, 4, 0, 7, 0, 2];

        List<ExpertAssignment> plan = Plan(cache, ids, ResidentFirstPolicy.Instance);

        Assert.Equal(ids.Length, plan.Sum(static assignment => assignment.Rows));
        Assert.Equal(plan.Count, plan.Select(static assignment => assignment.Key).Distinct().Count());
    }

    [Fact]
    public void Planning_ReadsResidencyAndChangesNothing()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1));
        ExpertCacheStats before = cache.Stats;
        int eventsBefore = cache.Events.Count;

        Plan(cache, [1, 2, 3], ResidentFirstPolicy.Instance);

        Assert.Equal(eventsBefore, cache.Events.Count);
        Assert.Equal(before.Hits, cache.Stats.Hits);
        Assert.Equal(before.Misses, cache.Stats.Misses);
        Assert.Equal(before.ResidentExperts, cache.Stats.ResidentExperts);
    }

    [Fact]
    public void ForcedPolicies_PlaceEveryExpertRegardlessOfResidency()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1), (0, 2));
        int[] ids = [0, 1, 2, 3];

        Assert.All(Plan(cache, ids, new ForcedPlacementPolicy(_ => ExpertPlacement.Cpu)), static a => Assert.Equal(ExpertPlacement.Cpu, a.Placement));
        Assert.All(Plan(cache, ids, new ForcedPlacementPolicy(_ => ExpertPlacement.Gpu)), static a => Assert.Equal(ExpertPlacement.Gpu, a.Placement));

        // A deterministic half by expert parity: the same split on every call.
        ForcedPlacementPolicy half = new(static key => key.Expert % 2 == 0 ? ExpertPlacement.Gpu : ExpertPlacement.Cpu);
        List<ExpertAssignment> first = Plan(cache, ids, half);
        List<ExpertAssignment> second = Plan(cache, ids, half);
        Assert.Equal(first, second);
        Assert.Equal(ExpertPlacement.Gpu, first.Single(static a => a.Key.Expert == 0).Placement);
        Assert.Equal(ExpertPlacement.Cpu, first.Single(static a => a.Key.Expert == 1).Placement);
    }

    [Fact]
    public void Bank_IsCarriedIntoTheKeysAndTheResidencyLookup()
    {
        using FakeExpertCache cache = CacheWithResident((3, 2));

        List<ExpertAssignment> inBank3 = Plan(cache, [2], ResidentFirstPolicy.Instance, bank: 3);
        List<ExpertAssignment> inBank0 = Plan(cache, [2], ResidentFirstPolicy.Instance, bank: 0);

        Assert.Equal(new ExpertKey(0, 2, 3), inBank3.Single().Key);
        Assert.Equal(ExpertPlacement.Gpu, inBank3.Single().Placement);
        Assert.Equal(ExpertPlacement.Cpu, inBank0.Single().Placement);
    }

    [Fact]
    public void BadInput_IsRejected()
    {
        using FakeExpertCache cache = CacheWithResident();
        int[] counts = new int[8];
        bool[] resident = new bool[8];
        ExpertKey[] keys = new ExpertKey[8];
        ExpertAssignment[] output = new ExpertAssignment[8];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ExpertScheduler.Plan(cache, [8], 0, 0, 8, ResidentFirstPolicy.Instance, counts, resident, keys, output));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ExpertScheduler.Plan(cache, [1], 0, 0, 8, ResidentFirstPolicy.Instance, new int[4], resident, keys, output));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ExpertScheduler.Plan(cache, [1, 2], 0, 0, 8, ResidentFirstPolicy.Instance, counts, resident, keys, new ExpertAssignment[1]));
    }
}
