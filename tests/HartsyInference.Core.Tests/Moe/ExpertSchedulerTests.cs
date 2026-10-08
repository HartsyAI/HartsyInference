using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tests.Backends;
using Xunit;

namespace HartsyInference.Core.Tests.Moe;

/// <summary>Placement planning: residency decides where experts run, every routed pair is counted, and planning pins what it plans.</summary>
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

    private static ExpertAssignment[] Run(FakeExpertCache cache, int[] ids, IMissExecutionPolicy policy, out ExpertLease lease, ushort bank = 0)
    {
        int[] counts = new int[8];
        bool[] resident = new bool[8];
        ExpertKey[] keys = new ExpertKey[8];
        ExpertAssignment[] output = new ExpertAssignment[8];
        List<ExpertKey> misses = [];
        int written = ExpertScheduler.Plan(cache, ids, layer: 0, bank, expertCount: 8, policy, counts, resident, keys, output, misses, out lease);
        return output[..written];
    }

    private static ExpertAssignment[] Plan(FakeExpertCache cache, int[] ids, IMissExecutionPolicy policy, ushort bank = 0)
    {
        ExpertAssignment[] assignments = Run(cache, ids, policy, out ExpertLease lease, bank);
        lease.Dispose();
        return assignments;
    }

    [Fact]
    public void ResidentExpertsRunOnTheGpu_AndMissesRunOnTheCpu()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1), (0, 2));

        ExpertAssignment[] plan = Plan(cache, [1, 1, 3, 2, 0], ResidentFirstPolicy.Instance);

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

        ExpertAssignment[] plan = Plan(cache, ids, ResidentFirstPolicy.Instance);

        Assert.Equal(ids.Length, plan.Sum(static assignment => assignment.Rows));
        Assert.Equal(plan.Length, plan.Select(static assignment => assignment.Key).Distinct().Count());
    }

    [Fact]
    public void Planning_PinsTheResidentExpertsAndUploadsNothing()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1), (0, 2));
        int uploadsBefore = cache.Events.Count(static e => e.StartsWith("upload"));

        ExpertAssignment[] plan = Run(cache, [1, 2, 3], ResidentFirstPolicy.Instance, out ExpertLease lease);

        Assert.Equal(2, cache.Stats.PinnedExperts);
        Assert.Equal(uploadsBefore, cache.Events.Count(static e => e.StartsWith("upload")));
        Assert.Equal(3, plan.Length);
        lease.Dispose();
        Assert.Equal(0, cache.Stats.PinnedExperts);
    }

    [Fact]
    public void ExpertsPlannedForTheGpu_CannotBeEvictedWhileThePlanIsHeld()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1));
        Run(cache, [1], ResidentFirstPolicy.Instance, out ExpertLease lease);

        cache.Trim(0);

        Assert.Equal(1, cache.Stats.ResidentExperts);
        lease.Dispose();
        cache.Trim(0);
        Assert.Equal(0, cache.Stats.ResidentExperts);
    }

    [Fact]
    public void Policy_CannotPlaceAMissOnTheGpu()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1));
        ForcedPlacementPolicy allGpu = new(static _ => ExpertPlacement.Gpu);

        Assert.Throws<InvalidOperationException>(() => Run(cache, [1, 2], allGpu, out _));
        Assert.Equal(0, cache.Stats.PinnedExperts);
    }

    [Fact]
    public void ForcedPolicies_PlaceResidentExpertsAsRequested()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1), (0, 2));
        int[] ids = [1, 2];

        Assert.All(Plan(cache, ids, new ForcedPlacementPolicy(_ => ExpertPlacement.Cpu)), static a => Assert.Equal(ExpertPlacement.Cpu, a.Placement));
        Assert.All(Plan(cache, ids, new ForcedPlacementPolicy(_ => ExpertPlacement.Gpu)), static a => Assert.Equal(ExpertPlacement.Gpu, a.Placement));

        ForcedPlacementPolicy half = new(static key => key.Expert == 1 ? ExpertPlacement.Gpu : ExpertPlacement.Cpu);
        ExpertAssignment[] first = Plan(cache, ids, half);
        ExpertAssignment[] second = Plan(cache, ids, half);
        Assert.Equal(first, second);
        Assert.Equal(ExpertPlacement.Gpu, first.Single(static a => a.Key.Expert == 1).Placement);
        Assert.Equal(ExpertPlacement.Cpu, first.Single(static a => a.Key.Expert == 2).Placement);
    }

    [Fact]
    public void Bank_IsCarriedIntoTheKeysAndTheResidencyLookup()
    {
        using FakeExpertCache cache = CacheWithResident((3, 2));

        ExpertAssignment[] inBank3 = Plan(cache, [2], ResidentFirstPolicy.Instance, bank: 3);
        ExpertAssignment[] inBank0 = Plan(cache, [2], ResidentFirstPolicy.Instance, bank: 0);

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
        List<ExpertKey> misses = [];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ExpertScheduler.Plan(cache, [8], 0, 0, 8, ResidentFirstPolicy.Instance, counts, resident, keys, output, misses, out _));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ExpertScheduler.Plan(cache, [1], 0, 0, 8, ResidentFirstPolicy.Instance, new int[4], resident, keys, output, misses, out _));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ExpertScheduler.Plan(cache, [1, 2], 0, 0, 8, ResidentFirstPolicy.Instance, counts, resident, keys, new ExpertAssignment[1], misses, out _));
    }
}
