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

    private static ExpertAssignment[] Run(FakeExpertCache cache, int[] ids, IMissExecutionPolicy policy, ExpertLease lease, ushort bank = 0)
    {
        int[] counts = new int[8];
        bool[] resident = new bool[8];
        ExpertKey[] keys = new ExpertKey[8];
        ExpertAssignment[] output = new ExpertAssignment[8];
        List<ExpertKey> misses = new(8);
        int written = ExpertScheduler.Plan(cache, ids, layer: 0, bank, expertCount: 8, policy, counts, resident, keys, output, misses, lease);
        return output[..written];
    }

    private static ExpertAssignment[] Plan(FakeExpertCache cache, int[] ids, IMissExecutionPolicy policy, ushort bank = 0)
    {
        using ExpertLease lease = new();
        return Run(cache, ids, policy, lease, bank);
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
    public void ExpertsPlannedForTheGpu_CannotBeEvictedWhileThePlanIsHeld()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1));
        using ExpertLease lease = new();
        Run(cache, [1], ResidentFirstPolicy.Instance, lease);

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

        Assert.Throws<InvalidOperationException>(() => Run(cache, [1, 2], allGpu, new ExpertLease()));
        Assert.Equal(0, cache.Stats.PinnedExperts);
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
    public void ReusedLease_PlansAllocateNothingAfterWarmup()
    {
        using FakeExpertCache cache = CacheWithResident((0, 1), (0, 2));
        int[] counts = new int[8];
        bool[] resident = new bool[8];
        ExpertKey[] keys = new ExpertKey[8];
        ExpertAssignment[] output = new ExpertAssignment[8];
        List<ExpertKey> misses = new(8);
        ExpertLease lease = new();
        int[] ids = [1, 2, 3];

        // Warm up so the lease, the cache's scratch and the pools have settled before measuring.
        for (int i = 0; i < 4; i++)
        {
            misses.Clear();
            ExpertScheduler.Plan(cache, ids, 0, 0, 8, ResidentFirstPolicy.Instance, counts, resident, keys, output, misses, lease);
            lease.Dispose();
        }

        long allocated = 0;
        for (int i = 0; i < 100; i++)
        {
            misses.Clear();
            long before = GC.GetAllocatedBytesForCurrentThread();
            ExpertScheduler.Plan(cache, ids, 0, 0, 8, ResidentFirstPolicy.Instance, counts, resident, keys, output, misses, lease);
            allocated += GC.GetAllocatedBytesForCurrentThread() - before;
            lease.Dispose();
        }

        Assert.Equal(0, allocated);
        Assert.Equal(0, cache.Stats.PinnedExperts);
    }
}
