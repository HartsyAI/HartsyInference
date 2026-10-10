using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tests.Backends;
using Xunit;

namespace HartsyInference.Core.Tests.Moe;

/// <summary>
/// Heterogeneous execution: whichever side runs an expert, it gets the same rows and writes its output to the same place. The
/// device runner here is the reference math, so these tests check placement and layout, not device numerics.
/// </summary>
public sealed class HeterogeneousExpertExecutorTests
{
    private const int Hidden = 8;
    private const int Intermediate = 6;
    private const int ExpertCount = 8;

    private static F32ExpertWeights MakeWeights(int seed)
    {
        Random rng = new(seed);
        float[] Fill(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() - 0.5) * 0.5f).ToArray();
        return new F32ExpertWeights(Hidden, Intermediate, Fill(Intermediate * Hidden), Fill(Intermediate * Hidden), Fill(Hidden * Intermediate));
    }

    private static F32ExpertWeights[] AllWeights { get; } = Enumerable.Range(0, ExpertCount).Select(static e => MakeWeights(100 + e)).ToArray();

    private static float[] Gathered(int pairs, int seed)
    {
        Random rng = new(seed);
        return Enumerable.Range(0, pairs * Hidden).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
    }

    /// <summary>A device runner that records each call and computes with the reference.</summary>
    private sealed class RecordingDevice(ExpertProgram program) : IExpertDeviceRunner
    {
        public List<(ExpertKey Key, int Rows)> Calls { get; } = [];

        public void Run(ExpertKey key, ReadOnlySpan<float> x, int rows, Span<float> y)
        {
            Calls.Add((key, rows));
            ExpertProgramReference.Apply(program, AllWeights[key.Expert], x, rows, y);
        }
    }

    private static ExpertAssignment[] Plan(params (int Expert, ExpertPlacement Placement, int Rows)[] experts) =>
        experts.Select(e => new ExpertAssignment(new ExpertKey(0, e.Expert), e.Placement, e.Rows)).ToArray();

    private static float[] Execute(ExpertProgram program, ExpertAssignment[] plan, float[] gathered, IExpertDeviceRunner? device)
    {
        float[] output = new float[gathered.Length];
        HeterogeneousExpertExecutor.Execute(program, plan, gathered, Hidden, output, static key => AllWeights[key.Expert], device);
        return output;
    }

    [Fact]
    public void AnySplitOfTheSamePlan_GivesTheSameOutput()
    {
        (int Expert, ExpertPlacement Placement, int Rows)[] experts =
        [
            (0, ExpertPlacement.Cpu, 2), (2, ExpertPlacement.Cpu, 1), (3, ExpertPlacement.Cpu, 3), (5, ExpertPlacement.Cpu, 2),
        ];
        float[] gathered = Gathered(8, seed: 1);
        ExpertProgram program = ExpertProgram.Swiglu;

        float[] allCpu = Execute(program, Plan(experts), gathered, device: null);

        ExpertAssignment[] alternating = Plan(experts
            .Select((e, i) => (e.Expert, i % 2 == 0 ? ExpertPlacement.Gpu : ExpertPlacement.Cpu, e.Rows)).ToArray());
        float[] split = Execute(program, alternating, gathered, new RecordingDevice(program));

        ExpertAssignment[] allGpu = Plan(experts.Select(e => (e.Expert, ExpertPlacement.Gpu, e.Rows)).ToArray());
        float[] everythingOnDevice = Execute(program, allGpu, gathered, new RecordingDevice(program));

        Assert.Equal(allCpu, split);
        Assert.Equal(allCpu, everythingOnDevice);
    }

    [Fact]
    public void DeviceRunner_ReceivesOnlyItsOwnAssignments_WithTheirRows()
    {
        RecordingDevice device = new(ExpertProgram.Swiglu);
        ExpertAssignment[] plan = Plan((1, ExpertPlacement.Gpu, 2), (4, ExpertPlacement.Cpu, 1), (6, ExpertPlacement.Gpu, 3));

        Execute(ExpertProgram.Swiglu, plan, Gathered(6, seed: 2), device);

        Assert.Equal([(new ExpertKey(0, 1), 2), (new ExpertKey(0, 6), 3)], device.Calls);
    }

    [Fact]
    public void GpuAssignment_WithoutADevice_IsRefused()
    {
        ExpertAssignment[] plan = Plan((0, ExpertPlacement.Gpu, 1));
        Assert.Throws<InvalidOperationException>(() => Execute(ExpertProgram.Swiglu, plan, Gathered(1, 4), device: null));
    }

    private static ExpertWeights Tensors(ExpertKey key) => new(
        key,
        new ExpertMatrix(new Tensor(new TensorShape(Hidden * Intermediate), DType.F32)),
        new ExpertMatrix(new Tensor(new TensorShape(Hidden * Intermediate), DType.F32)),
        new ExpertMatrix(new Tensor(new TensorShape(Hidden * Intermediate), DType.F32)));

    [Fact]
    public void PlannerOutput_DrivesTheExecutor_OnlyResidentExpertsReachTheDevice()
    {
        using FakeExpertCache cache = new(16 * Hidden * Intermediate * 3 * 4L);
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Tensors), 0, ExpertCount, 0));
        foreach (int expert in new[] { 1, 2 })
        {
            using ExpertLease warm = cache.Acquire([new ExpertKey(0, expert, 0)]);
        }

        int[] ids = [1, 1, 3, 2, 0];
        int[] counts = new int[ExpertCount];
        bool[] resident = new bool[ExpertCount];
        ExpertKey[] keys = new ExpertKey[ExpertCount];
        ExpertAssignment[] assignments = new ExpertAssignment[ExpertCount];
        List<ExpertKey> misses = new(ExpertCount);
        using ExpertLease lease = new();
        int planned = ExpertScheduler.Plan(cache, ids, layer: 0, bank: 0, ExpertCount, ResidentFirstPolicy.Instance, counts, resident, keys,
            assignments, misses, lease);
        ExpertAssignment[] plan = assignments[..planned];

        int pairs = plan.Sum(static a => a.Rows);
        float[] gathered = Gathered(pairs, seed: 7);
        RecordingDevice device = new(ExpertProgram.Swiglu);
        float[] output = new float[gathered.Length];
        HeterogeneousExpertExecutor.Execute(ExpertProgram.Swiglu, plan, gathered, Hidden, output, static key => AllWeights[key.Expert], device);

        Assert.Equal(plan.Where(a => a.Placement == ExpertPlacement.Gpu).Select(a => a.Key), device.Calls.Select(c => c.Key));
        Assert.All(device.Calls, call => Assert.Contains(call.Key.Expert, new[] { 1, 2 }));
        ExpertAssignment[] allCpu = plan.Select(a => a with { Placement = ExpertPlacement.Cpu }).ToArray();
        Assert.Equal(Execute(ExpertProgram.Swiglu, allCpu, gathered, device: null), output);
    }
}
