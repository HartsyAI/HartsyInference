using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// <see cref="CudaExpertDeviceRunner"/> on a real backend: parity with the F32 reference per program variant, the no-upload guard,
/// and the planner-to-executor slice with mixed placement. Builds a <see cref="CudaBackend"/>, so it needs the backend's kernel
/// bundle to load on this device.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaExpertDeviceRunnerTests
{
    private const int Hidden = 64;
    private const int Intermediate = 32;
    private const int ExpertCount = 8;
    private const int Rows = 3;
    private const long ExpertBytes = 3L * Intermediate * Hidden * sizeof(float);

    /// <summary>Absolute tolerance on outputs of magnitude about 1; the same bound <see cref="CudaExpertKernelTests"/> measures.</summary>
    private const double ToleranceAbs = 1e-5;

    private readonly ITestOutputHelper _output;

    public CudaExpertDeviceRunnerTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <summary>The host F32 weights of one expert, generated from its key so every test and the reference agree.</summary>
    private static F32ExpertWeights HostWeights(ExpertKey key)
    {
        Random rng = new(key.Layer * 7919 + key.Expert * 104729 + key.Bank * 13 + 1);
        return new F32ExpertWeights(Hidden, Intermediate, Fill(rng, Intermediate * Hidden), Fill(rng, Intermediate * Hidden),
            Fill(rng, Hidden * Intermediate));
    }

    private static float[] Fill(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 0.5 - 0.25);
        return values;
    }

    private static Tensor Matrix(float[] values, int rows, int cols)
    {
        Tensor tensor = new(new TensorShape(rows, cols), DType.F32);
        float* data = (float*)tensor.DataPointer;
        for (int i = 0; i < values.Length; i++) data[i] = values[i];
        return tensor;
    }

    /// <summary>The bank's resolver: the device-side expert built from <see cref="HostWeights"/>.</summary>
    private static ExpertWeights DeviceWeights(ExpertKey key)
    {
        F32ExpertWeights host = HostWeights(key);
        return new ExpertWeights(key,
            new ExpertMatrix(Matrix(host.Gate, Intermediate, Hidden)),
            new ExpertMatrix(Matrix(host.Down, Hidden, Intermediate)),
            new ExpertMatrix(Matrix(host.Up, Intermediate, Hidden)));
    }

    private static float[] Inputs(int seed, int count)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    private static double MaxAbs(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Run_MatchesReference_ForEveryProgramVariant()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        using CudaExpertKernels kernels = new(PtxDir());
        ExpertProgram[] programs =
        [
            ExpertProgram.Swiglu,
            ExpertProgram.GeGlu,
            ExpertProgram.SwigluClamped(0.5f),
            new ExpertProgram(ExpertActivation.Relu, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity),
            new ExpertProgram(ExpertActivation.ReluSquared, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity),
        ];
        ExpertKey[] keys = [new(0, 1), new(0, 4), new(0, 6)];
        foreach (ExpertProgram program in programs)
        {
            using CudaExpertCache cache = new(backend, 4 * ExpertBytes, [new ExpertBank(0, ExpertCount, DeviceWeights)]);
            using ExpertLease lease = cache.Acquire(keys);
            CudaExpertDeviceRunner runner = new(backend, kernels, lease, program, Hidden, Intermediate);
            double worst = 0;
            foreach (ExpertKey key in keys)
            {
                float[] x = Inputs(key.Expert + 17, Rows * Hidden);
                float[] y = new float[Rows * Hidden];
                float[] reference = new float[Rows * Hidden];
                runner.Run(key, x, Rows, y);
                ExpertProgramReference.Apply(program, HostWeights(key), x, Rows, reference);
                worst = Math.Max(worst, MaxAbs(y, reference));
            }
            _output.WriteLine($"{program.Activation} clamped={program.IsClamped}: maxAbs={worst:E3}");
            Assert.True(worst <= ToleranceAbs,
                $"{program.Activation} clamped={program.IsClamped}: max abs error {worst:E3} exceeds {ToleranceAbs:E1}.");
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Run_ThrowsForAnExpertTheLeaseDoesNotHold_AndUploadsNothing()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        using CudaExpertKernels kernels = new(PtxDir());
        using CudaExpertCache cache = new(backend, 4 * ExpertBytes, [new ExpertBank(0, ExpertCount, DeviceWeights)]);
        using ExpertLease lease = cache.Acquire([new ExpertKey(0, 2)]);
        CudaExpertDeviceRunner runner = new(backend, kernels, lease, ExpertProgram.Swiglu, Hidden, Intermediate);

        ExpertCacheStats before = cache.Stats;
        (long _, long _, long missesBefore) = GpuTransferHelper.GetStats();
        float[] x = Inputs(3, Rows * Hidden);
        float[] y = new float[Rows * Hidden];
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() => runner.Run(new ExpertKey(0, 5), x, Rows, y));
        _output.WriteLine(error.Message);

        ExpertCacheStats after = cache.Stats;
        (long _, long _, long missesAfter) = GpuTransferHelper.GetStats();
        Assert.Equal(before.BytesUploaded, after.BytesUploaded);
        Assert.Equal(before.Misses, after.Misses);
        Assert.Equal(missesBefore, missesAfter);
        Assert.Equal(before.ResidentExperts, after.ResidentExperts);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Plan_MixedPlacement_ExecutesToTheSameOutputAsAllCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        using CudaExpertKernels kernels = new(PtxDir());
        using CudaExpertCache cache = new(backend, 4 * ExpertBytes, [new ExpertBank(0, ExpertCount, DeviceWeights)]);
        ExpertProgram program = ExpertProgram.SwigluClamped(0.5f);

        // Experts 0 and 1 are resident before planning; 2, 3 and 5 are misses the planner must run on the CPU.
        using (ExpertLease warm = cache.Acquire([new ExpertKey(0, 0), new ExpertKey(0, 1)]))
        {
            backend.Sync();
        }

        int[] ids = [0, 1, 2, 3, 2, 5, 0];
        int[] countScratch = new int[ExpertCount];
        bool[] residentScratch = new bool[ExpertCount];
        ExpertKey[] keyScratch = new ExpertKey[ExpertCount];
        ExpertAssignment[] plan = new ExpertAssignment[ExpertCount];
        List<ExpertKey> misses = new(ExpertCount);
        using ExpertLease lease = new();
        int count = ExpertScheduler.Plan(cache, ids, 0, 0, ExpertCount, ResidentFirstPolicy.Instance,
            countScratch, residentScratch, keyScratch, plan, misses, lease);
        ExpertAssignment[] assignments = plan[..count];
        Assert.Contains(assignments, a => a.Placement == ExpertPlacement.Gpu);
        Assert.Contains(assignments, a => a.Placement == ExpertPlacement.Cpu);

        int total = assignments.Sum(a => a.Rows);
        float[] gathered = Inputs(41, total * Hidden);
        float[] mixed = new float[total * Hidden];
        float[] cpu = new float[total * Hidden];
        ExpertAssignment[] allCpu = assignments.Select(a => a with { Placement = ExpertPlacement.Cpu }).ToArray();
        CudaExpertDeviceRunner runner = new(backend, kernels, lease, program, Hidden, Intermediate);
        HeterogeneousExpertExecutor.Execute(program, assignments, gathered, Hidden, mixed, HostWeights, runner);
        HeterogeneousExpertExecutor.Execute(program, allCpu, gathered, Hidden, cpu, HostWeights, null);

        double error = MaxAbs(mixed, cpu);
        _output.WriteLine($"mixed placement: {assignments.Count(a => a.Placement == ExpertPlacement.Gpu)} GPU, " +
            $"{assignments.Count(a => a.Placement == ExpertPlacement.Cpu)} CPU, maxAbs={error:E3}");
        Assert.True(error <= ToleranceAbs, $"Mixed placement differs from all-CPU by {error:E3}.");
    }
}
