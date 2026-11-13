using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Kimi;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The sharded decoder on CUDA, one device per rank, over NCCL where it loads and the host-staged collective otherwise. A synthetic BF16
/// checkpoint with MXFP4 experts runs on the devices and must match the single-rank CPU logits. GpuIntegration lane: skipped unless the box has
/// the devices, and run by hand because the GPU is shared.</summary>
public sealed class KimiK31ShardedCudaTests : IDisposable
{
    // BF16 weights on the GPU GEMM against the CPU's F32 reference: the same stored values, a different summation order.
    private const float Tol = 1e-3f;

    private static readonly int[] Tokens = [4, 4, 9, 0, 13, 7];

    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-kimi-cuda-" + Guid.NewGuid().ToString("N"));

    public KimiK31ShardedCudaTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void TwoRanks_CudaShardedCheckpoint_MatchesTheCpuLogits() => Run(2, KimiSyntheticCheckpoint.ShardedConfig);

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void EightRanks_CudaShardedCheckpoint_MatchesTheCpuLogits() => Run(8, KimiSyntheticCheckpoint.ShardedEightConfig);

    private void Run(int ranks, KimiK31Config config)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < ranks) { _output.WriteLine($"SKIPPED: needs {ranks} physical GPUs."); return; }
        KimiSyntheticCheckpoint.WriteSharded(_dir, config, shardFiles: 3, packedExperts: true, bf16Dense: true);

        float[] reference;
        using (CpuBackend cpuBackend = new())
        using (KimiK31Checkpoint cpu = KimiK31Checkpoint.Load(_dir, cpuBackend, maxPositions: 16))
        {
            reference = Forward(cpu, ranks: 1);
        }

        List<IBackend> backends = new(ranks);
        ICollectiveComm? comm = null;
        try
        {
            for (int d = 0; d < ranks; d++) backends.Add(new CudaBackend(d, PtxDir()));
            comm = CollectiveComm.Create(backends);
            using KimiK31Checkpoint sharded = KimiK31Checkpoint.Load(_dir, backends, comm, maxPositions: 16);
            float[] actual = Forward(sharded, ranks);
            Assert.Equal(reference.Length, actual.Length);
            for (int i = 0; i < reference.Length; i++) Assert.Equal(reference[i], actual[i], Tol);
        }
        finally
        {
            comm?.Dispose();
            foreach (IBackend backend in backends) backend.Dispose();
        }
    }

    private static float[] Forward(KimiK31Checkpoint checkpoint, int ranks)
    {
        using KimiK31State state = new(checkpoint.Config, capacity: 16, ranks);
        return checkpoint.Model.Forward(Tokens, state);
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("The PTX directory src/HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
