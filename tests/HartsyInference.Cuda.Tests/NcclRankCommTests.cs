using System.Net;
using System.Net.Sockets;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Distributed;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Multi-node bootstrap path exercised in one process: two rank threads, each on its own GPU, join through the TCP
/// rendezvous and <c>ncclCommInitRank</c>, then run AllReduce and AllGather. Real multi-process/multi-node runs need rented
/// hardware; this gate covers the bootstrap protocol and the per-rank communicator on a 2-GPU Linux box with libnccl.</summary>
// TODO(untested): this test has never run. It needs a 2-GPU Linux host with libnccl; on Windows and CUDA-less hosts it only skips.
// TODO(unsure): both ranks share one process, so this does not exercise separate processes or nodes.
// TODO(unsure): FreePort() is racy; another process can take the port before rank 0 binds it.
[Collection("CudaSerial")]
public sealed class NcclRankCommTests
{
    private readonly ITestOutputHelper _output;
    public NcclRankCommTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(ptxDir)) return ptxDir;
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }

    private static int FreePort()
    {
        TcpListener probe = new(IPAddress.Loopback, 0);
        probe.Start();
        int port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();
        return port;
    }

    private static unsafe Tensor Filled(long count, Func<long, float> value)
    {
        Tensor t = new(new TensorShape(count), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < count; i++) p[i] = value(i);
        return t;
    }

    private static DistributedRendezvousOptions Opts(int rank, int port) => new()
    {
        WorldSize = 2,
        Rank = rank,
        MasterAddress = "127.0.0.1",
        MasterPort = port,
        JobToken = "nccl-rank-test",
        Timeout = TimeSpan.FromSeconds(60),
    };

    /// <summary>Runs <paramref name="work"/> on a dedicated thread: NCCL's init and each collective block until every rank has called in.</summary>
    private static Task<T> OnRankThread<T>(Func<T> work) =>
        Task.Factory.StartNew(work, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public async Task TwoRankThreads_JoinThroughRendezvous_AllReduceAndAllGatherExact()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!NcclRankComm.IsAvailable) { _output.WriteLine("SKIPPED: libnccl not resolvable on this host."); return; }

        int port = FreePort();
        const long n = 1 << 16;
        Task<float[]> rank0 = OnRankThread(() => RunRank(0, port, n));
        Task<float[]> rank1 = OnRankThread(() => RunRank(1, port, n));
        float[][] results = await Task.WhenAll(rank0, rank1);

        for (long i = 0; i < n; i++)
        {
            float a = (i % 977) * 0.03125f - 7.5f;
            float b = (i % 1409) * -0.015625f + 3.25f;
            float expectedSum = a + b;
            Assert.Equal(expectedSum, results[0][i]);
            Assert.Equal(expectedSum, results[1][i]);
        }
        _output.WriteLine($"NCCL 2-rank bootstrap: AllReduce over {n} F32 bit-exact on both ranks; AllGather verified in-thread.");
    }

    /// <summary>Joins as <paramref name="rank"/>, reduces, gathers; returns the reduced buffer for the caller's check.</summary>
    private static float[] RunRank(int rank, int port, long n)
    {
        using CudaBackend backend = new CudaBackend(rank, PtxDir());
        using NcclRankComm comm = NcclRankComm.JoinAsync(backend, Opts(rank, port), CancellationToken.None).GetAwaiter().GetResult();
        Assert.Equal("nccl", comm.Transport);
        Assert.Equal(rank, comm.Rank);
        Assert.Equal(2, comm.WorldSize);

        using Tensor buffer = rank == 0
            ? Filled(n, i => (i % 977) * 0.03125f - 7.5f)
            : Filled(n, i => (i % 1409) * -0.015625f + 3.25f);
        comm.AllReduceSum(buffer);

        using Tensor gatherSend = Filled(4096, i => i + 1_000f * (rank + 1));
        using Tensor gatherRecv = Filled(4096 * 2, _ => -1f);
        comm.AllGather(gatherSend, gatherRecv);
        unsafe
        {
            float* r = (float*)gatherRecv.DataPointer;
            for (long i = 0; i < 4096; i++)
            {
                Assert.True(r[i] == i + 1_000f, $"rank {rank} block0[{i}] = {r[i]}");
                Assert.True(r[4096 + i] == i + 2_000f, $"rank {rank} block1[{i}] = {r[4096 + i]}");
            }
        }

        float[] reduced = new float[n];
        unsafe
        {
            float* p = (float*)buffer.DataPointer;
            for (long i = 0; i < n; i++) reduced[i] = p[i];
        }
        return reduced;
    }
}
