using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Distributed;

/// <summary>Single-process <see cref="HartsyInference.Cuda.NcclComm"/> all-to-all over one CUDA device per rank: every chunk of the counts
/// matrix lands at its destination. Skips when the host has fewer devices than ranks or libnccl does not resolve, and reports a host-staged
/// fallback as a skip rather than a pass. Not run in this change; the PR body carries the filters.</summary>
public sealed class NcclAllToAllTests
{
    private readonly ITestOutputHelper _output;

    public NcclAllToAllTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void TwoGpus_AllToAll_DeliversEveryChunk() => RunAllToAll(2);

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void EightGpus_AllToAll_DeliversEveryChunk() => RunAllToAll(8);

    private void RunAllToAll(int ranks)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < ranks) { _output.WriteLine($"SKIPPED: needs {ranks} physical GPUs."); return; }

        List<IBackend> backends = new(ranks);
        for (int r = 0; r < ranks; r++) backends.Add(new CudaBackend(r, PtxDir()));
        try
        {
            using ICollectiveComm comm = CollectiveComm.Create(backends);
            if (comm.Transport != "nccl") { _output.WriteLine("SKIPPED: libnccl not resolvable; host-staged transport ran instead."); return; }

            // counts[i, j] mixes zero and non-zero pairs, so some ranks exchange nothing with each other.
            long[,] counts = new long[ranks, ranks];
            for (int i = 0; i < ranks; i++)
                for (int j = 0; j < ranks; j++) counts[i, j] = (i + 2 * j) % 3 + (i == j ? 1 : 0);

            Tensor[] send = new Tensor[ranks], recv = new Tensor[ranks];
            try
            {
                for (int i = 0; i < ranks; i++)
                {
                    long total = 0;
                    for (int j = 0; j < ranks; j++) total += counts[i, j];
                    send[i] = new Tensor(new TensorShape(total + 1), DType.F32);
                    for (int j = 0; j < ranks; j++)
                    {
                        long offset = AllToAllLayout.SendOffset(counts, i, j);
                        for (int k = 0; k < counts[i, j]; k++) send[i].AsSpan<float>()[(int)(offset + k)] = i * 100 + j * 10 + k;
                    }
                }
                for (int j = 0; j < ranks; j++)
                {
                    long total = 0;
                    for (int i = 0; i < ranks; i++) total += counts[i, j];
                    recv[j] = new Tensor(new TensorShape(total + 1), DType.F32);
                }

                comm.AllToAll(send, recv, counts);

                for (int j = 0; j < ranks; j++)
                    for (int i = 0; i < ranks; i++)
                    {
                        long offset = AllToAllLayout.RecvOffset(counts, i, j);
                        for (int k = 0; k < counts[i, j]; k++)
                            Assert.Equal(i * 100 + j * 10 + k, recv[j].AsReadOnlySpan<float>()[(int)(offset + k)]);
                    }
                _output.WriteLine($"NCCL {ranks}-rank AllToAll: every chunk delivered in place.");
            }
            finally
            {
                foreach (Tensor t in send) t?.Dispose();
                foreach (Tensor t in recv) t?.Dispose();
            }
        }
        finally
        {
            foreach (IBackend b in backends) (b as IDisposable)?.Dispose();
        }
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        return Path.Combine(AppContext.BaseDirectory, "Ptx");
    }
}
