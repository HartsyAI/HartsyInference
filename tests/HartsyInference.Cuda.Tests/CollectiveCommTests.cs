using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Phase-1 collective transport gates: NCCL AllReduce/AllGather bit-exactness against a CPU reference
/// on the real 2-GPU pair, the host-staged fallback's correctness, and a measured cross-device AllGather
/// bandwidth row for the benchmark doc. With 2 ranks an F32 sum is a single add per element, so the NCCL
/// result must be BIT-exact vs the CPU <c>a+b</c> — no tolerance. Transport selection is asserted, not
/// assumed: on this box libnccl resolves from the standard probe dirs, so a silent fall-back to host-staged
/// is a FAILURE here (it would still be numerically correct — the assert exists to catch a broken resolver).</summary>
[Collection("CudaSerial")]
public sealed class CollectiveCommTests
{
    private readonly ITestOutputHelper _output;
    public CollectiveCommTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static unsafe Tensor Filled(long count, Func<long, float> value)
    {
        Tensor t = new(new TensorShape(count), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < count; i++) p[i] = value(i);
        return t;
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public unsafe void Nccl_TwoGpus_AllReduceBitExact_AllGatherBlocksCorrect_WithBandwidth()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }

        using CudaBackend backend0 = new CudaBackend(0, PtxDir());
        using CudaBackend backend1 = new CudaBackend(1, PtxDir());
        using ICollectiveComm comm = CollectiveComm.Create([backend0, backend1]);
        _output.WriteLine($"transport={comm.Transport} ranks={comm.RankCount}");
        Assert.Equal("nccl", comm.Transport);

        // ── AllReduce: bit-exact vs CPU a+b (2 ranks → one add, fixed order) ──
        const long n = 1 << 20;
        using (Tensor a = Filled(n, i => (i % 977) * 0.03125f - 7.5f))
        using (Tensor b = Filled(n, i => (i % 1409) * -0.015625f + 3.25f))
        using (Tensor expected = Filled(n, _ => 0f))
        {
            float* pa = (float*)a.DataPointer;
            float* pb = (float*)b.DataPointer;
            float* pe = (float*)expected.DataPointer;
            for (long i = 0; i < n; i++) pe[i] = pa[i] + pb[i];

            comm.AllReduceSum([a, b]);

            float* ra = (float*)a.DataPointer;
            float* rb = (float*)b.DataPointer;
            long mismA = 0, mismB = 0;
            for (long i = 0; i < n; i++)
            {
                if (ra[i] != pe[i]) mismA++;
                if (rb[i] != pe[i]) mismB++;
            }
            _output.WriteLine($"AllReduceSum over {n} F32: rank0 mismatches={mismA}, rank1 mismatches={mismB} (bit-exact required)");
            Assert.Equal(0, mismA);
            Assert.Equal(0, mismB);
        }

        // ── AllGather: rank-major block layout on both ranks ──
        const long g = 4096;
        using (Tensor send0 = Filled(g, i => i + 1_000f))
        using (Tensor send1 = Filled(g, i => i + 2_000f))
        using (Tensor recv0 = Filled(g * 2, _ => -1f))
        using (Tensor recv1 = Filled(g * 2, _ => -1f))
        {
            comm.AllGather([send0, send1], [recv0, recv1]);
            foreach ((Tensor recv, string label) in new[] { (recv0, "rank0"), (recv1, "rank1") })
            {
                float* r = (float*)recv.DataPointer;
                for (long i = 0; i < g; i++)
                {
                    Assert.True(r[i] == i + 1_000f, $"{label} block0[{i}] = {r[i]}, expected {i + 1_000f}");
                    Assert.True(r[g + i] == i + 2_000f, $"{label} block1[{i}] = {r[g + i]}, expected {i + 2_000f}");
                }
            }
            _output.WriteLine($"AllGather over {g}+{g} F32: rank-major layout verified on both ranks.");
        }

        // ── Bandwidth row: 256 MB per rank, warm run timed (upload excluded via a warmup that also uploads) ──
        const long big = 64L << 20;
        using (Tensor bigSend0 = Filled(big, i => i * 0.25f))
        using (Tensor bigSend1 = Filled(big, i => i * 0.5f))
        using (Tensor bigRecv0 = new(new TensorShape(big * 2), DType.F32))
        using (Tensor bigRecv1 = new(new TensorShape(big * 2), DType.F32))
        {
            comm.AllGather([bigSend0, bigSend1], [bigRecv0, bigRecv1]);   // warmup + upload
            Stopwatch sw = Stopwatch.StartNew();
            comm.AllGather([bigSend0, bigSend1], [bigRecv0, bigRecv1]);
            sw.Stop();
            // Cross-device traffic with 2 ranks: each rank's 256 MB block must reach the other card → 512 MB moved.
            double crossGb = 2 * (big * 4.0) / (1 << 30);
            double gbps = crossGb / sw.Elapsed.TotalSeconds;
            _output.WriteLine($"AllGather 256 MB/rank (warm): {sw.Elapsed.TotalMilliseconds:F0} ms, cross-device {crossGb:F2} GB → {gbps:F2} GB/s effective");
            float* spot = (float*)bigRecv0.DataPointer;
            Assert.True(spot[big] == 0f && spot[big + 2] == 1.0f, "big AllGather block-1 spot check failed");
        }
    }

    /// <summary>The universal fallback must be numerically correct on plain host tensors with no GPU at all,
    /// and the factory must pick it (with a logged reason) when ranks share one CUDA device — NCCL requires
    /// distinct devices, and silently picking it anyway would hand two threads one communicator device.</summary>
    [Fact]
    public unsafe void HostStaged_Fallback_CorrectAndChosenForDuplicateOrdinals()
    {
        const long n = 8192;
        using (HostStagedComm comm = new(2))
        using (Tensor a = Filled(n, i => i * 0.125f))
        using (Tensor b = Filled(n, i => 1024f - i * 0.0625f))
        {
            float[] expected = new float[n];
            float* pa = (float*)a.DataPointer;
            float* pb = (float*)b.DataPointer;
            for (long i = 0; i < n; i++) expected[i] = pa[i] + pb[i];
            comm.AllReduceSum([a, b]);
            for (long i = 0; i < n; i++)
            {
                Assert.True(((float*)a.DataPointer)[i] == expected[i] && ((float*)b.DataPointer)[i] == expected[i],
                    $"host-staged AllReduce mismatch at {i}");
            }
        }

        using (HostStagedComm comm = new(2))
        using (Tensor s0 = Filled(16, i => i))
        using (Tensor s1 = Filled(16, i => 100f + i))
        using (Tensor r0 = new(new TensorShape(32), DType.F32))
        using (Tensor r1 = new(new TensorShape(32), DType.F32))
        {
            comm.AllGather([s0, s1], [r0, r1]);
            float* r = (float*)r1.DataPointer;
            Assert.True(r[0] == 0f && r[15] == 15f && r[16] == 100f && r[31] == 115f, "host-staged AllGather layout wrong");
        }

        if (!CudaContext.IsAvailable()) { _output.WriteLine("factory duplicate-ordinal check skipped: CUDA unavailable"); return; }
        using CudaBackend first = new CudaBackend(0, PtxDir());
        using CudaBackend second = new CudaBackend(0, PtxDir());
        using ICollectiveComm fromFactory = CollectiveComm.Create([first, second]);
        _output.WriteLine($"duplicate-ordinal factory transport = {fromFactory.Transport}");
        Assert.Equal("host-staged", fromFactory.Transport);
    }
}
