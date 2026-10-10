using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The fused-dequant int8 mma GEMM, checked against the EXPLICIT pair it replaces (cuBLASLt int8
/// GEMM into an int32 accumulator, then <c>w8a8_dequant_bias</c>) — not against a second copy of its own
/// logic. Non-square shapes so a transposed axis cannot pass. Also reports achieved TOPS against the same
/// pair, since the whole point of the kernel is to beat that pair end-to-end: it must clear the PAIR's
/// throughput, not the bare GEMM's, to be worth wiring in.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("CudaSerial")]
public sealed unsafe class Int8MmaGemmTests
{
    private readonly ITestOutputHelper _output;
    public Int8MmaGemmTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor I8(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.I8);
        sbyte* p = (sbyte*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < (long)rows * cols; i++) p[i] = (sbyte)rng.Next(-127, 128);
        return t;
    }

    private static Tensor F32(int n, int seed, float lo, float hi)
    {
        Tensor t = new Tensor(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) p[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return t;
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    // N must be a whole multiple of the 256-wide block tile (only M is predicated), so these shapes changed with
    // the tile: 384 is no longer expressible.
    [InlineData(256, 256, 128, 0u, "small")]
    [InlineData(256, 512, 128, 1u, "gelu")]
    [InlineData(129, 768, 64, 0u, "raggedM_wideN")]   // one k-tile, 3 N tiles, ragged M
    [InlineData(4992, 4096, 4096, 0u, "attn_qkvo")]
    // The M the kernel actually sees is NOT the token count: Int8ResidentRowChunk splits it against a byte budget,
    // so 1280x736x145f's 17,480 tokens arrive as 9362 + 8118 — both ragged, at 64 k-tiles. Every ragged case above
    // is a shallow-K toy (1 or 3 k-tiles) and the only deep-K case is tile-aligned, so the shipping geometry's cell
    // was the one combination never covered. Worse, that budget derives from FREE VRAM unless
    // HARTSY_INT8_ROW_BUDGET_MB pins it, so these split points move run to run and a defect here would appear and
    // vanish across identical invocations.
    [InlineData(9362, 4096, 4096, 0u, "raggedM_deepK_chunk0")]
    public void FusedMmaGemm_MatchesCublasLtPlusDequant(int m, int n, int k, uint actMode, string label)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        CudaKernels ker = cuda.Kernels!;
        Assert.True(ker.HasInt8MmaGemm(m, n, k), $"fused mma unavailable for {m}x{n}x{k}");
        using Int8GemmExecutor gemm = new Int8GemmExecutor();

        using Tensor a = I8(m, k, 3), b = I8(n, k, 5);
        using Tensor actS = F32(m, 7, 0.002f, 0.02f), wS = F32(n, 11, 0.002f, 0.02f);
        using Tensor outMma = new Tensor(new TensorShape(m, n), DType.F16);
        using Tensor outRef = new Tensor(new TensorShape(m, n), DType.F16);

        ulong dA = GpuTransferHelper.CopyToDevice(a), dB = GpuTransferHelper.CopyToDevice(b);
        ulong dAS = GpuTransferHelper.CopyToDevice(actS), dWS = GpuTransferHelper.CopyToDevice(wS);
        ulong dMma = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 2));
        ulong dRef = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 2));
        ulong dAcc = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 4));
        try
        {
            gemm.Run(dB, dA, dAcc, m, n, k, 0);
            ker.LaunchW8A8DequantBias(dRef, dAcc, dAS, dWS, 0, m, n, 0, outF16: true, actMode: actMode);
            cuda.Sync();
            GpuTransferHelper.CopyToHost(outRef, dRef, (nuint)((long)m * n * 2));
            using Tensor fb = outRef.CastTo(DType.F32);

            // BOTH layout arms against the same reference, in one process. The swizzled kernel parks operands
            // somewhere else in shared; it does not change what mma sees, so anything but max abs 0 is an
            // indexing bug — the mainloop accumulates int32 (order-independent) and the epilogue is per-element.
            foreach (bool swizzle in new[] { true, false })
            {
                ker.LaunchInt8MmaGemmDequant(dMma, dA, dB, dAS, dWS, 0, m, n, k, actMode, 0, swizzle);
                cuda.Sync();
                GpuTransferHelper.CopyToHost(outMma, dMma, (nuint)((long)m * n * 2));
                using Tensor fa = outMma.CastTo(DType.F32);
                float* pa = (float*)fa.DataPointer, pb = (float*)fb.DataPointer;
                double maxAbs = 0, maxRel = 0;
                for (long i = 0; i < (long)m * n; i++)
                {
                    double d = Math.Abs(pa[i] - pb[i]);
                    if (d > maxAbs) maxAbs = d;
                    double denom = Math.Abs(pb[i]) + 1e-3;
                    if (d / denom > maxRel) maxRel = d / denom;
                }
                string arm = swizzle ? "swizzled" : "padded";
                _output.WriteLine($"{label} [{arm}]: max abs {maxAbs:G4}, max rel {maxRel:G4}");
                Assert.True(maxAbs == 0, $"{label} [{arm}]: max abs {maxAbs} — fused mma disagrees with cuBLASLt+dequant");
            }
        }
        finally
        {
            foreach (ulong p in new[] { dA, dB, dAS, dWS, dMma, dRef, dAcc }) GpuTransferHelper.FreeDevice(p);
        }
    }

    /// <summary>Times <paramref name="body"/> as the BEST of several batches rather than one batch's mean.
    /// A single 20-rep batch of these kernels runs 10-50 ms — too short for the GPU to leave its idle clock
    /// state, which made the invariant reference arm swing 393 -> 322 TOPS between runs and would have had this
    /// harness attribute clock noise to kernel edits. Min-of-batches after a long warmup is the robust estimator:
    /// contention and clock ramp only ever ADD time, so the floor is the real cost.</summary>
    private static double BestMs(Action<int> body, int warmup = 32, int batches = 3, int reps = 32)
    {
        for (int i = 0; i < warmup; i++) body(i);
        CudaDriverApi.cuStreamSynchronize(0);
        double best = double.MaxValue;
        for (int b = 0; b < batches; b++)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < reps; i++) body(i);
            CudaDriverApi.cuStreamSynchronize(0);
            double ms = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds / reps;
            if (ms < best) best = ms;
        }
        return best;
    }

    /// <summary>Round-robin set of weight buffers, sized so the working set cannot sit in L2.</summary>
    /// <remarks>Re-running a GEMM against ONE resident weight leaves it hot in the 4090's 72 MB L2 for every rep
    /// after the first, which flatters exactly the kernels that are bandwidth-hungry — and the fused mma kernel is
    /// the bandwidth-hungry one here (its whole 128×256 tile choice was an arithmetic-intensity argument). Measured
    /// that way it beat the cuBLASLt pair by +4.7%, and wiring it in made the model SLOWER. A real step touches
    /// each weight once, cold. Rotating over enough copies to exceed L2 is what makes this harness predictive.</remarks>
    private sealed class ColdBuffers : IDisposable
    {
        private readonly List<ulong> _buffers = [];
        public int Count => _buffers.Count;
        public ulong this[int i] => _buffers[i % _buffers.Count];

        public ColdBuffers(nuint bytesEach, long targetTotalBytes = 192L << 20)
        {
            int copies = (int)Math.Clamp(targetTotalBytes / (long)Math.Max(bytesEach, 1), 2, 24);
            for (int i = 0; i < copies; i++) _buffers.Add(GpuTransferHelper.AllocateDevice(bytesEach));
        }

        public void Dispose() { foreach (ulong p in _buffers) GpuTransferHelper.FreeDevice(p); }
    }

}
