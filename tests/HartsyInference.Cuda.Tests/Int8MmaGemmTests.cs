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

    [Theory]
    // N must be a whole multiple of the 256-wide block tile (only M is predicated), so these shapes changed with
    // the tile: 384 is no longer expressible.
    [InlineData(256, 256, 128, 0u, "small")]
    [InlineData(200, 256, 192, 0u, "raggedM")]        // M not a multiple of 128 -> epilogue predication
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
    [InlineData(8118, 4096, 4096, 0u, "raggedM_deepK_chunk1")]
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

    /// <summary>Head-to-head at LTX-2.5's real shapes: fused kernel vs the cuBLASLt GEMM + dequant pair it
    /// would replace, both timed end-to-end. Diagnostic — it prints, it does not gate.</summary>
    [Theory]
    [InlineData(4992, 16384, 4096, "ffn_up")]
    [InlineData(4992, 4096, 16384, "ffn_down")]
    [InlineData(4992, 4096, 4096, "attn_qkvo")]
    // Small-m shapes the `k <= 2n` gate also admits: audio attention/FFN and the text-side k/v projections.
    // A 128x256 block tile at m=256 launches 2 M-blocks, so the whole grid can be a fraction of one wave — the
    // regime where a big-tile kernel is structurally wrong and cuBLASLt's small-m heuristic wins.
    [InlineData(256, 2048, 2048, "audio_attn")]
    [InlineData(256, 8192, 2048, "audio_ffn_up")]
    [InlineData(512, 4096, 4096, "text_kv")]
    public void FusedMmaGemm_VersusCublasLtPair_Throughput(int m, int n, int k, string label)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        CudaKernels ker = cuda.Kernels!;
        using Int8GemmExecutor gemm = new Int8GemmExecutor();

        ulong dA = GpuTransferHelper.AllocateDevice((nuint)((long)m * k));
        ulong dAS = GpuTransferHelper.AllocateDevice((nuint)(m * 4));
        ulong dWS = GpuTransferHelper.AllocateDevice((nuint)(n * 4));
        ulong dOut = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 2));
        ulong dAcc = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 4));
        using ColdBuffers weights = new ColdBuffers((nuint)((long)n * k));
        try
        {
            double flop = 2.0 * m * n * k;

            // Occupancy diagnostic: at 128x128x64 with STAGES shared stages, whether TWO blocks fit per SM is
            // decided by registers and shared bytes together, and it is the difference between 8 and 16 warps
            // of latency hiding. CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0, LOCAL_SIZE_BYTES = 3, NUM_REGS = 4.
            CudaDriverApi.cuFuncGetAttribute(out int numRegs, 4, ker.Int8MmaGemmFunction);
            CudaDriverApi.cuFuncGetAttribute(out int spillBytes, 3, ker.Int8MmaGemmFunction);
            CudaDriverApi.cuFuncGetAttribute(out int padRegs, 4, ker.Int8MmaGemmPadFunction);
            CudaDriverApi.cuFuncGetAttribute(out int padSpill, 3, ker.Int8MmaGemmPadFunction);
            _output.WriteLine($"  swizzled: {numRegs} regs/thread, {spillBytes} B local (spill), "
                + $"{CudaKernels.Int8MmaSharedBytes} B dynamic -> regs cap {65536 / Math.Max(1, numRegs * 256)} blocks/SM, "
                + $"shared caps {100 * 1024 / CudaKernels.Int8MmaSharedBytes} blocks/SM");
            // If the padded control's register count or spill moved, the file edit contaminated the BASELINE and
            // every delta below is void. It shipped at 195 regs, 0 spill.
            _output.WriteLine($"  padded  : {padRegs} regs/thread, {padSpill} B local (spill), "
                + $"{CudaKernels.Int8MmaSharedBytesPad} B dynamic");

            double mmaMs = BestMs(r => ker.LaunchInt8MmaGemmDequant(dOut, dA, weights[r], dAS, dWS, 0, m, n, k, 0u, 0, true));
            double padMs = BestMs(r => ker.LaunchInt8MmaGemmDequant(dOut, dA, weights[r], dAS, dWS, 0, m, n, k, 0u, 0, false));
            double pairMs = BestMs(r =>
            {
                gemm.Run(weights[r], dA, dAcc, m, n, k, 0);
                ker.LaunchW8A8DequantBias(dOut, dAcc, dAS, dWS, 0, m, n, 0, outF16: true, actMode: 0u);
            });
            // The pair is INVARIANT across kernel edits — if its TOPS moves between runs, the measurement is
            // drifting and the fused arms' numbers are not comparable either.
            double gemmOnlyMs = BestMs(r => gemm.Run(weights[r], dA, dAcc, m, n, k, 0));

            _output.WriteLine($"{label,-10} m={m} n={n} k={k}   swizzled {mmaMs:F3} ms = {flop / (mmaMs * 1e-3) / 1e12:F1} TOPS" +
                $"  |  padded {padMs:F3} ms = {flop / (padMs * 1e-3) / 1e12:F1} TOPS" +
                $"  |  pair {pairMs:F3} ms = {flop / (pairMs * 1e-3) / 1e12:F1} TOPS" +
                $"   (gemm alone {flop / (gemmOnlyMs * 1e-3) / 1e12:F1} TOPS, dequant {pairMs - gemmOnlyMs:F3} ms)" +
                $"  |  swizzled vs pair {(pairMs / mmaMs - 1) * 100:+0.0;-0.0}%" +
                $"  vs padded {(padMs / mmaMs - 1) * 100:+0.0;-0.0}%  [{weights.Count} cold weight buffers]");
            Assert.True(mmaMs > 0);
        }
        finally
        {
            foreach (ulong p in new[] { dA, dAS, dWS, dOut, dAcc }) GpuTransferHelper.FreeDevice(p);
        }
    }
}
