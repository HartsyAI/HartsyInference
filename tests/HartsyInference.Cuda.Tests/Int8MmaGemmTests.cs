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
    [InlineData(256, 256, 128, 0u, "small")]
    [InlineData(200, 256, 192, 0u, "raggedM")]        // M not a multiple of 128 -> epilogue predication
    [InlineData(256, 384, 128, 1u, "gelu")]
    [InlineData(4992, 4096, 4096, 0u, "attn_qkvo")]
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
            ker.LaunchInt8MmaGemmDequant(dMma, dA, dB, dAS, dWS, 0, m, n, k, actMode, 0);
            gemm.Run(dB, dA, dAcc, m, n, k, 0);
            ker.LaunchW8A8DequantBias(dRef, dAcc, dAS, dWS, 0, m, n, 0, outF16: true, actMode: actMode);
            cuda.Sync();

            GpuTransferHelper.CopyToHost(outMma, dMma, (nuint)((long)m * n * 2));
            GpuTransferHelper.CopyToHost(outRef, dRef, (nuint)((long)m * n * 2));
            using Tensor fa = outMma.CastTo(DType.F32);
            using Tensor fb = outRef.CastTo(DType.F32);
            float* pa = (float*)fa.DataPointer, pb = (float*)fb.DataPointer;
            double maxAbs = 0, maxRel = 0;
            for (long i = 0; i < (long)m * n; i++)
            {
                double d = Math.Abs(pa[i] - pb[i]);
                if (d > maxAbs) maxAbs = d;
                double denom = Math.Abs(pb[i]) + 1e-3;
                if (d / denom > maxRel) maxRel = d / denom;
            }
            _output.WriteLine($"{label}: max abs {maxAbs:G4}, max rel {maxRel:G4}");
            Assert.True(maxRel < 2e-2, $"{label}: max rel {maxRel} — fused mma disagrees with cuBLASLt+dequant");
        }
        finally
        {
            foreach (ulong p in new[] { dA, dB, dAS, dWS, dMma, dRef, dAcc }) GpuTransferHelper.FreeDevice(p);
        }
    }

    /// <summary>Head-to-head at LTX-2.5's real shapes: fused kernel vs the cuBLASLt GEMM + dequant pair it
    /// would replace, both timed end-to-end. Diagnostic — it prints, it does not gate.</summary>
    [Theory]
    [InlineData(4992, 16384, 4096, "ffn_up")]
    [InlineData(4992, 4096, 16384, "ffn_down")]
    [InlineData(4992, 4096, 4096, "attn_qkvo")]
    public void FusedMmaGemm_VersusCublasLtPair_Throughput(int m, int n, int k, string label)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        CudaKernels ker = cuda.Kernels!;
        using Int8GemmExecutor gemm = new Int8GemmExecutor();

        ulong dA = GpuTransferHelper.AllocateDevice((nuint)((long)m * k));
        ulong dB = GpuTransferHelper.AllocateDevice((nuint)((long)n * k));
        ulong dAS = GpuTransferHelper.AllocateDevice((nuint)(m * 4));
        ulong dWS = GpuTransferHelper.AllocateDevice((nuint)(n * 4));
        ulong dOut = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 2));
        ulong dAcc = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 4));
        try
        {
            const int Reps = 20;
            double flop = 2.0 * m * n * k;

            for (int i = 0; i < 3; i++) ker.LaunchInt8MmaGemmDequant(dOut, dA, dB, dAS, dWS, 0, m, n, k, 0u, 0);
            CudaDriverApi.cuStreamSynchronize(0);
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < Reps; i++) ker.LaunchInt8MmaGemmDequant(dOut, dA, dB, dAS, dWS, 0, m, n, k, 0u, 0);
            CudaDriverApi.cuStreamSynchronize(0);
            double mmaMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds / Reps;

            for (int i = 0; i < 3; i++)
            {
                gemm.Run(dB, dA, dAcc, m, n, k, 0);
                ker.LaunchW8A8DequantBias(dOut, dAcc, dAS, dWS, 0, m, n, 0, outF16: true, actMode: 0u);
            }
            CudaDriverApi.cuStreamSynchronize(0);
            t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < Reps; i++)
            {
                gemm.Run(dB, dA, dAcc, m, n, k, 0);
                ker.LaunchW8A8DequantBias(dOut, dAcc, dAS, dWS, 0, m, n, 0, outF16: true, actMode: 0u);
            }
            CudaDriverApi.cuStreamSynchronize(0);
            double pairMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds / Reps;

            _output.WriteLine($"{label,-10} m={m} n={n} k={k}   fused-mma {mmaMs:F3} ms = {flop / (mmaMs * 1e-3) / 1e12:F1} TOPS" +
                $"   |  cuBLASLt+dequant {pairMs:F3} ms = {flop / (pairMs * 1e-3) / 1e12:F1} TOPS" +
                $"   |  {(pairMs / mmaMs - 1) * 100:+0.0;-0.0}% vs pair");
            Assert.True(mmaMs > 0);
        }
        finally
        {
            foreach (ulong p in new[] { dA, dB, dAS, dWS, dOut, dAcc }) GpuTransferHelper.FreeDevice(p);
        }
    }
}
