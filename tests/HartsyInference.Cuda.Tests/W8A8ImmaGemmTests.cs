using System.Diagnostics;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Stage-1 gate for the W8A8 IMMA lane (INFERENCE_ACCEL_GRIND §H5): (1) the cuBLASLt int8
/// TN plain-layout GEMM is CORRECT vs a CPU int32 reference, and (2) raw int8 GEMM time vs the engine's
/// F16 cuBLASLt GEMM at real DiT shapes — the UPPER BOUND of any W8A8 win (quant/dequant overhead only
/// subtracts from it). If this A/B doesn't clear ~1.3×, the whole calibration+epilogue build is dead
/// before it starts (measure-first). Interleaved trials = in-run clock control. Run explicitly:
///   CUDA_VISIBLE_DEVICES=1 dotnet test --filter "FullyQualifiedName~W8A8ImmaGemmTests"
/// (Category=W8A8Bench, excluded from sweeps; CVD=1 lands on the 3060 — the IMMA target class.)</summary>
[Collection("CudaSerial")]
[Trait("Category", "W8A8Bench")]
public sealed unsafe class W8A8ImmaGemmTests
{
    private readonly ITestOutputHelper _output;
    public W8A8ImmaGemmTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static ulong Upload(void* host, long bytes)
    {
        ulong d = GpuTransferHelper.AllocateDevice((nuint)bytes);
        CudaDriverApi.cuMemcpyHtoD(d, (nint)host, (nuint)bytes).ThrowOnError();
        return d;
    }

    [Fact]
    public void Int8Gemm_MatchesCpuInt32Reference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int M = 16, N = 32, K = 64;

        sbyte[] input = new sbyte[M * K];
        sbyte[] weight = new sbyte[N * K];
        Random rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = (sbyte)rng.Next(-127, 128);
        for (int i = 0; i < weight.Length; i++) weight[i] = (sbyte)rng.Next(-127, 128);

        int[] expected = new int[M * N];
        for (int mi = 0; mi < M; mi++)
            for (int ni = 0; ni < N; ni++)
            {
                int acc = 0;
                for (int ki = 0; ki < K; ki++) acc += input[mi * K + ki] * weight[ni * K + ki];
                expected[mi * N + ni] = acc;
            }

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Int8GemmExecutor exec = new Int8GemmExecutor();
        Assert.True(exec.IsSupported, "cuBLASLt unavailable");

        ulong dIn = 0, dW = 0, dOut = 0;
        try
        {
            fixed (sbyte* pi = input) dIn = Upload(pi, input.Length);
            fixed (sbyte* pw = weight) dW = Upload(pw, weight.Length);
            dOut = GpuTransferHelper.AllocateDevice((nuint)(M * N * sizeof(int)));

            exec.Run(dW, dIn, dOut, M, N, K, cuda.Stream.Handle);
            cuda.Sync();

            int[] actual = new int[M * N];
            fixed (int* pa = actual)
                CudaDriverApi.cuMemcpyDtoH((nint)pa, dOut, (nuint)(M * N * sizeof(int))).ThrowOnError();

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i]);
            _output.WriteLine($"int8 TN plain-layout GEMM exact vs CPU int32 reference ({M}x{N}x{K}).");
        }
        finally
        {
            if (dIn != 0) GpuTransferHelper.FreeDevice(dIn);
            if (dW != 0) GpuTransferHelper.FreeDevice(dW);
            if (dOut != 0) GpuTransferHelper.FreeDevice(dOut);
        }
    }

    /// <summary>Raw IMMA-vs-F16 GEMM upper-bound A/B at DiT shapes (Chroma/Flux class: hidden 3072,
    /// S=4608 joint 1024² sequence).</summary>
    [Theory]
    [InlineData(4608, 3072, 3072)]     // per-projection qkv/out
    [InlineData(4608, 9216, 3072)]     // fused qkv
    [InlineData(4608, 12288, 3072)]    // FFN up / proj_mlp
    [InlineData(4608, 3072, 15360)]    // single-block proj_out (K = hidden + mlp)
    public void Int8Gemm_Vs_F16Gemm_DitShapes(int m, int n, int k)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Int8GemmExecutor int8 = new Int8GemmExecutor();
        Assert.True(int8.IsSupported && cuda.LtGemm.IsSupported, "cuBLASLt unavailable");

        Random rng = new Random(7);
        sbyte[] i8Host = new sbyte[(long)Math.Max((long)m * k, (long)n * k)];
        for (int i = 0; i < i8Host.Length; i++) i8Host[i] = (sbyte)rng.Next(-127, 128);
        Half[] f16Host = new Half[i8Host.Length];
        for (int i = 0; i < f16Host.Length; i++) f16Host[i] = (Half)(rng.NextDouble() * 0.1 - 0.05);

        ulong dIn8 = 0, dW8 = 0, dOut32 = 0, dIn16 = 0, dW16 = 0, dOut16 = 0;
        try
        {
            fixed (sbyte* p = i8Host)
            {
                dIn8 = Upload(p, (long)m * k);
                dW8 = Upload(p, (long)n * k);
            }
            fixed (Half* p = f16Host)
            {
                dIn16 = Upload(p, (long)m * k * 2);
                dW16 = Upload(p, (long)n * k * 2);
            }
            dOut32 = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * sizeof(int)));
            dOut16 = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 2));

            nint stream = cuda.Stream.Handle;
            // Warmup both arms (algo heuristic + JIT).
            int8.Run(dW8, dIn8, dOut32, m, n, k, stream);
            cuda.LtGemm.Run(dW16, dIn16, dOut16, m, n, k, 1.0f,
                CublasApi.CUDA_R_16F, CublasApi.CUDA_R_16F, 0, CublasLtApi.CUBLASLT_EPILOGUE_DEFAULT, stream);
            cuda.Sync();

            const int Trials = 10;
            double[] i8Ms = new double[Trials];
            double[] f16Ms = new double[Trials];
            Stopwatch sw = new Stopwatch();
            for (int t = 0; t < Trials; t++)
            {
                sw.Restart();
                int8.Run(dW8, dIn8, dOut32, m, n, k, stream);
                cuda.Sync();
                sw.Stop();
                i8Ms[t] = sw.Elapsed.TotalMilliseconds;

                sw.Restart();
                cuda.LtGemm.Run(dW16, dIn16, dOut16, m, n, k, 1.0f,
                    CublasApi.CUDA_R_16F, CublasApi.CUDA_R_16F, 0, CublasLtApi.CUBLASLT_EPILOGUE_DEFAULT, stream);
                cuda.Sync();
                sw.Stop();
                f16Ms[t] = sw.Elapsed.TotalMilliseconds;
            }

            double MeanOf(double[] xs) { double s = 0; foreach (double v in xs) s += v; return s / xs.Length; }
            double StdOf(double[] xs, double mean) { double s = 0; foreach (double v in xs) s += (v - mean) * (v - mean); return Math.Sqrt(s / (xs.Length - 1)); }
            double mI = MeanOf(i8Ms), mF = MeanOf(f16Ms);
            double sI = StdOf(i8Ms, mI), sF = StdOf(f16Ms, mF);
            double welch = (mF - mI) / Math.Sqrt(sI * sI / Trials + sF * sF / Trials);
            double tflops = 2.0 * m * n * k / (mI * 1e9);
            _output.WriteLine($"[{m}x{n}x{k}] int8: {mI:F3} ± {sI:F3} ms ({tflops:F1} TOPS) | " +
                $"f16: {mF:F3} ± {sF:F3} ms | int8 speedup {mF / mI:F3}x | Welch t={welch:F1}");
            _output.WriteLine($"  int8 trials: [{string.Join(", ", i8Ms.Select(v => v.ToString("F3")))}]");
            _output.WriteLine($"  f16 trials:  [{string.Join(", ", f16Ms.Select(v => v.ToString("F3")))}]");
        }
        finally
        {
            if (dIn8 != 0) GpuTransferHelper.FreeDevice(dIn8);
            if (dW8 != 0) GpuTransferHelper.FreeDevice(dW8);
            if (dOut32 != 0) GpuTransferHelper.FreeDevice(dOut32);
            if (dIn16 != 0) GpuTransferHelper.FreeDevice(dIn16);
            if (dW16 != 0) GpuTransferHelper.FreeDevice(dW16);
            if (dOut16 != 0) GpuTransferHelper.FreeDevice(dOut16);
        }
    }
}
