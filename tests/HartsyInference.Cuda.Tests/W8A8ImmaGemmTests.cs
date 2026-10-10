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

    [Trait("Category", "GpuIntegration")]
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

    /// <summary>Correctness gate for the SmoothQuant <c>invScale</c> param added to
    /// <c>w8a8_quant_rowwise_{f16,f32}</c> (W8A8_HANDOFF.md item 1, offline-gate-confirmed 2026-07-24):
    /// X_hat = X * invScale must be computed BEFORE the per-row absmax/quantize, i.e. the row's dequant
    /// scale itself must reflect the smoothed magnitudes, not just the final int8 values. Runs the kernel
    /// with a per-channel invScale vector and compares (rowScale, int8 q) against a CPU reference that
    /// applies the exact same scale-then-quantize formula host-side. Also runs with invScale=0 (the
    /// existing no-smoothing path) to confirm the null-pointer branch is unaffected — this is the
    /// regression half of the gate.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false, 8, 64)]  // no smoothing (invScale=0): must exactly match the pre-existing behavior
    [InlineData(true, 17, 130)] // non-multiple-of-4 cols/rows, srcF16
    public void W8A8QuantRowwise_InvScale_MatchesCpuReference(bool useInvScale, int rows, int cols)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        if (!cuda.Kernels!.HasW8A8Kernels) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }

        Random rng = new Random(7);
        Half[] act = new Half[(long)rows * cols];
        for (int i = 0; i < act.Length; i++) act[i] = (Half)((rng.NextDouble() * 2 - 1) * (0.05 + 5 * rng.NextDouble()));
        float[] invScale = new float[cols];
        for (int i = 0; i < cols; i++) invScale[i] = useInvScale ? (float)(0.2 + 2.0 * rng.NextDouble()) : 1f;

        // CPU reference: exact port of the kernel's math (per-row absmax over the SMOOTHED values, then
        // round-to-nearest, clamp[-127,127]).
        sbyte[] expectedQ = new sbyte[(long)rows * cols];
        float[] expectedRowScale = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            float amax = 0f;
            for (int c = 0; c < cols; c++)
            {
                float v = (float)act[r * cols + c] * invScale[c];
                amax = MathF.Max(amax, MathF.Abs(v));
            }
            float scale = amax > 0f ? amax / 127f : 1f;
            float inv = amax > 0f ? 127f / amax : 0f;
            expectedRowScale[r] = scale;
            for (int c = 0; c < cols; c++)
            {
                float v = (float)act[r * cols + c] * invScale[c] * inv;
                int iv = (int)MathF.Round(v);
                if (iv > 127) iv = 127;
                if (iv < -127) iv = -127;
                expectedQ[r * cols + c] = (sbyte)iv;
            }
        }

        ulong dAct = 0, dQ = 0, dRowScale = 0, dInvScale = 0;
        try
        {
            fixed (Half* p = act) dAct = Upload(p, (long)rows * cols * 2);
            dQ = GpuTransferHelper.AllocateDevice((nuint)((long)rows * cols));
            dRowScale = GpuTransferHelper.AllocateDevice((nuint)(rows * sizeof(float)));
            if (useInvScale) fixed (float* p = invScale) dInvScale = Upload(p, cols * sizeof(float));

            cuda.Kernels!.LaunchW8A8QuantRowwise(dQ, dRowScale, dAct, rows, cols, cuda.Stream.Handle,
                srcF16: true, invScale: dInvScale);
            cuda.Sync();

            sbyte[] actualQ = new sbyte[(long)rows * cols];
            float[] actualRowScale = new float[rows];
            fixed (sbyte* p = actualQ) CudaDriverApi.cuMemcpyDtoH((nint)p, dQ, (nuint)actualQ.Length).ThrowOnError();
            fixed (float* p = actualRowScale) CudaDriverApi.cuMemcpyDtoH((nint)p, dRowScale, (nuint)(rows * sizeof(float))).ThrowOnError();

            for (int r = 0; r < rows; r++)
                Assert.True(MathF.Abs(actualRowScale[r] - expectedRowScale[r]) <= 1e-6f * MathF.Max(1f, MathF.Abs(expectedRowScale[r])),
                    $"row {r} scale mismatch: expected {expectedRowScale[r]:e6}, got {actualRowScale[r]:e6}");
            for (int i = 0; i < expectedQ.Length; i++)
                Assert.Equal(expectedQ[i], actualQ[i]);
            _output.WriteLine($"invScale={useInvScale} [{rows}x{cols}]: exact match vs CPU reference " +
                $"({rows} row scales, {expectedQ.Length} int8 values).");
        }
        finally
        {
            foreach (ulong ptr in new[] { dAct, dQ, dRowScale, dInvScale })
                if (ptr != 0) GpuTransferHelper.FreeDevice(ptr);
        }
    }

    /// <summary>Full W8A8 chain (per-row activation quant → IMMA GEMM → dequant+bias epilogue) vs the F16
    /// GEMM with fused bias: accuracy on Gaussian-ish activations and per-channel-quantized weights, plus
    /// interleaved wall-time A/B. Accuracy expectation: per-row+per-channel INT8 keeps relL2 in the ~1e-2
    /// class (the ViDiT-Q W8A8 regime); the perf number is the HONEST chain speedup, prologue+epilogue paid.</summary>
    /// <summary>Raw IMMA-vs-F16 GEMM upper-bound A/B at DiT shapes (Chroma/Flux class: hidden 3072,
    /// S=4608 joint 1024² sequence).</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(4608, 3072, 3072)]     // per-projection qkv/out
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
