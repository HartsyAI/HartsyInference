using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Asks cuBLASLt, on THIS driver, which int8 output configurations it will actually accept — so a
/// perf decision about the W8A8 dequant epilogue rests on the heuristic's answer instead of on doc-reading.
///
/// <para>Why it matters: the resident int8 chain writes an <c>[M, N]</c> INT32 accumulator to HBM and a
/// separate kernel reads it back to apply <c>actScale[row] · wScale[col]</c> (+bias, +GELU). At the LTX-2.5
/// FFN-up shape that round trip is 4992·16384·4 B each way ≈ 654 MB ≈ 0.65 ms — almost exactly the measured
/// per-call gap to comfy-kitchen's CUTLASS kernel, which fuses dequant into its epilogue. If cuBLASLt accepts
/// a 16F <c>D</c> the accumulator never reaches memory; if it also accepts a device-VECTOR alpha, the
/// per-output-channel weight scale rides along for free (our <c>D</c> is <c>[N, M]</c> column-major, so
/// cuBLASLt's alpha-per-row indexes N = the output channel). Diagnostic: it asserts nothing about speed,
/// it prints what the driver allows.</para></summary>
[Collection("CudaSerial")]
public sealed unsafe class Int8GemmEpilogueProbeTests
{
    private readonly ITestOutputHelper _output;
    public Int8GemmEpilogueProbeTests(ITestOutputHelper output) => _output = output;

    // cublasLtMatmulDescAttributes_t: 0 = COMPUTE_TYPE, 1 = SCALE_TYPE, 2 = POINTER_MODE.
    private const int CUBLASLT_MATMUL_DESC_POINTER_MODE = 2;
    // cublasLtPointerMode_t: 0 = HOST, 1 = DEVICE, 2 = DEVICE_VECTOR,
    // 3 = ALPHA_DEVICE_VECTOR_BETA_ZERO, 4 = ALPHA_DEVICE_VECTOR_BETA_HOST.
    private const int CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO = 3;
    private const int CUBLAS_COMPUTE_32F = 68;

    /// <param name="dTypeName">CUDA_R_* name for D.</param>
    private bool HeuristicAccepts(nint lt, int computeType, int scaleType, int dType, bool alphaVector,
        int m, int n, int k, out string detail)
    {
        nint desc = 0, la = 0, lb = 0, ld = 0, pref = 0;
        try
        {
            int rc = CublasLtApi.cublasLtMatmulDescCreate(out desc, computeType, scaleType);
            if (rc != 0) { detail = $"DescCreate rc={rc}"; return false; }

            int transA = CublasApi.CUBLAS_OP_T, transB = CublasApi.CUBLAS_OP_N;
            CublasLtApi.cublasLtMatmulDescSetAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int));
            CublasLtApi.cublasLtMatmulDescSetAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int));
            if (alphaVector)
            {
                int pm = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
                rc = CublasLtApi.cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(int));
                if (rc != 0) { detail = $"SetPointerMode rc={rc}"; return false; }
            }

            CublasLtApi.cublasLtMatrixLayoutCreate(out la, CublasApi.CUDA_R_8I, (ulong)k, (ulong)n, k);
            CublasLtApi.cublasLtMatrixLayoutCreate(out lb, CublasApi.CUDA_R_8I, (ulong)k, (ulong)m, k);
            rc = CublasLtApi.cublasLtMatrixLayoutCreate(out ld, dType, (ulong)n, (ulong)m, n);
            if (rc != 0) { detail = $"LayoutD rc={rc}"; return false; }

            CublasLtApi.cublasLtMatmulPreferenceCreate(out pref);
            ulong ws = CublasLtApi.DefaultWorkspaceBytes;
            CublasLtApi.cublasLtMatmulPreferenceSetAttribute(
                pref, CublasLtApi.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, (nuint)sizeof(ulong));

            byte* res = stackalloc byte[128 * 8];
            int hRc = CublasLtApi.cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 8, res, out int count);
            detail = $"heuristic rc={hRc} algos={count}";
            return hRc == 0 && count > 0;
        }
        finally
        {
            if (pref != 0) CublasLtApi.cublasLtMatmulPreferenceDestroy(pref);
            if (ld != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(ld);
            if (lb != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(lb);
            if (la != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(la);
            if (desc != 0) CublasLtApi.cublasLtMatmulDescDestroy(desc);
        }
    }

    /// <summary>Splits the resident int8 <c>Linear</c> into "the cuBLASLt GEMM" and "everything else"
    /// (ConvRot pass + per-row quant + the int32 dequant epilogue), by timing the bare
    /// <see cref="Int8GemmExecutor"/> at the same shape. Without this split it is impossible to tell whether
    /// to attack the GEMM kernel or the surrounding passes.</summary>
    [Theory]
    [InlineData(4992, 16384, 4096, "ffn_up")]
    [InlineData(4992, 4096, 16384, "ffn_down")]
    [InlineData(4992, 4096, 4096, "attn_qkvo")]
    public void GemmVersusChainSplit(int m, int n, int k, string label)
    {
        using CudaBackend cuda = new CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
        using Int8GemmExecutor gemm = new Int8GemmExecutor();
        Assert.True(gemm.IsSupported);

        ulong w8 = GpuTransferHelper.AllocateDevice((nuint)((long)n * k));
        ulong a8 = GpuTransferHelper.AllocateDevice((nuint)((long)m * k));
        ulong o32 = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * 4));
        try
        {
            nint stream = 0;
            for (int i = 0; i < 3; i++) gemm.Run(w8, a8, o32, m, n, k, stream);
            CudaDriverApi.cuStreamSynchronize(stream);

            const int Reps = 20;
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < Reps; i++) gemm.Run(w8, a8, o32, m, n, k, stream);
            CudaDriverApi.cuStreamSynchronize(stream);
            double gemmMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds / Reps;

            double tops = 2.0 * m * n * k / (gemmMs * 1e-3) / 1e12;
            // The int32 accumulator round trip the dequant epilogue forces: written by the GEMM, read back.
            double int32RoundTripMb = 2.0 * m * n * 4 / (1024.0 * 1024.0);
            _output.WriteLine($"{label,-10} m={m} n={n} k={k}  GEMM-only {gemmMs:F3} ms = {tops:F1} TOPS   " +
                $"| int32 round trip {int32RoundTripMb:F0} MB (~{int32RoundTripMb / 1024.0 / 0.9 * 1000:F2} ms at 900 GB/s)");
            Assert.True(gemmMs > 0);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(w8);
            GpuTransferHelper.FreeDevice(a8);
            GpuTransferHelper.FreeDevice(o32);
        }
    }

    [Fact]
    public void WhichInt8OutputConfigsDoesCublasLtAccept()
    {
        using CudaBackend _ = new CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
        Assert.Equal(0, CublasLtApi.cublasLtCreate(out nint lt));
        try
        {
            const int m = 4992, n = 16384, k = 4096;   // LTX-2.5 FFN-up, the shape with the biggest round trip
            (string Label, int Compute, int Scale, int D, bool Vec)[] cases =
            [
                ("32I compute, INT32 D, scalar alpha  (what we ship)", CublasApi.CUBLAS_COMPUTE_32I, CublasApi.CUDA_R_32I, CublasApi.CUDA_R_32I, false),
                ("32I compute, F16 D,   scalar alpha  (kills int32 HBM)", CublasApi.CUBLAS_COMPUTE_32I, CublasApi.CUDA_R_32I, CublasApi.CUDA_R_16F, false),
                ("32I compute, F16 D,   F32 scale", CublasApi.CUBLAS_COMPUTE_32I, CublasApi.CUDA_R_32F, CublasApi.CUDA_R_16F, false),
                ("32F compute, F16 D,   F32 scale", CUBLAS_COMPUTE_32F, CublasApi.CUDA_R_32F, CublasApi.CUDA_R_16F, false),
                ("32I compute, INT32 D, ALPHA VECTOR", CublasApi.CUBLAS_COMPUTE_32I, CublasApi.CUDA_R_32I, CublasApi.CUDA_R_32I, true),
                ("32I compute, F16 D,   ALPHA VECTOR", CublasApi.CUBLAS_COMPUTE_32I, CublasApi.CUDA_R_32F, CublasApi.CUDA_R_16F, true),
                ("32F compute, F16 D,   ALPHA VECTOR", CUBLAS_COMPUTE_32F, CublasApi.CUDA_R_32F, CublasApi.CUDA_R_16F, true),
            ];
            foreach ((string label, int comp, int scale, int d, bool vec) in cases)
            {
                bool ok = HeuristicAccepts(lt, comp, scale, d, vec, m, n, k, out string detail);
                _output.WriteLine($"{(ok ? "ACCEPT" : "reject")}  {label,-52} {detail}");
            }
        }
        finally { CublasLtApi.cublasLtDestroy(lt); }
    }
}
