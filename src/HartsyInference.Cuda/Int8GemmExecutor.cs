using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>INT8 tensor-core (IMMA) GEMM via cuBLASLt for the W8A8 path (INFERENCE_ACCEL_GRIND §H5, QUANTIZATION_LOW_PRECISION_INFERENCE §5). Ampere SM 8.6 (RTX 3060 class) has native INT8 tensor cores at ~2× the F16 rate but NO fp8 MMA — int8 is that hardware's only true low-precision tensor-core GEMM. <para>Layout matches <see cref="LtGemmExecutor"/>: row-major <c>D_i32[M, N] = input_i8[M, K] · weight_i8^T[N, K]</c>, dispatched TN (OP_T on the weight, OP_N on the input) with plain column-major layouts — on cuBLASLt 12/13 the int8 TN plain-layout config routes to IMMA kernels directly, so the Turing-era COL32/COL4_4R2_8C interleaved orderings (and their per-call cublasLtMatrixTransform cost) are only needed if the heuristic refuses this config; probe with <see cref="CublasLtExecutorBase.IsSupported"/> + a smoke Run. int8 TN requires K % 4 == 0 and N % 4 == 0 (lda/ldb/ldc multiples of 4) — every DiT shape in the fleet satisfies this. Output stays raw int32; the caller's dequant epilogue applies <c>actScale[row] · wScale · D</c> (+bias) — cuBLASLt cannot dequantize int32 into a float epilogue with per-row vectors, so that stays a custom kernel.</para></summary>
public sealed unsafe class Int8GemmExecutor : CublasLtExecutorBase
{
    /// <summary>int8 IMMA itself needs SM 7.5+, below the engine's CUDA floor, so handle creation is the only gate; a per-shape heuristic miss still throws from <see cref="Run"/>.</summary>
    public Int8GemmExecutor() : base(supported: true)
    {
    }

    /// <summary>Runs <c>D_i32[M, N] = input_i8[M, K] · weight_i8^T[N, K]</c> (alpha=1, beta=0, int32 accumulate). All pointers are device pointers; <paramref name="outPtr"/> must hold M·N int32.</summary>
    /// <remarks><para>The five per-call cuBLASLt object creations below look like obvious waste and are not.
    /// Caching the descriptor, the layouts and the heuristic pick — removing ~45k host cuBLASLt calls per LTX-2.5
    /// step — was built and measured on 2026-08-13 with 4 interleaved reps per arm: <b>1428.5 vs 1428.4 ms/step,
    /// paired mean +0.1, t = 0.02</b>. A dead null, so the cache was reverted rather than carried. The reason is
    /// structural and applies to every host-side optimization on this path: the GPU runs at 99-100% SM occupancy,
    /// so host time spent queuing work is hidden behind execution and never reaches the wall clock. The driver
    /// calls really are slow in isolation (<c>Int8ResidentHostCostTests</c> measures <c>cuMemGetInfo</c> at 5.2 µs
    /// and a pool alloc/free pair at 1.2 µs); that is simply not where the time goes.</para>
    /// <para>An earlier version of this note reported the same conclusion from a single run per arm, before the
    /// harness's ~25 ms/step between-run spread was quantified — which established nothing. The numbers above
    /// replace it.</para>
    /// <para>Also do-not-re-chase: autotuning the 16 heuristic candidates on real buffers is WORSE (1543 vs
    /// 1510 ms/step) because a 3-rep timing is noisy enough to lock in a bad algo permanently. cuBLASLt's first
    /// heuristic pick is already its best; its int8 ceiling here is ~385 TOPS against comfy-kitchen's CUTLASS at
    /// 496-538 — that gap is a KERNEL problem (see VIDEO.md for their exact CUTLASS config), not a host one.</para></remarks>
    public void Run(ulong weight, ulong input, ulong outPtr, int m, int n, int k, nint stream)
    {
        if (!IsSupported)
            throw new InvalidOperationException("Int8GemmExecutor.Run called without cuBLASLt support. Check IsSupported first.");
        ThrowIfDisposed();
        if (k % 4 != 0 || n % 4 != 0)
            throw new ArgumentException($"int8 TN GEMM requires K and N to be multiples of 4; got K={k}, N={n}.");

        nint desc = 0, layoutA = 0, layoutB = 0, layoutD = 0, pref = 0;
        try
        {
            desc = CreateTnMatmulDesc(CublasApi.CUBLAS_COMPUTE_32I, CublasApi.CUDA_R_32I);
            CreateTnLayouts(CublasApi.DataTypeOf(DType.I8), CublasApi.DataTypeOf(DType.I8),
                CublasApi.DataTypeOf(DType.I32), m, n, k, out layoutA, out layoutB, out layoutD);

            CublasLtApi.cublasLtMatmulPreferenceCreate(out pref).ThrowOnCublasError();
            ulong wsBytes = WorkspaceBytes;
            CublasLtApi.cublasLtMatmulPreferenceSetAttribute(
                pref, CublasLtApi.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, (nuint)sizeof(ulong)).ThrowOnCublasError();

            byte* heuristic = stackalloc byte[128];
            int hRc = CublasLtApi.cublasLtMatmulAlgoGetHeuristic(LtHandle, desc, layoutA, layoutB, layoutD, layoutD,
                pref, 1, heuristic, out int returnedAlgoCount);
            if (hRc != 0 || returnedAlgoCount < 1)
                throw new InvalidOperationException(
                    $"cuBLASLt int8 heuristic found no algorithm for {m}x{n}x{k} (rc={hRc}, count={returnedAlgoCount}) — " +
                    "plain-layout int8 TN unsupported here; the COL32 interleaved path would be the fallback.");

            int a = 1, beta = 0;
            int mRc = Matmul(desc, &a, weight, layoutA, input, layoutB, &beta, outPtr, layoutD, (nint)heuristic, stream);
            if (mRc != 0)
                throw new InvalidOperationException($"cublasLtMatmul int8 rc={mRc} for {m}x{n}x{k}.");
        }
        finally
        {
            if (pref != 0) CublasLtApi.cublasLtMatmulPreferenceDestroy(pref);
            DestroyLayouts(layoutA, layoutB, layoutD);
            DestroyDesc(desc);
        }
    }
}
