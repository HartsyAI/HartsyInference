using System.Runtime.CompilerServices;

namespace HartsyInference.Cuda;

/// <summary>INT8 tensor-core (IMMA) GEMM via cuBLASLt for the W8A8 path (INFERENCE_ACCEL_GRIND §H5,
/// QUANTIZATION_LOW_PRECISION_INFERENCE §5). Ampere SM 8.6 (RTX 3060 class) has native INT8 tensor cores at
/// ~2× the F16 rate but NO fp8 MMA — int8 is that hardware's only true low-precision tensor-core GEMM.
///
/// <para>Layout matches <see cref="LtGemmExecutor"/>: row-major <c>D_i32[M, N] = input_i8[M, K] ·
/// weight_i8^T[N, K]</c>, dispatched TN (OP_T on the weight, OP_N on the input) with plain column-major
/// layouts — on cuBLASLt 12/13 the int8 TN plain-layout config routes to IMMA kernels directly, so the
/// Turing-era COL32/COL4_4R2_8C interleaved orderings (and their per-call cublasLtMatrixTransform cost) are
/// only needed if the heuristic refuses this config; probe with <see cref="IsSupported"/> + a smoke Run.
/// int8 TN requires K % 4 == 0 and N % 4 == 0 (lda/ldb/ldc multiples of 4) — every DiT shape in the fleet
/// satisfies this. Output stays raw int32; the caller's dequant epilogue applies
/// <c>actScale[row] · wScale · D</c> (+bias) — cuBLASLt cannot dequantize int32 into a float epilogue with
/// per-row vectors, so that stays a custom kernel.</para></summary>
public sealed unsafe class Int8GemmExecutor : IDisposable
{
    private nint _ltHandle;
    private ulong _workspace;
    private readonly nuint _workspaceBytes;
    private int _disposed;

    /// <summary>Whether the cuBLASLt handle was created (int8 IMMA itself needs SM 7.5+; the engine's CUDA
    /// floor is above that, so handle creation is the only gate — a per-shape heuristic failure still throws
    /// from <see cref="Run"/>).</summary>
    public bool IsSupported { get; }

    public Int8GemmExecutor()
    {
        if (CublasLtApi.cublasLtCreate(out _ltHandle) != 0)
        {
            IsSupported = false;
            _workspaceBytes = 0;
            return;
        }
        IsSupported = true;
        _workspaceBytes = (nuint)CublasLtApi.DefaultWorkspaceBytes;
        _workspace = CudaMemory.AllocatePersistent(_workspaceBytes);
    }

    /// <summary>Runs <c>D_i32[M, N] = input_i8[M, K] · weight_i8^T[N, K]</c> (alpha=1, beta=0, int32
    /// accumulate). All pointers are device pointers; <paramref name="outPtr"/> must hold M·N int32.</summary>
    /// <remarks>MEASURED 2026-08-13, do not re-chase: caching the descriptors + the heuristic result per shape
    /// (removing ~45k redundant host cuBLASLt calls per step) is within run noise, and autotuning the 16
    /// heuristic candidates on real buffers is WORSE (1543 vs 1510 ms/step) because a 3-rep timing is noisy
    /// enough to lock in a bad algo permanently. cuBLASLt's first heuristic pick is already its best; its
    /// int8 ceiling here is ~385 TOPS against comfy-kitchen's CUTLASS at 496-538.</remarks>
    public void Run(ulong weight, ulong input, ulong outPtr, int m, int n, int k, nint stream)
    {
        if (!IsSupported)
            throw new InvalidOperationException("Int8GemmExecutor.Run called without cuBLASLt support. Check IsSupported first.");
        ThrowIfDisposed();
        if (k % 4 != 0 || n % 4 != 0)
            throw new ArgumentException($"int8 TN GEMM requires K and N to be multiples of 4; got K={k}, N={n}.");

        nint matmulDesc = 0, layoutA = 0, layoutB = 0, layoutD = 0, pref = 0;
        try
        {
            CublasLtApi.cublasLtMatmulDescCreate(out matmulDesc, CublasApi.CUBLAS_COMPUTE_32I, CublasApi.CUDA_R_32I).ThrowOnCublasError();

            int transA = CublasApi.CUBLAS_OP_T;
            int transB = CublasApi.CUBLAS_OP_N;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int)).ThrowOnCublasError();
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int)).ThrowOnCublasError();

            // weight [N, K] row-major -> operand A as KxN col-major (OP_T). input [M, K] row-major
            // -> operand B as KxM col-major (OP_N). output D [M, N] row-major == NxM col-major, int32.
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutA, CublasApi.CUDA_R_8I, (ulong)k, (ulong)n, k).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutB, CublasApi.CUDA_R_8I, (ulong)k, (ulong)m, k).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutD, CublasApi.CUDA_R_32I, (ulong)n, (ulong)m, n).ThrowOnCublasError();

            CublasLtApi.cublasLtMatmulPreferenceCreate(out pref).ThrowOnCublasError();
            ulong wsBytes = _workspaceBytes;
            CublasLtApi.cublasLtMatmulPreferenceSetAttribute(
                pref, CublasLtApi.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, (nuint)sizeof(ulong)).ThrowOnCublasError();

            byte* heuristic = stackalloc byte[128];
            int hRc = CublasLtApi.cublasLtMatmulAlgoGetHeuristic(_ltHandle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
                pref, 1, heuristic, out int returnedAlgoCount);
            if (hRc != 0 || returnedAlgoCount < 1)
                throw new InvalidOperationException(
                    $"cuBLASLt int8 heuristic found no algorithm for {m}x{n}x{k} (rc={hRc}, count={returnedAlgoCount}) — " +
                    "plain-layout int8 TN unsupported here; the COL32 interleaved path would be the fallback.");

            int a = 1, beta = 0;
            int mRc = CublasLtApi.cublasLtMatmul(_ltHandle, matmulDesc,
                &a,
                weight, layoutA,
                input, layoutB,
                &beta,
                outPtr, layoutD,
                outPtr, layoutD,
                (nint)heuristic, (nint)_workspace, _workspaceBytes, stream);
            if (mRc != 0)
                throw new InvalidOperationException($"cublasLtMatmul int8 rc={mRc} for {m}x{n}x{k}.");
        }
        finally
        {
            if (pref != 0) CublasLtApi.cublasLtMatmulPreferenceDestroy(pref);
            if (layoutA != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutA);
            if (layoutB != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutB);
            if (layoutD != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutD);
            if (matmulDesc != 0) CublasLtApi.cublasLtMatmulDescDestroy(matmulDesc);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(Int8GemmExecutor));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        if (_workspace != 0)
        {
            CudaMemory.Free(_workspace);
            _workspace = 0;
        }
        if (_ltHandle != 0)
        {
            CublasLtApi.cublasLtDestroy(_ltHandle);
            _ltHandle = 0;
        }
        GC.SuppressFinalize(this);
    }

    ~Int8GemmExecutor()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        if (_workspace != 0)
        {
            CudaMemory.Free(_workspace);
            _workspace = 0;
        }
        if (_ltHandle != 0)
        {
            CublasLtApi.cublasLtDestroy(_ltHandle);
            _ltHandle = 0;
        }
    }
}
