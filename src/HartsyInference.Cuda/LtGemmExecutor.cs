using System.Runtime.CompilerServices;

namespace HartsyInference.Cuda;

/// <summary>General-precision cuBLASLt GEMM with epilogue fusion (bias and/or activation folded into the
/// matmul).</summary>
/// <remarks>
/// <para>Unlike <see cref="Fp8GemmExecutor"/> this path works on every SM the engine targets (cuBLASLt ships
/// with CUDA 12), so it is usable on the RTX 3060 dev box. Fusing a per-output-channel bias add into the
/// epilogue removes a separate <c>BiasAdd</c> kernel launch plus an output-sized HBM round-trip per Linear;
/// the bias result is bit-equivalent to the unfused path.</para>
///
/// <para>Layout matches <see cref="CudaBackend.Linear"/> and <see cref="Fp8GemmExecutor"/>: row-major
/// <c>output[M, N] = input[M, K] · weight^T[N, K]</c>, dispatched as cuBLAS <c>OP_T</c> on the weight and
/// <c>OP_N</c> on the input. Compute is F32-accumulate (CUBLAS_COMPUTE_32F) regardless of operand dtype.</para>
/// </remarks>
public sealed unsafe class LtGemmExecutor : IDisposable
{
    private nint _ltHandle;
    private ulong _workspace;
    private readonly nuint _workspaceBytes;
    private int _disposed;

    /// <summary>Opt-out of TF32 tensor-core GEMM for F32 operands (fall back to plain F32 CUDA cores).</summary>
    private static readonly bool EnvOff = Environment.GetEnvironmentVariable("HARTSY_GEMM_NO_TF32") == "1";

    /// <summary>Whether the cuBLASLt handle was created successfully. False only if the cuBLASLt library is
    /// unavailable, in which case callers fall back to <c>cublasGemmEx</c> + separate bias add.</summary>
    public bool IsSupported { get; }

    /// <summary>Initializes the executor, allocating the cuBLASLt handle + workspace. If cuBLASLt is unavailable,
    /// leaves itself unsupported without allocating GPU resources.</summary>
    public LtGemmExecutor()
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

    /// <summary>Runs <c>output[M, N] = activation(alpha · input[M, K] · weight^T[N, K] + bias)</c> with the chosen
    /// epilogue. The heuristic algorithm is queried per call; for the inference shape set this is cheap relative
    /// to the GEMM, but a future optimization could cache the algo by (m, n, k, dtype).</summary>
    /// <param name="biasPtr">Per-output-channel bias of length N in <paramref name="dType"/>, or 0 when the
    /// epilogue carries no bias.</param>
    /// <param name="abType">cudaDataType of both operands (e.g. CUDA_R_16F).</param>
    /// <param name="dType">cudaDataType of the output (and bias).</param>
    /// <param name="epilogue">A <c>CUBLASLT_EPILOGUE_*</c> value.</param>
    public void Run(ulong weight, ulong input, ulong outPtr, int m, int n, int k, float alpha,
        int abType, int dType, ulong biasPtr, int epilogue, nint stream)
    {
        if (!IsSupported)
            throw new InvalidOperationException("LtGemmExecutor.Run called without cuBLASLt support. Check IsSupported first.");
        ThrowIfDisposed();

        nint matmulDesc = 0, layoutA = 0, layoutB = 0, layoutD = 0, pref = 0;
        try
        {
            // F32 operands: use TF32 tensor cores (CUBLAS_COMPUTE_32F_FAST_TF32) instead of plain F32 CUDA cores —
            // ~2× on Ada with a 10-bit multiply mantissa (F32 range preserved, F32 accumulate). This is the
            // PyTorch/ComfyUI default (allow_tf32) and diffusion tolerates it. F16/other operands keep COMPUTE_32F,
            // which already runs on F16 tensor cores (F16 multiply, F32 accumulate) — only the F32-operand path was
            // leaving the tensor cores idle (the dominant Wan-DiT GEMM cost). Opt-out via HARTSY_GEMM_NO_TF32=1.
            int computeType = (abType == CublasApi.CUDA_R_32F && !EnvOff)
                ? CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32
                : CublasApi.CUBLAS_COMPUTE_32F;
            CublasLtApi.cublasLtMatmulDescCreate(out matmulDesc, computeType, CublasApi.CUDA_R_32F).ThrowOnCublasError();

            int transA = CublasApi.CUBLAS_OP_T;
            int transB = CublasApi.CUBLAS_OP_N;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int)).ThrowOnCublasError();
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int)).ThrowOnCublasError();

            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(int)).ThrowOnCublasError();
            if (biasPtr != 0)
            {
                ulong biasArg = biasPtr;
                CublasLtApi.cublasLtMatmulDescSetAttribute(
                    matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasArg, (nuint)sizeof(ulong)).ThrowOnCublasError();
            }

            // weight [N, K] row-major -> operand A as KxN col-major (OP_T). input [M, K] row-major
            // -> operand B as KxM col-major (OP_N). output D [M, N] row-major == NxM col-major.
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutA, abType, (ulong)k, (ulong)n, k).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutB, abType, (ulong)k, (ulong)m, k).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutD, dType, (ulong)n, (ulong)m, n).ThrowOnCublasError();

            CublasLtApi.cublasLtMatmulPreferenceCreate(out pref).ThrowOnCublasError();
            ulong wsBytes = _workspaceBytes;
            CublasLtApi.cublasLtMatmulPreferenceSetAttribute(
                pref, CublasLtApi.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, (nuint)sizeof(ulong)).ThrowOnCublasError();

            // cublasLtMatmulHeuristicResult_t is an opaque blob; the first field is the algo
            // struct (cublasLtMatmulAlgo_t = 8 x uint64). A single 128-byte buffer covers it
            // with margin, and we only consume the leading algo via its address.
            byte* heuristic = stackalloc byte[128];
            int hRc = CublasLtApi.cublasLtMatmulAlgoGetHeuristic(_ltHandle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
                pref, 1, heuristic, out int returnedAlgoCount);
            if (hRc != 0)
                throw new InvalidOperationException(
                    $"cublasLtMatmulAlgoGetHeuristic rc={hRc} for {m}x{n}x{k} abType={abType} dType={dType} " +
                    $"epilogue={epilogue} bias={(biasPtr != 0)}.");
            if (returnedAlgoCount < 1)
                throw new InvalidOperationException(
                    $"cuBLASLt found no algorithm for GEMM {m}x{n}x{k} (abType={abType}, dType={dType}, epilogue={epilogue}).");

            float a = alpha, beta = 0.0f;
            int mRc = CublasLtApi.cublasLtMatmul(_ltHandle, matmulDesc,
                &a,
                weight, layoutA,
                input, layoutB,
                &beta,
                outPtr, layoutD,
                outPtr, layoutD,
                (nint)heuristic, (nint)_workspace, _workspaceBytes, stream);
            if (mRc != 0)
                throw new InvalidOperationException(
                    $"cublasLtMatmul rc={mRc} for {m}x{n}x{k} abType={abType} dType={dType} epilogue={epilogue} bias={(biasPtr != 0)}.");
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
            throw new ObjectDisposedException(nameof(LtGemmExecutor));
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

    ~LtGemmExecutor()
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
