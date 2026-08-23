using System.Runtime.CompilerServices;

namespace HartsyInference.Cuda;

/// <summary>Native FP8 GEMM via cublasLtMatmul, gated on Ada+ (SM 8.9+) GPUs.</summary>
/// <remarks>
/// <para>Single-tensor allocation lifecycle: a per-call descriptor + layout set is created and torn down
/// inside <see cref="Run"/>. The handle and workspace are owned by the caller and reused across calls.</para>
///
/// <para><b>Hardware requirement.</b> FP8 tensor-core paths exist only on Ada (SM 8.9+) and Hopper (SM 9.0+).
/// On Ampere (SM 8.0/8.6/8.7), the constructor returns <see cref="IsSupported"/> = false and callers must
/// fall back to the cast-then-FP16-GEMM path in <see cref="CudaBackend"/>.</para>
///
/// <para><b>Untested locally.</b> The author's hardware is RTX 3060 (SM 8.6 — Ampere). This path was written
/// from cuBLASLt documentation + reference samples but has not been exercised end-to-end on Ada hardware.
/// The dispatch is gated on <see cref="IsSupported"/> + an explicit opt-in flag in <see cref="CudaBackend"/>
/// until validated on a real Ada GPU.</para>
/// </remarks>
public sealed unsafe class Fp8GemmExecutor : IDisposable
{
    private nint _ltHandle;
    private ulong _workspace;
    private readonly nuint _workspaceBytes;
    private int _disposed;

    /// <summary>Whether the running GPU supports native FP8 GEMM (SM 8.9+).</summary>
    public bool IsSupported { get; }

    /// <summary>Compute capability detected at construction.</summary>
    public int SmMajor { get; }

    /// <summary>Compute capability detected at construction.</summary>
    public int SmMinor { get; }

    /// <summary>Initializes the executor. Allocates the cuBLASLt handle + workspace if SM ≥ 8.9; otherwise leaves itself in an unsupported state without allocating any GPU resources.</summary>
    public Fp8GemmExecutor(int smMajor, int smMinor)
    {
        SmMajor = smMajor;
        SmMinor = smMinor;
        IsSupported = (smMajor == 8 && smMinor >= 9) || smMajor >= 9;
        if (!IsSupported)
        {
            _workspaceBytes = 0;
            return;
        }

        CublasLtApi.cublasLtCreate(out _ltHandle).ThrowOnCublasError();
        _workspaceBytes = (nuint)CublasLtApi.DefaultWorkspaceBytes;
        _workspace = CudaMemory.AllocatePersistent(_workspaceBytes);
    }

    /// <summary>Runs an FP8 Linear GEMM on Ada+ matching <see cref="CudaBackend"/>'s row-major convention: <c>output[M, N] = input[M, K] · weight^T[N, K]</c>.</summary>
    /// <remarks>Per-tensor weight scale is folded into the cuBLAS alpha (a separate device pointer for the
    /// descriptor's A_SCALE_POINTER attribute could be wired later for true cublasLt-style per-tensor scaling,
    /// but for the typical ComfyUI fp8_scaled / BFL distilled case where every weight already has a single
    /// scalar Fp8ScaleFactor, alpha-folding is exact).
    ///
    /// <para>Operands:</para>
    /// <list type="bullet">
    /// <item><description><paramref name="weight"/> — FP8 weight tensor, device pointer, shape [N, K],
    /// row-major.</description></item>
    /// <item><description><paramref name="input"/> — FP8 activation, device pointer, shape [M, K],
    /// row-major.</description></item>
    /// <item><description><paramref name="outPtr"/> — F16 (or F32 when <paramref name="outF32"/>) output,
    /// device pointer, shape [M, N], row-major.</description></item>
    /// <item><description><paramref name="weightScale"/> — Per-tensor weight scale (cuBLAS alpha). Pass 1.0f
    /// when the tensor has no scale.</description></item>
    /// <item><description><paramref name="inputScaleDev"/> — Optional DEVICE pointer to one F32: the
    /// activation's per-tensor DEQUANT scale (<c>amax/448</c>, written by the absmax kernels). Wired to
    /// CUBLASLT_MATMUL_DESC_B_SCALE_POINTER so dynamic activation quantization needs no host sync.
    /// 0 = unscaled.</description></item>
    /// </list></remarks>
    public void Run(ulong weight, ulong input, ulong outPtr, int m, int n, int k, float weightScale, nint stream,
        ulong inputScaleDev = 0, bool outF32 = false)
    {
        if (!IsSupported)
        {
            throw new InvalidOperationException(
                $"Fp8GemmExecutor.Run called on unsupported hardware (SM {SmMajor}.{SmMinor}). Caller must check IsSupported first.");
        }
        ThrowIfDisposed();

        nint matmulDesc = 0, layoutA = 0, layoutB = 0, layoutC = 0;
        try
        {
            CublasLtApi.cublasLtMatmulDescCreate(
                out matmulDesc,
                CublasApi.CUBLAS_COMPUTE_32F,
                CublasApi.CUDA_R_32F).ThrowOnCublasError();

            int transA = CublasApi.CUBLAS_OP_T;
            int transB = CublasApi.CUBLAS_OP_N;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int)).ThrowOnCublasError();
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int)).ThrowOnCublasError();
            if (inputScaleDev != 0)
            {
                // Per-tensor activation dequant scale, read by the GEMM from device memory.
                ulong bScale = inputScaleDev;
                CublasLtApi.cublasLtMatmulDescSetAttribute(
                    matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale, (nuint)sizeof(ulong)).ThrowOnCublasError();
            }

            // weight: [N, K] fp8 transposed → operand A. input: [M, K] fp8 → operand B.
            // Output C: [M, N] f16/f32. Matches CudaBackend.Linear's CUBLAS_OP_T / CUBLAS_OP_N order.
            int outType = outF32 ? CublasApi.CUDA_R_32F : CublasApi.CUDA_R_16F;
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutA, CublasApi.CUDA_R_8F_E4M3, (ulong)k, (ulong)n, k).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutB, CublasApi.CUDA_R_8F_E4M3, (ulong)k, (ulong)m, k).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutC, outType, (ulong)n, (ulong)m, n).ThrowOnCublasError();

            float alpha = weightScale, beta = 0.0f;
            CublasLtApi.cublasLtMatmul(
                _ltHandle, matmulDesc,
                &alpha,
                weight, layoutA,
                input, layoutB,
                &beta,
                outPtr, layoutC,
                outPtr, layoutC,
                0, (nint)_workspace, _workspaceBytes, stream).ThrowOnCublasError();
        }
        finally
        {
            if (layoutA != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutA);
            if (layoutB != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutB);
            if (layoutC != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutC);
            if (matmulDesc != 0) CublasLtApi.cublasLtMatmulDescDestroy(matmulDesc);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(Fp8GemmExecutor));
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

    ~Fp8GemmExecutor()
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
