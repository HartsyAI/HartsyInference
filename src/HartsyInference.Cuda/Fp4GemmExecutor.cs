using System.Runtime.CompilerServices;

namespace HartsyInference.Cuda;

/// <summary>Native FP4 (e2m1) GEMM via cublasLtMatmul, gated on Blackwell (SM 10.0+) GPUs. Mirrors
/// <see cref="Fp8GemmExecutor"/>: a per-call descriptor + layout set is created and torn down inside
/// <see cref="Run"/>; the handle and workspace are owned here and reused across calls.
///
/// <para><b>Hardware requirement.</b> Tensor-core FP4 paths exist only on Blackwell (SM 10.0 for
/// datacenter B100/B200, SM 12.0 for consumer RTX 50xx). On everything earlier the constructor reports
/// <see cref="IsSupported"/> = false and callers must fall back to dequant→F16 GEMM (the same strategy
/// the engine uses for MXFP4/NVFP4/NF4 today).</para>
///
/// <para><b>Untested locally and not yet wired into dispatch.</b> The author's hardware is RTX 3060
/// (SM 8.6, Ampere) — exactly the situation <see cref="Fp8GemmExecutor"/> documents. This executor was
/// written from the cuBLASLt 12.8 FP4 documentation but has never been exercised on Blackwell. It is
/// therefore <b>not</b> referenced by <c>CudaBackend</c>'s GEMM dispatch yet; it compiles and is ready
/// for validation. Two things must be confirmed on real hardware before wiring it in:</para>
/// <list type="number">
///   <item>The block-scaling <b>scale-mode</b> descriptor attribute (VEC16-UE4M3 for NVFP4, VEC32-UE8M0
///         for MXFP4). The attribute IDs were intentionally not hard-coded here to avoid shipping a
///         wrong/colliding enum value; set them once verified against the installed CUDA headers.</item>
///   <item>The exact block-scale tensor layout cuBLASLt expects via the A/B scale pointers.</item>
/// </list></summary>
public sealed unsafe class Fp4GemmExecutor : IDisposable
{
    private nint _ltHandle;
    private ulong _workspace;
    private readonly nuint _workspaceBytes;
    private int _disposed;

    /// <summary>Whether the running GPU supports native FP4 GEMM (SM 10.0+, Blackwell).</summary>
    public bool IsSupported { get; }

    /// <summary>Compute capability detected at construction.</summary>
    public int SmMajor { get; }

    /// <summary>Compute capability detected at construction.</summary>
    public int SmMinor { get; }

    /// <summary>Initializes the executor. Allocates the cuBLASLt handle + workspace if SM ≥ 10.0;
    /// otherwise leaves itself unsupported without allocating any GPU resources.</summary>
    public Fp4GemmExecutor(int smMajor, int smMinor)
    {
        SmMajor = smMajor;
        SmMinor = smMinor;
        IsSupported = smMajor >= 10;
        if (!IsSupported)
        {
            _workspaceBytes = 0;
            return;
        }

        CublasLtApi.cublasLtCreate(out _ltHandle).ThrowOnError();
        _workspaceBytes = (nuint)CublasLtApi.DefaultWorkspaceBytes;
        _workspace = CudaMemory.AllocatePersistent(_workspaceBytes);
    }

    /// <summary>Runs an FP4 Linear GEMM matching <see cref="CudaBackend"/>'s row-major convention:
    /// <c>output[M, N] = input[M, K] · weight^T[N, K]</c>. Weight and input are e2m1 (2 elements/byte);
    /// <paramref name="weightBlockScale"/> / <paramref name="inputBlockScale"/> are the device pointers to
    /// their microscaling block-scale tensors.
    ///
    /// <para><b>Gated.</b> Throws on non-Blackwell hardware — callers must check <see cref="IsSupported"/>.
    /// The block-scaling scale-mode attribute is not set here (see type remarks); until that is wired and
    /// validated this method must not be relied upon for correctness.</para></summary>
    public void Run(ulong weight, ulong weightBlockScale, ulong input, ulong inputBlockScale, ulong outPtr,
        int m, int n, int k, nint stream)
    {
        if (!IsSupported)
        {
            throw new InvalidOperationException(
                $"Fp4GemmExecutor.Run called on unsupported hardware (SM {SmMajor}.{SmMinor}). Native FP4 GEMM needs Blackwell (SM 10.0+). Caller must check IsSupported and fall back to dequant→F16.");
        }
        ThrowIfDisposed();

        nint matmulDesc = 0, layoutA = 0, layoutB = 0, layoutC = 0;
        try
        {
            CublasLtApi.cublasLtMatmulDescCreate(
                out matmulDesc,
                CublasApi.CUBLAS_COMPUTE_32F,
                CublasApi.CUDA_R_32F).ThrowOnError();

            int transA = CublasApi.CUBLAS_OP_T;
            int transB = CublasApi.CUBLAS_OP_N;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int)).ThrowOnError();
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int)).ThrowOnError();

            // Microscaling block-scale pointers (one positive scale per FP4 block).
            ulong wScale = weightBlockScale;
            ulong iScale = inputBlockScale;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wScale, (nuint)sizeof(ulong)).ThrowOnError();
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &iScale, (nuint)sizeof(ulong)).ThrowOnError();

            // weight: [N, K] fp4 transposed → operand A. input: [M, K] fp4 → operand B. Output C: [M, N] f16.
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutA, CublasApi.CUDA_R_4F_E2M1, (ulong)k, (ulong)n, k).ThrowOnError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutB, CublasApi.CUDA_R_4F_E2M1, (ulong)k, (ulong)m, k).ThrowOnError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutC, CublasApi.CUDA_R_16F, (ulong)n, (ulong)m, n).ThrowOnError();

            float alpha = 1.0f, beta = 0.0f;
            CublasLtApi.cublasLtMatmul(
                _ltHandle, matmulDesc,
                &alpha,
                weight, layoutA,
                input, layoutB,
                &beta,
                outPtr, layoutC,
                outPtr, layoutC,
                0, (nint)_workspace, _workspaceBytes, stream).ThrowOnError();
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
            throw new ObjectDisposedException(nameof(Fp4GemmExecutor));
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
}
