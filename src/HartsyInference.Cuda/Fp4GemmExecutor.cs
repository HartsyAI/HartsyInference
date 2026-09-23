using System.Runtime.CompilerServices;

namespace HartsyInference.Cuda;

/// <summary>Native FP4 (e2m1) GEMM via cublasLtMatmul, gated on Blackwell (SM 10.0+) GPUs. Mirrors <see cref="Fp8GemmExecutor"/>: a per-call descriptor + layout set is created and torn down inside <see cref="Run"/>; the handle and workspace are owned here and reused across calls. <para><b>Hardware requirement.</b> Tensor-core FP4 paths exist only on Blackwell (SM 10.0 for datacenter B100/B200, SM 12.0 for consumer RTX 50xx). On everything earlier the constructor reports <see cref="IsSupported"/> = false and callers must fall back to dequant→F16 GEMM (the same strategy the engine uses for MXFP4/NVFP4/NF4 today).</para> <para><b>Implemented against the headers, never executed.</b> Tensor-core FP4 needs Blackwell and the hardware here is Ada/Ampere, so no run has ever reached this code. The scale-mode attribute IDs and the block-scale element types are taken from <c>cublasLt.h</c> / <c>library_types.h</c> (CUDA 13.6) rather than guessed, and the block-scale layout question is settled: cuBLASLt wants its own blocked layout, which is exactly what ComfyUI checkpoints already store, so a checkpoint's scale tensor is passed through unaltered. What remains unverified is whether a real Blackwell GEMM accepts this descriptor set and returns correct numbers — treat the first run on such a card as bring-up, not regression.</para></summary>
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

    /// <summary>Initializes the executor. Allocates the cuBLASLt handle + workspace if SM ≥ 10.0; otherwise leaves itself unsupported without allocating any GPU resources.</summary>
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

        CublasLtApi.cublasLtCreate(out _ltHandle).ThrowOnCublasError();
        _workspaceBytes = (nuint)CublasLtApi.DefaultWorkspaceBytes;
        _workspace = CudaMemory.AllocatePersistent(_workspaceBytes);
    }

    /// <summary>Runs an FP4 Linear GEMM matching <see cref="CudaBackend"/>'s row-major convention: <c>output[M, N] = input[M, K] · weight^T[N, K]</c>. Weight and input are e2m1 (2 elements/byte); <paramref name="weightBlockScale"/> / <paramref name="inputBlockScale"/> are the device pointers to their microscaling block-scale tensors. <para><b>Gated.</b> Throws on non-Blackwell hardware — callers must check <see cref="IsSupported"/>. <paramref name="format"/> selects the block-scaling mode, which must accompany the scale pointers: without it cuBLASLt reads each pointer as one per-tensor F32 scalar.</para></summary>
    public void Run(ulong weight, ulong weightBlockScale, ulong input, ulong inputBlockScale, ulong outPtr,
        int m, int n, int k, nint stream, Fp4BlockScaleFormat format = Fp4BlockScaleFormat.Nvfp4)
    {
        if (!IsSupported)
        {
            throw new InvalidOperationException(
                $"Fp4GemmExecutor.Run called on unsupported hardware (SM {SmMajor}.{SmMinor}). Native FP4 GEMM " +
                "needs Blackwell (SM 10.0+). Caller must check IsSupported and fall back to dequant→F16.");
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

            // Microscaling block-scale pointers (one positive scale per FP4 block).
            ulong wScale = weightBlockScale;
            ulong iScale = inputBlockScale;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wScale, (nuint)sizeof(ulong)).ThrowOnCublasError();
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &iScale, (nuint)sizeof(ulong)).ThrowOnCublasError();

            // The mode has to accompany the pointers: left unset, cuBLASLt reads each as one F32 scalar for the whole
            // matrix (the fp8 meaning) instead of a per-block tensor, which misreads the scales rather than failing.
            int scaleMode = format == Fp4BlockScaleFormat.Nvfp4
                ? CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
                : CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(int)).ThrowOnCublasError();
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                matmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(int)).ThrowOnCublasError();

            // weight: [N, K] fp4 transposed → operand A. input: [M, K] fp4 → operand B. Output C: [M, N] f16.
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutA, CublasApi.CUDA_R_4F_E2M1, (ulong)k, (ulong)n, k).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutB, CublasApi.CUDA_R_4F_E2M1, (ulong)k, (ulong)m, k).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(out layoutC, CublasApi.CUDA_R_16F, (ulong)n, (ulong)m, n).ThrowOnCublasError();

            float alpha = 1.0f, beta = 0.0f;
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

    ~Fp4GemmExecutor()
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
