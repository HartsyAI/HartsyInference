using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>Block-scaled GEMM via cublasLtMatmul — NVFP4, MXFP4 and MXFP8 operands with their microscaling scale tensors — gated on Blackwell (SM 10.0 datacenter, SM 12.0 consumer). Below that <see cref="CublasLtExecutorBase.IsSupported"/> is false and callers dequantize to F16 instead. Both scale tensors pass through in cuBLASLt's blocked layout (what ComfyUI checkpoints store and what <c>block_quant.cu</c> writes), and alpha and beta are read from device memory, so the activation's per-tensor scale costs no host sync. Run on an RTX PRO 6000 and an RTX 5090 (CC 12.0); cuBLASLt 13.8 accepts m = 1.</summary>
public sealed unsafe class BlockScaledGemmExecutor : CublasLtExecutorBase
{
    /// <summary>Compute capability detected at construction.</summary>
    public int SmMajor { get; }

    /// <summary>Compute capability detected at construction.</summary>
    public int SmMinor { get; }

    public BlockScaledGemmExecutor(int smMajor, int smMinor) : base(CudaArch.Sm(smMajor, smMinor) >= CudaArch.Blackwell)
    {
        SmMajor = smMajor;
        SmMinor = smMinor;
    }

    /// <summary>Runs <c>output[M, N] = alpha · input[M, K] · weight[N, K]ᵀ</c>. Both operands are packed in <paramref name="format"/>'s element type; <paramref name="weightBlockScale"/> / <paramref name="inputBlockScale"/> are their scale tensors in cuBLASLt's blocked layout (rows padded to 128, block columns to 4). <paramref name="alphaBetaDev"/> is a device pointer to two F32s, alpha then beta: alpha carries every per-tensor factor outside the block scales, beta is 0. Throws below Blackwell — callers check <see cref="CublasLtExecutorBase.IsSupported"/>.</summary>
    public void Run(ulong weight, ulong weightBlockScale, ulong input, ulong inputBlockScale, ulong outPtr,
        int m, int n, int k, ulong alphaBetaDev, nint stream, BlockScaleFormat format, bool outF32 = false)
    {
        if (!IsSupported)
        {
            throw new InvalidOperationException(
                $"BlockScaledGemmExecutor.Run called on unsupported hardware (SM {SmMajor}.{SmMinor}). Block-scaled " +
                "GEMM needs Blackwell (SM 10.0+). Caller must check IsSupported and fall back to dequant→F16.");
        }
        ThrowIfDisposed();
        if (k % format.GroupSize() != 0)
            throw new ArgumentException($"K={k} is not a multiple of the {format} block size {format.GroupSize()}.");

        nint desc = 0, layoutA = 0, layoutB = 0, layoutC = 0;
        try
        {
            desc = CreateTnMatmulDesc(CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUDA_R_32F);
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_POINTER_MODE, CublasLtApi.CUBLASLT_POINTER_MODE_DEVICE);
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, weightBlockScale);
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, inputBlockScale);
            // The mode has to accompany the pointers: left unset, cuBLASLt reads each as one F32 scalar for the whole
            // matrix (the fp8 meaning) instead of a per-block tensor, which misreads the scales rather than failing.
            int scaleMode = format.ScaleMode();
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_MODE, scaleMode);
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_MODE, scaleMode);
            int operandType = CublasApi.DataTypeOf(format.OperandType());
            CreateTnLayouts(operandType, operandType, CublasApi.DataTypeOf(outF32 ? DType.F32 : DType.F16),
                m, n, k, out layoutA, out layoutB, out layoutC);

            Matmul(desc, (void*)alphaBetaDev, weight, layoutA, input, layoutB, (void*)(alphaBetaDev + sizeof(float)),
                outPtr, layoutC, algo: 0, stream).ThrowOnCublasError();
        }
        finally
        {
            DestroyLayouts(layoutA, layoutB, layoutC);
            DestroyDesc(desc);
        }
    }

    /// <summary><see cref="Run"/> for a caller with a fallback: a request cuBLASLt declines (invalid value, arch mismatch,
    /// not supported) returns that status and leaves <paramref name="outPtr"/> unwritten; 0 means the GEMM ran. An
    /// execution or allocation failure still throws — those leave nothing a fallback could recover from. Every call in
    /// <see cref="Run"/> is cuBLASLt (no algorithm query: <c>algo</c> is null, and an empty heuristic surfaces as
    /// NOT_SUPPORTED), so the matched codes are cuBLAS statuses, never driver errors.</summary>
    public int TryRun(ulong weight, ulong weightBlockScale, ulong input, ulong inputBlockScale, ulong outPtr,
        int m, int n, int k, ulong alphaBetaDev, nint stream, BlockScaleFormat format, bool outF32 = false)
    {
        try
        {
            Run(weight, weightBlockScale, input, inputBlockScale, outPtr, m, n, k, alphaBetaDev, stream, format, outF32);
            return 0;
        }
        catch (CudaException error) when (error.ErrorCode is CublasInvalidValue or CublasArchMismatch or CublasNotSupported)
        {
            return error.ErrorCode;
        }
    }

    private const int CublasInvalidValue = 7, CublasArchMismatch = 8, CublasNotSupported = 15;
}
