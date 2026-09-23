using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>Native FP4 (e2m1) GEMM via cublasLtMatmul, gated on Blackwell (SM 10.0 datacenter, SM 12.0 consumer). Below that <see cref="CublasLtExecutorBase.IsSupported"/> is false and callers fall back to dequant→F16 GEMM. <para><b>Implemented against the headers, never executed.</b> No Blackwell card has run this code. The scale-mode attribute IDs and block-scale element types are taken from <c>cublasLt.h</c> / <c>library_types.h</c> (CUDA 13.6), and the block-scale layout question is settled: cuBLASLt wants its own blocked layout, which is what ComfyUI checkpoints store, so a checkpoint's scale tensor passes through unaltered. What remains unverified is whether a real Blackwell GEMM accepts this descriptor set and returns correct numbers — treat the first run on such a card as bring-up, not regression.</para></summary>
public sealed unsafe class Fp4GemmExecutor : CublasLtExecutorBase
{
    /// <summary>Compute capability detected at construction.</summary>
    public int SmMajor { get; }

    /// <summary>Compute capability detected at construction.</summary>
    public int SmMinor { get; }

    public Fp4GemmExecutor(int smMajor, int smMinor) : base(CudaArch.Sm(smMajor, smMinor) >= CudaArch.Blackwell)
    {
        SmMajor = smMajor;
        SmMinor = smMinor;
    }

    /// <summary>Runs <c>output[M, N] = input[M, K] · weight[N, K]ᵀ</c>. Weight and input are e2m1 (2 elements/byte); <paramref name="weightBlockScale"/> / <paramref name="inputBlockScale"/> are device pointers to their microscaling block-scale tensors, and <paramref name="format"/> names the block-scaling mode they use. Throws on non-Blackwell hardware — callers check <see cref="CublasLtExecutorBase.IsSupported"/>.</summary>
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

        nint desc = 0, layoutA = 0, layoutB = 0, layoutC = 0;
        try
        {
            desc = CreateTnMatmulDesc(CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUDA_R_32F);
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, weightBlockScale);
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, inputBlockScale);
            // The mode has to accompany the pointers: left unset, cuBLASLt reads each as one F32 scalar for the whole
            // matrix (the fp8 meaning) instead of a per-block tensor, which misreads the scales rather than failing.
            int scaleMode = format == Fp4BlockScaleFormat.Nvfp4
                ? CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
                : CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_MODE, scaleMode);
            SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_MODE, scaleMode);
            CreateTnLayouts(CublasApi.DataTypeOf(DType.F4E2M1), CublasApi.DataTypeOf(DType.F4E2M1),
                CublasApi.DataTypeOf(DType.F16), m, n, k, out layoutA, out layoutB, out layoutC);

            float alpha = 1.0f, beta = 0.0f;
            Matmul(desc, &alpha, weight, layoutA, input, layoutB, &beta, outPtr, layoutC, algo: 0, stream).ThrowOnCublasError();
        }
        finally
        {
            DestroyLayouts(layoutA, layoutB, layoutC);
            DestroyDesc(desc);
        }
    }
}
