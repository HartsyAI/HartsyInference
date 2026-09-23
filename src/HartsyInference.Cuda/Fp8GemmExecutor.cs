using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>Native FP8 GEMM via cublasLtMatmul, gated on Ada+ (SM 8.9+). The per-tensor weight scale is folded into alpha — exact for the single scalar <c>Fp8ScaleFactor</c> every fp8_scaled checkpoint carries — and the activation's per-tensor dequant scale is read from device memory through B_SCALE_POINTER, so dynamic activation quantization needs no host sync. Below Ada <see cref="CublasLtExecutorBase.IsSupported"/> is false and callers take the cast-then-F16-GEMM path.</summary>
public sealed unsafe class Fp8GemmExecutor : CublasLtExecutorBase
{
    /// <summary>Compute capability detected at construction.</summary>
    public int SmMajor { get; }

    /// <summary>Compute capability detected at construction.</summary>
    public int SmMinor { get; }

    public Fp8GemmExecutor(int smMajor, int smMinor) : base(CudaArch.Sm(smMajor, smMinor) >= CudaArch.Ada)
    {
        SmMajor = smMajor;
        SmMinor = smMinor;
    }

    /// <summary>Runs <c>output[M, N] = input[M, K] · weight[N, K]ᵀ</c> on fp8 operands, row-major device pointers. <paramref name="weightType"/> and <paramref name="inputType"/> name each operand's fp8 flavour (default E4M3); cuBLASLt takes every pairing except E5M2 × E5M2, refused here instead of surfacing as a status code. <paramref name="inputScaleDev"/> is a device pointer to the activation's dequant scale (<c>amax/448</c>), 0 = unscaled.</summary>
    public void Run(ulong weight, ulong input, ulong outPtr, int m, int n, int k, float weightScale, nint stream,
        ulong inputScaleDev = 0, bool outF32 = false, DType? weightType = null, DType? inputType = null)
    {
        DType wType = weightType ?? DType.F8E4M3;
        DType iType = inputType ?? DType.F8E4M3;
        if (!wType.IsFp8 || !iType.IsFp8)
            throw new ArgumentException($"fp8 GEMM operands must be fp8; got weight {wType}, input {iType}.");
        if (wType == DType.F8E5M2 && iType == DType.F8E5M2)
            throw new ArgumentException("cuBLASLt has no E5M2 × E5M2 fp8 GEMM; one operand must be E4M3.");
        if (!IsSupported)
        {
            throw new InvalidOperationException(
                $"Fp8GemmExecutor.Run called on unsupported hardware (SM {SmMajor}.{SmMinor}). Caller must check IsSupported first.");
        }
        ThrowIfDisposed();

        nint desc = 0, layoutA = 0, layoutB = 0, layoutC = 0;
        try
        {
            desc = CreateTnMatmulDesc(CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUDA_R_32F);
            if (inputScaleDev != 0)
                SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, inputScaleDev);
            CreateTnLayouts(CublasApi.DataTypeOf(wType), CublasApi.DataTypeOf(iType),
                CublasApi.DataTypeOf(outF32 ? DType.F32 : DType.F16), m, n, k, out layoutA, out layoutB, out layoutC);

            float alpha = weightScale, beta = 0.0f;
            Matmul(desc, &alpha, weight, layoutA, input, layoutB, &beta, outPtr, layoutC, algo: 0, stream).ThrowOnCublasError();
        }
        finally
        {
            DestroyLayouts(layoutA, layoutB, layoutC);
            DestroyDesc(desc);
        }
    }
}
