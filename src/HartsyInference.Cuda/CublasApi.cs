using System.Runtime.InteropServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>P/Invoke bindings for cuBLAS. Library name "cublas" is resolved at runtime by CudaLibraryResolver to cublas64_12.dll (Windows) or libcublas.so.12 (Linux).</summary>
internal static partial class CublasApi
{
    private const string LibName = "cublas";

    // ── Handle Management ───────────────────────────────────────────────

    [LibraryImport(LibName, EntryPoint = "cublasCreate_v2")]
    internal static partial int cublasCreate(out nint handle);

    [LibraryImport(LibName, EntryPoint = "cublasDestroy_v2")]
    internal static partial int cublasDestroy(nint handle);

    [LibraryImport(LibName, EntryPoint = "cublasSetStream_v2")]
    internal static partial int cublasSetStream(nint handle, nint stream);

    [LibraryImport(LibName, EntryPoint = "cublasGetVersion_v2")]
    internal static partial int cublasGetVersion(nint handle, out int version);

    // ── Mixed-Precision GEMM ────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static unsafe partial int cublasGemmEx(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        void* alpha,
        ulong A, int Atype, int lda,
        ulong B, int Btype, int ldb,
        void* beta,
        ulong C, int Ctype, int ldc,
        int computeType, int algo);

    // ── Batched GEMM ────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static unsafe partial int cublasGemmStridedBatchedEx(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        void* alpha,
        ulong A, int Atype, int lda, long strideA,
        ulong B, int Btype, int ldb, long strideB,
        void* beta,
        ulong C, int Ctype, int ldc, long strideC,
        int batchCount,
        int computeType, int algo);

    // ── Operation Constants ─────────────────────────────────────────────

    internal const int CUBLAS_OP_N = 0;
    internal const int CUBLAS_OP_T = 1;
    internal const int CUBLAS_OP_C = 2;

    // ── Data Type Constants ─────────────────────────────────────────────

    internal const int CUDA_R_32F = 0;
    internal const int CUDA_R_64F = 1;
    internal const int CUDA_R_16F = 2;
    internal const int CUDA_R_8F_E4M3 = 28;  // CUDA 11.8+ (Ada / SM 8.9+)
    internal const int CUDA_R_8F_E5M2 = 29;  // CUDA 11.8+ (Ada / SM 8.9+)
    internal const int CUDA_R_16BF = 14;
    // Values are cudaDataType, read from library_types.h (CUDA 13.6). 34 is past the end of that enum, which is
    // what CUDA_R_8F_UE8M0 was set to — nothing had caught it because no FP4 path was wired up to use it.
    internal const int CUDA_R_4F_E2M1 = 33;  // Blackwell (SM 10.0 / 12.0) native FP4 GEMM operand
    internal const int CUDA_R_6F_E2M3 = 31;  // Blackwell FP6; no codec or loader mapping in the engine yet
    internal const int CUDA_R_6F_E3M2 = 32;  // Blackwell FP6; likewise
    internal const int CUDA_R_8F_UE4M3 = 28; // NVFP4's block-scale type — an alias of E4M3, not a distinct value
    internal const int CUDA_R_8F_UE8M0 = 30; // MXFP4/MXFP8's exponent-only block-scale type
    internal const int CUDA_R_8I = 3;        // int8 GEMM operand (IMMA tensor cores on SM 7.5+)
    internal const int CUDA_R_32I = 10;      // int32 accumulate/output for int8 GEMM

    /// <summary>The cudaDataType for a tensor dtype — the one map every cuBLAS/cuBLASLt layout is created through, so an operand's element type is never restated by hand at a call site.</summary>
    internal static int DataTypeOf(DType dtype)
    {
        if (dtype == DType.F32) return CUDA_R_32F;
        if (dtype == DType.F16) return CUDA_R_16F;
        if (dtype == DType.BF16) return CUDA_R_16BF;
        if (dtype == DType.F8E4M3) return CUDA_R_8F_E4M3;
        if (dtype == DType.F8E5M2) return CUDA_R_8F_E5M2;
        if (dtype == DType.F4E2M1) return CUDA_R_4F_E2M1;
        if (dtype == DType.I8) return CUDA_R_8I;
        if (dtype == DType.I32) return CUDA_R_32I;
        throw new NotSupportedException($"cuBLAS has no data type for {dtype}.");
    }

    // ── Compute Type Constants ──────────────────────────────────────────

    internal const int CUBLAS_COMPUTE_16F = 64;
    internal const int CUBLAS_COMPUTE_32F = 68;
    internal const int CUBLAS_COMPUTE_32F_FAST_16F = 74;
    internal const int CUBLAS_COMPUTE_32F_FAST_16BF = 75;
    internal const int CUBLAS_COMPUTE_32F_FAST_TF32 = 77;
    internal const int CUBLAS_COMPUTE_32I = 72;          // int8 operands, int32 accumulate (IMMA)

    // ── Algorithm Constants ─────────────────────────────────────────────

    internal const int CUBLAS_GEMM_DEFAULT = -1;
    internal const int CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99;
}
