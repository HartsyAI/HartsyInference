using System.Runtime.InteropServices;

namespace HartsyInference.Cuda;

/// <summary>P/Invoke bindings for cuBLAS. Library name "cublas" is resolved at runtime by CudaLibraryResolver
/// to cublas64_12.dll (Windows) or libcublas.so.12 (Linux).</summary>
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
    internal const int CUDA_R_4F_E2M1 = 33;  // CUDA 12.8+ (Blackwell / SM 10.0+) — native FP4 GEMM operand
    internal const int CUDA_R_8F_UE8M0 = 34; // CUDA 12.8+ — UE8M0 microscaling block-scale type
    internal const int CUDA_R_8I = 3;        // int8 GEMM operand (IMMA tensor cores on SM 7.5+)
    internal const int CUDA_R_32I = 10;      // int32 accumulate/output for int8 GEMM

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
