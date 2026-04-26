using System.Runtime.InteropServices;

namespace SharpInference.Cuda;

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

    [LibraryImport(LibName, EntryPoint = "cublasGetStream_v2")]
    internal static partial int cublasGetStream(nint handle, out nint stream);

    // ── FP32 GEMM ───────────────────────────────────────────────────────

    [LibraryImport(LibName, EntryPoint = "cublasSgemm_v2")]
    internal static unsafe partial int cublasSgemm(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        float* alpha,
        ulong A, int lda,
        ulong B, int ldb,
        float* beta,
        ulong C, int ldc);

    // ── FP16 GEMM ───────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static unsafe partial int cublasHgemm(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        ushort* alpha,
        ulong A, int lda,
        ulong B, int ldb,
        ushort* beta,
        ulong C, int ldc);

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

    // ── Compute Type Constants ──────────────────────────────────────────

    internal const int CUBLAS_COMPUTE_16F = 64;
    internal const int CUBLAS_COMPUTE_32F = 68;
    internal const int CUBLAS_COMPUTE_32F_FAST_16F = 74;
    internal const int CUBLAS_COMPUTE_32F_FAST_16BF = 75;
    internal const int CUBLAS_COMPUTE_32F_FAST_TF32 = 77;

    // ── Algorithm Constants ─────────────────────────────────────────────

    internal const int CUBLAS_GEMM_DEFAULT = -1;
    internal const int CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99;
}
