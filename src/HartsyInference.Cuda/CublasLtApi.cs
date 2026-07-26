using System.Runtime.InteropServices;

namespace HartsyInference.Cuda;

/// <summary>P/Invoke bindings for cuBLASLt — the lightweight cuBLAS API that supports FP8 GEMM with per-tensor scale
/// factors. Only usable on Ada (SM 8.9+) and Hopper (SM 9.0+) GPUs. On Ampere and below, the matrix-mul path falls
/// back to FP16 cublasGemmEx via <see cref="CudaBackend"/>'s cast-then-GEMM logic.</summary>
internal static partial class CublasLtApi
{
    private const string LibName = "cublasLt";

    // ── Handle Management ───────────────────────────────────────────────

    [LibraryImport(LibName, EntryPoint = "cublasLtCreate")]
    internal static partial int cublasLtCreate(out nint handle);

    [LibraryImport(LibName, EntryPoint = "cublasLtDestroy")]
    internal static partial int cublasLtDestroy(nint handle);

    // ── Matmul Descriptor ───────────────────────────────────────────────

    [LibraryImport(LibName, EntryPoint = "cublasLtMatmulDescCreate")]
    internal static partial int cublasLtMatmulDescCreate(out nint matmulDesc, int computeType, int scaleType);

    [LibraryImport(LibName, EntryPoint = "cublasLtMatmulDescDestroy")]
    internal static partial int cublasLtMatmulDescDestroy(nint matmulDesc);

    [LibraryImport(LibName, EntryPoint = "cublasLtMatmulDescSetAttribute")]
    internal static unsafe partial int cublasLtMatmulDescSetAttribute(
        nint matmulDesc, int attr, void* buf, nuint sizeInBytes);

    // ── Matrix Layout ───────────────────────────────────────────────────

    [LibraryImport(LibName, EntryPoint = "cublasLtMatrixLayoutCreate")]
    internal static partial int cublasLtMatrixLayoutCreate(out nint matLayout, int type, ulong rows, ulong cols, long ld);

    [LibraryImport(LibName, EntryPoint = "cublasLtMatrixLayoutDestroy")]
    internal static partial int cublasLtMatrixLayoutDestroy(nint matLayout);

    [LibraryImport(LibName, EntryPoint = "cublasLtMatrixLayoutSetAttribute")]
    internal static unsafe partial int cublasLtMatrixLayoutSetAttribute(
        nint matLayout, int attr, void* buf, nuint sizeInBytes);

    // ── Matmul Preference (algorithm heuristic) ─────────────────────────

    [LibraryImport(LibName, EntryPoint = "cublasLtMatmulPreferenceCreate")]
    internal static partial int cublasLtMatmulPreferenceCreate(out nint pref);

    [LibraryImport(LibName, EntryPoint = "cublasLtMatmulPreferenceDestroy")]
    internal static partial int cublasLtMatmulPreferenceDestroy(nint pref);

    [LibraryImport(LibName, EntryPoint = "cublasLtMatmulPreferenceSetAttribute")]
    internal static unsafe partial int cublasLtMatmulPreferenceSetAttribute(
        nint pref, int attr, void* buf, nuint sizeInBytes);

    [LibraryImport(LibName, EntryPoint = "cublasLtMatmulAlgoGetHeuristic")]
    internal static unsafe partial int cublasLtMatmulAlgoGetHeuristic(
        nint lightHandle, nint computeDesc,
        nint Adesc, nint Bdesc, nint Cdesc, nint Ddesc,
        nint preference, int requestedAlgoCount,
        void* heuristicResultsArray, out int returnAlgoCount);

    // ── Matmul Dispatch ─────────────────────────────────────────────────

    [LibraryImport(LibName, EntryPoint = "cublasLtMatmul")]
    internal static unsafe partial int cublasLtMatmul(
        nint lightHandle, nint computeDesc,
        void* alpha,
        ulong A, nint Adesc,
        ulong B, nint Bdesc,
        void* beta,
        ulong C, nint Cdesc,
        ulong D, nint Ddesc,
        nint algo, nint workspace, nuint workspaceSizeInBytes, nint stream);

    // ── Matmul Descriptor Attributes ────────────────────────────────────

    internal const int CUBLASLT_MATMUL_DESC_TRANSA = 3;
    internal const int CUBLASLT_MATMUL_DESC_TRANSB = 4;
    internal const int CUBLASLT_MATMUL_DESC_EPILOGUE = 7;
    internal const int CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8;
    internal const int CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 32;
    internal const int CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17;
    internal const int CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18;
    internal const int CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19;
    internal const int CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20;
    internal const int CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21;
    internal const int CUBLASLT_MATMUL_DESC_FAST_ACCUM = 28;

    // ── Matrix Layout Attributes ────────────────────────────────────────

    internal const int CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5;
    internal const int CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6;

    // ── Preference Attributes ───────────────────────────────────────────

    internal const int CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1;

    // ── Epilogue values (cublasLtEpilogue_t) ────────────────────────────
    // Fuse bias and/or activation into the GEMM epilogue, removing a separate
    // elementwise kernel launch plus an activation-sized HBM round-trip per GEMM.
    // BIAS is bit-equivalent to a separate per-output-channel bias add; RELU/GELU
    // are the cuBLASLt fused activations (GELU is the erf-exact form), only safe to
    // request when they match the model's activation.

    internal const int CUBLASLT_EPILOGUE_DEFAULT = 1;
    internal const int CUBLASLT_EPILOGUE_RELU = 2;
    internal const int CUBLASLT_EPILOGUE_BIAS = 4;
    internal const int CUBLASLT_EPILOGUE_RELU_BIAS = 6;
    internal const int CUBLASLT_EPILOGUE_GELU = 32;
    internal const int CUBLASLT_EPILOGUE_GELU_BIAS = 36;

    // ── Default workspace size (cuBLASLt minimum guidance: 32 MiB Hopper, 4 MiB Ada; a larger pool lets the
    // internal heuristic pick split-K / larger-tile algos for the big M×K fp8 DiT shapes) ───────

    internal const ulong DefaultWorkspaceBytes = 64UL * 1024 * 1024;
}
