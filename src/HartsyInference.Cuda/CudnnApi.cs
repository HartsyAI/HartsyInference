using System.Runtime.InteropServices;

namespace HartsyInference.Cuda;

/// <summary>
/// P/Invoke bindings for the cuDNN backend graph API (cuDNN 9). Library name "cudnn" is resolved at
/// runtime by <see cref="CudaLibraryResolver"/> to libcudnn.so.9 (Linux) / cudnn64_9.dll (Windows).
/// Only the handle + generic backend-descriptor entry points are needed — the fused scaled-dot-product
/// attention graph is assembled from these primitives in <see cref="CudnnSdpa"/>.
/// </summary>
internal static partial class CudnnApi
{
    private const string LibName = "cudnn";

    // ── Handle / version ────────────────────────────────────────────────
    [LibraryImport(LibName)] internal static partial int cudnnCreate(out nint handle);
    [LibraryImport(LibName)] internal static partial int cudnnDestroy(nint handle);
    [LibraryImport(LibName)] internal static partial int cudnnSetStream(nint handle, nint streamId);
    [LibraryImport(LibName)] internal static partial nuint cudnnGetVersion();
    /// <summary>The CUDA Runtime version cuDNN was BUILT against (e.g. 12080 → CUDA 12.8). Compile-time constant,
    /// needs no handle/context — safe to call right after the library loads to gate on a CUDA-major mismatch.</summary>
    [LibraryImport(LibName)] internal static partial nuint cudnnGetCudartVersion();
    [LibraryImport(LibName)] internal static partial nint cudnnGetErrorString(int status);

    // ── Backend descriptor graph API ────────────────────────────────────
    [LibraryImport(LibName)] internal static partial int cudnnBackendCreateDescriptor(int descriptorType, out nint descriptor);
    [LibraryImport(LibName)] internal static partial int cudnnBackendDestroyDescriptor(nint descriptor);
    [LibraryImport(LibName)] internal static partial int cudnnBackendFinalize(nint descriptor);

    [LibraryImport(LibName)]
    internal static unsafe partial int cudnnBackendSetAttribute(nint descriptor, int attributeName, int attributeType, long elementCount, void* arrayOfElements);

    [LibraryImport(LibName)]
    internal static unsafe partial int cudnnBackendGetAttribute(nint descriptor, int attributeName, int attributeType, long requestedElementCount, out long elementCount, void* arrayOfElements);

    [LibraryImport(LibName)]
    internal static partial int cudnnBackendExecute(nint handle, nint executionPlan, nint variantPack);

    internal static string ErrorString(int status)
    {
        nint p = cudnnGetErrorString(status);
        return p == 0 ? $"cudnn status {status}" : (Marshal.PtrToStringAnsi(p) ?? $"cudnn status {status}");
    }

    // ── Backend enum constants (ABI-stable across cuDNN 9.x; values from cudnn_graph_v9.h 9.24) ──
    internal const int CUDNN_STATUS_SUCCESS = 0;

    // cudnnBackendDescriptorType_t
    internal const int CUDNN_BACKEND_POINTWISE_DESCRIPTOR = 0;
    internal const int CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = 1;
    internal const int CUDNN_BACKEND_ENGINE_DESCRIPTOR = 2;
    internal const int CUDNN_BACKEND_ENGINECFG_DESCRIPTOR = 3;
    internal const int CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = 4;
    internal const int CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = 5;
    internal const int CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = 10;
    internal const int CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = 12;
    internal const int CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = 13;
    internal const int CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = 15;
    internal const int CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = 16;
    internal const int CUDNN_BACKEND_TENSOR_DESCRIPTOR = 17;
    internal const int CUDNN_BACKEND_MATMUL_DESCRIPTOR = 18;
    internal const int CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = 19;
    internal const int CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR = 45;   // cuDNN >= 9.21 unified softmax op

    // cudnnBackendAttributeType_t
    internal const int CUDNN_TYPE_HANDLE = 0;
    internal const int CUDNN_TYPE_DATA_TYPE = 1;
    internal const int CUDNN_TYPE_BOOLEAN = 2;
    internal const int CUDNN_TYPE_INT64 = 3;
    internal const int CUDNN_TYPE_FLOAT = 4;
    internal const int CUDNN_TYPE_VOID_PTR = 6;
    internal const int CUDNN_TYPE_CONVOLUTION_MODE = 7;
    internal const int CUDNN_TYPE_HEUR_MODE = 8;
    internal const int CUDNN_TYPE_POINTWISE_MODE = 14;
    internal const int CUDNN_TYPE_BACKEND_DESCRIPTOR = 15;

    // cudnnBackendAttributeName_t
    internal const int CUDNN_ATTR_POINTWISE_MODE = 0;
    internal const int CUDNN_ATTR_POINTWISE_MATH_PREC = 1;
    internal const int CUDNN_ATTR_CONVOLUTION_COMP_TYPE = 100;
    internal const int CUDNN_ATTR_CONVOLUTION_CONV_MODE = 101;
    internal const int CUDNN_ATTR_CONVOLUTION_DILATIONS = 102;
    internal const int CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = 103;
    internal const int CUDNN_ATTR_CONVOLUTION_POST_PADDINGS = 104;
    internal const int CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS = 105;
    internal const int CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS = 106;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = 700;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = 701;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = 702;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W = 703;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X = 704;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y = 705;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = 706;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = 707;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = 708;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = 709;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = 710;
    internal const int CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = 711;
    internal const int CUDNN_ATTR_ENGINEHEUR_MODE = 200;
    internal const int CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201;
    internal const int CUDNN_ATTR_ENGINEHEUR_RESULTS = 202;
    internal const int CUDNN_ATTR_ENGINECFG_ENGINE = 300;
    internal const int CUDNN_ATTR_EXECUTION_PLAN_HANDLE = 400;
    internal const int CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = 401;
    internal const int CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = 402;
    internal const int CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = 750;
    internal const int CUDNN_ATTR_OPERATION_POINTWISE_XDESC = 751;
    internal const int CUDNN_ATTR_OPERATION_POINTWISE_BDESC = 752;
    internal const int CUDNN_ATTR_OPERATION_POINTWISE_YDESC = 753;
    internal const int CUDNN_ATTR_OPERATIONGRAPH_HANDLE = 800;
    internal const int CUDNN_ATTR_OPERATIONGRAPH_OPS = 801;
    internal const int CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 802;
    internal const int CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = 900;
    internal const int CUDNN_ATTR_TENSOR_DATA_TYPE = 901;
    internal const int CUDNN_ATTR_TENSOR_DIMENSIONS = 902;
    internal const int CUDNN_ATTR_TENSOR_STRIDES = 903;
    internal const int CUDNN_ATTR_TENSOR_UNIQUE_ID = 906;
    internal const int CUDNN_ATTR_TENSOR_IS_VIRTUAL = 907;
    internal const int CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = 1000;
    internal const int CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001;
    internal const int CUDNN_ATTR_VARIANT_PACK_WORKSPACE = 1003;
    internal const int CUDNN_ATTR_ENGINE_OPERATION_GRAPH = 1300;
    internal const int CUDNN_ATTR_ENGINE_GLOBAL_INDEX = 1301;
    internal const int CUDNN_ATTR_MATMUL_COMP_TYPE = 1500;
    internal const int CUDNN_ATTR_OPERATION_MATMUL_ADESC = 1520;
    internal const int CUDNN_ATTR_OPERATION_MATMUL_BDESC = 1521;
    internal const int CUDNN_ATTR_OPERATION_MATMUL_CDESC = 1522;
    internal const int CUDNN_ATTR_OPERATION_MATMUL_DESC = 1523;
    internal const int CUDNN_ATTR_OPERATION_SOFTMAX_XDESC = 3100;
    internal const int CUDNN_ATTR_OPERATION_SOFTMAX_YDESC = 3101;

    // cudnnDataType_t
    internal const int CUDNN_DATA_FLOAT = 0;
    internal const int CUDNN_DATA_HALF = 2;
    internal const int CUDNN_DATA_BFLOAT16 = 9;

    // cudnnConvolutionMode_t
    internal const int CUDNN_CROSS_CORRELATION = 1;

    // cudnnPointwiseMode_t
    internal const int CUDNN_POINTWISE_ADD = 0;
    internal const int CUDNN_POINTWISE_MUL = 1;

    // cudnnBackendHeurMode_t
    internal const int CUDNN_HEUR_MODE_A = 3;
    internal const int CUDNN_HEUR_MODE_FALLBACK = 2;
}
