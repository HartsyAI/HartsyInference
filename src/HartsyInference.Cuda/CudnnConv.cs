using System.Collections.Concurrent;
using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>
/// Convolution forward via cuDNN's backend graph API. Replaces the im2col→cuBLAS GEMM path for
/// F16/BF16 NCHW convolutions: cuDNN's heuristics pick tensor-core implicit-GEMM/Winograd engines that
/// never materialize the im2col matrix (an extra kH·kW-times-input-sized HBM write+read per conv —
/// the dominant conv cost in the SDXL UNet, which runs ~50 convolutions per step).
///
/// Graph = the single CONVOLUTION_FORWARD op (X ⊛ W → Y, cross-correlation, fp32 accumulate), NCHW
/// strided tensors, alpha 1 / beta 0. Bias stays a separate kernel in the caller — same numerics as the
/// GEMM path's bias add. Execution plans (heuristics + JIT) are cached by shape+dtype; per-plan device
/// workspace is allocated once and reused. Instances are per <see cref="CudaBackend"/> (one cuDNN handle
/// bound to the compute stream). Any failure is caught by the caller, which self-disables the route for
/// the session and falls back to im2col — a wrong shape costs one warning, never a session kill.
/// </summary>
internal sealed class CudnnConv : IDisposable
{
    private readonly nint _handle;
    private readonly ConcurrentDictionary<string, Plan> _plans = new();
    private bool _disposed;

    private sealed class Plan
    {
        public nint Execution;
        public ulong Workspace;
        public long WorkspaceBytes;
    }

    public CudnnConv(nint stream)
    {
        int st = cudnnCreate(out _handle);
        if (st != CUDNN_STATUS_SUCCESS)
            throw new InvalidOperationException($"cudnnCreate failed: {ErrorString(st)}");
        cudnnSetStream(_handle, stream);
    }

    private const long UidX = 1, UidW = 2, UidY = 3;

    /// <summary>Runs X[n,c,h,w] ⊛ W[k,c,r,s] → Y[n,k,outH,outW]. All pointers are device buffers of
    /// <paramref name="dataType"/> (CUDNN_DATA_HALF / CUDNN_DATA_BFLOAT16), contiguous NCHW.</summary>
    public unsafe void Execute(ulong x, ulong w, ulong y,
        long n, long c, long h, long wIn, long k, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padW, int dataType)
    {
        string key = $"{n},{c},{h},{wIn}|{k},{r},{s}|{strideH},{strideW},{padH},{padW}|{dataType}";
        Plan plan = _plans.GetOrAdd(key, _ => BuildPlan(n, c, h, wIn, k, r, s, outH, outW, strideH, strideW, padH, padW, dataType));

        nint vp = 0;
        try
        {
            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, out vp), "variant pack create");
            long* uids = stackalloc long[3] { UidX, UidW, UidY };
            void** ptrs = stackalloc void*[3] { (void*)x, (void*)w, (void*)y };
            void* ws = (void*)plan.Workspace;
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws);
            Check(cudnnBackendFinalize(vp), "variant pack finalize");
            Check(cudnnBackendExecute(_handle, plan.Execution, vp), "cudnnBackendExecute");
        }
        finally
        {
            if (vp != 0) cudnnBackendDestroyDescriptor(vp);
        }
    }

    private unsafe Plan BuildPlan(long n, long c, long h, long wIn, long k, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padW, int dataType)
    {
        List<nint> owned = new();
        try
        {
            long* xDim = stackalloc long[4] { n, c, h, wIn };
            long* xStr = stackalloc long[4] { c * h * wIn, h * wIn, wIn, 1 };
            nint tX = Tensor(owned, UidX, xDim, xStr, dataType);
            long* wDim = stackalloc long[4] { k, c, r, s };
            long* wStr = stackalloc long[4] { c * r * s, r * s, s, 1 };
            nint tW = Tensor(owned, UidW, wDim, wStr, dataType);
            long* yDim = stackalloc long[4] { n, k, outH, outW };
            long* yStr = stackalloc long[4] { k * outH * outW, outH * outW, outW, 1 };
            nint tY = Tensor(owned, UidY, yDim, yStr, dataType);

            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, out nint conv), "conv desc create");
            owned.Add(conv);
            long spatial = 2;
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
            int comp = CUDNN_DATA_FLOAT;
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp);
            int mode = CUDNN_CROSS_CORRELATION;
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
            long* dil = stackalloc long[2] { 1, 1 };
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dil);
            long* strides = stackalloc long[2] { strideH, strideW };
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, strides);
            long* pads = stackalloc long[2] { padH, padW };
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
            Check(cudnnBackendFinalize(conv), "conv desc finalize");

            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, out nint op), "conv op create");
            owned.Add(op);
            float alpha = 1.0f, beta = 0.0f;
            SetAttr(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
            SetAttr(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
            void* cp = (void*)conv, xp = (void*)tX, wp = (void*)tW, yp = (void*)tY;
            SetAttr(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cp);
            SetAttr(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xp);
            SetAttr(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wp);
            SetAttr(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yp);
            Check(cudnnBackendFinalize(op), "conv op finalize");

            nint graph;
            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, out graph), "graph create");
            owned.Add(graph);
            void* hp = (void*)_handle;
            nint* ops = stackalloc nint[1] { op };
            SetAttr(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &hp);
            SetAttr(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops);
            Check(cudnnBackendFinalize(graph), "graph finalize");

            (nint exec, long wsBytes) = BuildExecutionPlan(graph, owned);
            return new Plan
            {
                Execution = exec,
                WorkspaceBytes = wsBytes,
                Workspace = wsBytes > 0 ? CudaMemory.Allocate((nuint)wsBytes) : 0,
            };
        }
        finally
        {
            foreach (nint d in owned)
                cudnnBackendDestroyDescriptor(d);
        }
    }

    private unsafe (nint exec, long wsBytes) BuildExecutionPlan(nint graph, List<nint> owned)
    {
        foreach (int mode in new[] { CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK })
        {
            nint heur;
            if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, out heur) != CUDNN_STATUS_SUCCESS)
                continue;
            owned.Add(heur);
            void* gp = (void*)graph;
            SetAttr(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &gp);
            int m = mode;
            SetAttr(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &m);
            if (cudnnBackendFinalize(heur) != CUDNN_STATUS_SUCCESS)
                continue;

            const int maxCfgs = 32;
            nint[] cfgs = new nint[maxCfgs];
            for (int i = 0; i < maxCfgs; i++)
                cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, out cfgs[i]);
            long returned;
            fixed (nint* cfgPtr = cfgs)
            {
                int gst = cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR, maxCfgs, out returned, cfgPtr);
                if (gst != CUDNN_STATUS_SUCCESS) returned = 0;
            }
            for (int i = 0; i < maxCfgs; i++)
            {
                if (i < returned)
                {
                    (nint exec, long ws, bool ok) = TryPlan(cfgs[i]);
                    if (ok)
                    {
                        for (int j = 0; j < maxCfgs; j++) cudnnBackendDestroyDescriptor(cfgs[j]);
                        return (exec, ws);
                    }
                }
                cudnnBackendDestroyDescriptor(cfgs[i]);
            }
        }
        throw new InvalidOperationException("cuDNN conv: no engine config produced a valid execution plan");
    }

    private unsafe (nint exec, long ws, bool ok) TryPlan(nint cfg)
    {
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, out nint p) != CUDNN_STATUS_SUCCESS)
            return (0, 0, false);
        void* hp = (void*)_handle;
        void* cp = (void*)cfg;
        SetAttr(p, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &hp);
        SetAttr(p, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cp);
        if (cudnnBackendFinalize(p) != CUDNN_STATUS_SUCCESS)
        {
            cudnnBackendDestroyDescriptor(p);
            return (0, 0, false);
        }
        long ws = 0;
        cudnnBackendGetAttribute(p, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, out _, &ws);
        return (p, ws, true);
    }

    private unsafe nint Tensor(List<nint> owned, long uid, long* dims, long* strides, int dtype)
    {
        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, out nint t), "tensor create");
        owned.Add(t);
        int dt = dtype;
        SetAttr(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
        SetAttr(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims);
        SetAttr(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides);
        long id = uid;
        SetAttr(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id);
        long align = 16;
        SetAttr(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
        byte v = 0;
        SetAttr(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &v);
        Check(cudnnBackendFinalize(t), "tensor finalize");
        return t;
    }

    private static unsafe void SetAttr(nint desc, int attr, int type, long count, void* vals)
    {
        int st = cudnnBackendSetAttribute(desc, attr, type, count, vals);
        if (st != CUDNN_STATUS_SUCCESS)
            throw new InvalidOperationException($"cudnnBackendSetAttribute(attr={attr}) failed: {ErrorString(st)}");
    }

    private static void Check(int st, string what)
    {
        if (st != CUDNN_STATUS_SUCCESS)
            throw new InvalidOperationException($"cuDNN {what} failed: {ErrorString(st)}");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Plan p in _plans.Values)
        {
            if (p.Execution != 0) cudnnBackendDestroyDescriptor(p.Execution);
            if (p.Workspace != 0) CudaMemory.Free(p.Workspace);
        }
        _plans.Clear();
        if (_handle != 0) cudnnDestroy(_handle);
    }
}
