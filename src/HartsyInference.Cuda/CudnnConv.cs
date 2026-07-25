using System.Collections.Concurrent;
using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>
/// Convolution forward via cuDNN's backend graph API. Replaces the im2col→cuBLAS GEMM path for
/// F16/BF16 NCHW convolutions: cuDNN's heuristics pick tensor-core implicit-GEMM/Winograd engines that
/// never materialize the im2col matrix (an extra kH·kW-times-input-sized HBM write+read per conv —
/// the dominant conv cost in the SDXL UNet, which runs ~50 convolutions per step).
///
/// Graph = a single CONVOLUTION_FORWARD op (X ⊛ W → Y) or CONVOLUTION_BACKWARD_DATA op (transposed
/// convolution: DY ⊛ W → DX) — cross-correlation, fp32 accumulate — over NCHW
/// strided tensors, alpha 1 / beta 0. Bias stays a separate kernel in the caller — same numerics as the
/// GEMM path's bias add. Execution plans (heuristics + JIT) are cached by shape+dtype; workspace comes from
/// the stream-ordered pool per execution (capped per plan). Instances are per <see cref="CudaBackend"/> (one cuDNN handle
/// bound to the compute stream). Any failure is caught by the caller, which self-disables the route for
/// the session and falls back to im2col — a wrong shape costs one warning, never a session kill.
/// </summary>
internal sealed class CudnnConv : IDisposable
{
    // Engine configs demanding more scratch than this are skipped in favor of the next candidate — the audio
    // codecs cache a plan per (shape, dilation) pair, so an FFT-class engine with a multi-GB workspace on one
    // waveform-length conv would otherwise dominate device memory.
    private const long MaxWorkspaceBytes = 512L * 1024 * 1024;

    private readonly nint _handle;
    private readonly nint _stream;
    private readonly ConcurrentDictionary<string, Plan> _plans = new();
    private bool _disposed;

    private sealed class Plan
    {
        public nint Execution;
        public long WorkspaceBytes;
    }

    public CudnnConv(nint stream)
    {
        int st = cudnnCreate(out _handle);
        if (st != CUDNN_STATUS_SUCCESS)
            throw new InvalidOperationException($"cudnnCreate failed: {ErrorString(st)}");
        cudnnSetStream(_handle, stream);
        _stream = stream;
    }

    private const long UidX = 1, UidW = 2, UidY = 3;

    /// <summary>Runs X[n,c,h,w] ⊛ W[k,c,r,s] → Y[n,k,outH,outW]. All pointers are device buffers of
    /// <paramref name="dataType"/> (CUDNN_DATA_HALF / CUDNN_DATA_BFLOAT16), contiguous NCHW. W-padding may be
    /// asymmetric (<paramref name="padWPre"/> zeros before, <paramref name="padWPost"/> after) — the backend
    /// graph API keeps PRE/POST paddings as separate attributes, which lets causal (left-padded) 1D convs run
    /// without an explicit pad-copy.</summary>
    public unsafe void Execute(ulong x, ulong w, ulong y,
        long n, long c, long h, long wIn, long k, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padWPre, long padWPost, int dataType,
        long dilationH = 1, long dilationW = 1)
    {
        string key = $"f|{n},{c},{h},{wIn}|{k},{r},{s}|{strideH},{strideW},{padH},{padWPre},{padWPost}|{dilationH},{dilationW}|{dataType}";
        Plan plan = _plans.GetOrAdd(key, _ => BuildPlan(backwardData: false,
            n, c, h, wIn, k, r, s, outH, outW, strideH, strideW, padH, padWPre, padWPost, dataType, dilationH, dilationW));
        Run(plan, x, w, y);
    }

    /// <summary>Transposed convolution as cuDNN convolution-backward-data: DY[n,k,h,wIn] (the transpose-conv
    /// input) ⊛ W[k,c,r,s] → DX[n,c,outH,outW]. Geometry attributes describe the corresponding FORWARD conv
    /// (DX is the conv input), so the pads crop the full transposed output:
    /// outW = (wIn−1)·strideW + dilationW·(s−1) + 1 − padWPre − padWPost.</summary>
    public unsafe void ExecuteBackwardData(ulong dy, ulong w, ulong dx,
        long n, long k, long c, long h, long wIn, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padWPre, long padWPost, int dataType,
        long dilationH = 1, long dilationW = 1)
    {
        string key = $"d|{n},{c},{h},{wIn}|{k},{r},{s}|{strideH},{strideW},{padH},{padWPre},{padWPost}|{dilationH},{dilationW}|{dataType}";
        Plan plan = _plans.GetOrAdd(key, _ => BuildPlan(backwardData: true,
            n, c, outH, outW, k, r, s, h, wIn, strideH, strideW, padH, padWPre, padWPost, dataType, dilationH, dilationW));
        Run(plan, dx, w, dy);
    }

    // xPtr/yPtr are the X/Y-slot device pointers of the plan (forward: input/output; dgrad: DX/DY).
    // Workspace is taken from the stream-ordered pool per run and freed right after — dozens of cached plans
    // (audio codecs build one per shape/dilation) must not each pin a resident workspace for the session.
    private unsafe void Run(Plan plan, ulong xPtr, ulong w, ulong yPtr)
    {
        nint vp = 0;
        ulong workspace = 0;
        try
        {
            if (plan.WorkspaceBytes > 0)
                workspace = CudaMemory.AllocateAsync((nuint)plan.WorkspaceBytes, _stream);
            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, out vp), "variant pack create");
            long* uids = stackalloc long[3] { UidX, UidW, UidY };
            void** ptrs = stackalloc void*[3] { (void*)xPtr, (void*)w, (void*)yPtr };
            void* ws = (void*)workspace;
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws);
            Check(cudnnBackendFinalize(vp), "variant pack finalize");
            Check(cudnnBackendExecute(_handle, plan.Execution, vp), "cudnnBackendExecute");
        }
        finally
        {
            if (vp != 0) cudnnBackendDestroyDescriptor(vp);
            if (workspace != 0) CudaMemory.FreeAsync(workspace, _stream);
        }
    }

    // X-slot dims are (h, wIn) and Y-slot dims (outH, outW): for forward X is the conv input and Y the output;
    // for backwardData X is DX (the large transposed output) and Y is DY (the small input) — the conv descriptor
    // always describes the forward geometry, so callers pass the slot dims accordingly.
    private unsafe Plan BuildPlan(bool backwardData, long n, long c, long h, long wIn, long k, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padWPre, long padWPost, int dataType,
        long dilationH = 1, long dilationW = 1)
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
            long* dil = stackalloc long[2] { dilationH, dilationW };
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dil);
            long* strides = stackalloc long[2] { strideH, strideW };
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, strides);
            long* prePads = stackalloc long[2] { padH, padWPre };
            long* postPads = stackalloc long[2] { padH, padWPost };
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, prePads);
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, postPads);
            Check(cudnnBackendFinalize(conv), "conv desc finalize");

            int opDescType = backwardData
                ? CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
                : CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
            Check(cudnnBackendCreateDescriptor(opDescType, out nint op), "conv op create");
            owned.Add(op);
            float alpha = 1.0f, beta = 0.0f;
            SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                CUDNN_TYPE_FLOAT, 1, &alpha);
            SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                CUDNN_TYPE_FLOAT, 1, &beta);
            void* cp = (void*)conv, xp = (void*)tX, wp = (void*)tW, yp = (void*)tY;
            SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cp);
            SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xp);
            SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wp);
            SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yp);
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
            // See CudnnSdpa.BuildExecutionPlan's identical pattern for why this try/finally exists: TryPlan
            // can throw (SetAttr failure) instead of returning ok=false, and without this, an exception
            // mid-loop would leak up to 32 backend descriptors — worsening exactly the kind of
            // resource-pressure condition that causes such a throw.
            bool[] destroyed = new bool[maxCfgs];
            try
            {
                for (int i = 0; i < maxCfgs; i++)
                {
                    if (i < returned)
                    {
                        (nint exec, long ws, bool ok) = TryPlan(cfgs[i]);
                        if (ok && ws > MaxWorkspaceBytes)
                        {
                            cudnnBackendDestroyDescriptor(exec);
                            ok = false;
                        }
                        if (ok)
                        {
                            for (int j = 0; j < maxCfgs; j++)
                            {
                                if (!destroyed[j]) { cudnnBackendDestroyDescriptor(cfgs[j]); destroyed[j] = true; }
                            }
                            return (exec, ws);
                        }
                    }
                    cudnnBackendDestroyDescriptor(cfgs[i]);
                    destroyed[i] = true;
                }
            }
            finally
            {
                for (int i = 0; i < maxCfgs; i++)
                    if (!destroyed[i]) cudnnBackendDestroyDescriptor(cfgs[i]);
            }
        }
        throw new InvalidOperationException("cuDNN conv: no engine config produced a valid execution plan");
    }

    private unsafe (nint exec, long ws, bool ok) TryPlan(nint cfg)
    {
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, out nint p) != CUDNN_STATUS_SUCCESS)
            return (0, 0, false);
        try
        {
            void* hp = (void*)_handle;
            void* cp = (void*)cfg;
            SetAttr(p, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &hp);
            SetAttr(p, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cp);
        }
        catch
        {
            cudnnBackendDestroyDescriptor(p);
            throw;
        }
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
            throw new CudnnStatusException(st, $"cudnnBackendSetAttribute(attr={attr})");
    }

    private static void Check(int st, string what)
    {
        if (st != CUDNN_STATUS_SUCCESS)
            throw new CudnnStatusException(st, what);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Plan p in _plans.Values)
        {
            if (p.Execution != 0) cudnnBackendDestroyDescriptor(p.Execution);
        }
        _plans.Clear();
        if (_handle != 0) cudnnDestroy(_handle);
    }
}
