using System.Collections.Concurrent;
using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>Convolution forward via cuDNN's backend graph API. Replaces the im2col→cuBLAS GEMM path for F16/BF16 NCHW convolutions: cuDNN's heuristics pick tensor-core implicit-GEMM/Winograd engines that never materialize the im2col matrix (an extra kH·kW-times-input-sized HBM write+read per conv — the dominant conv cost in the SDXL UNet, which runs ~50 convolutions per step). Graph = a single CONVOLUTION_FORWARD op (X ⊛ W → Y) or CONVOLUTION_BACKWARD_DATA op (transposed convolution: DY ⊛ W → DX) — cross-correlation, fp32 accumulate — over NCHW strided tensors, alpha 1 / beta 0. Bias stays a separate kernel in the caller — same numerics as the GEMM path's bias add. Execution plans (heuristics + JIT) are cached by shape+dtype; workspace comes from the stream-ordered pool per execution (capped per plan). Instances are per <see cref="CudaBackend"/> (one cuDNN handle bound to the compute stream). Any failure is caught by the caller, which self-disables the route for the session and falls back to im2col — a wrong shape costs one warning, never a session kill.</summary>
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
        int st = cudnnCreate(out nint handle);
        if (st != CUDNN_STATUS_SUCCESS)
            throw new InvalidOperationException($"cudnnCreate failed: {ErrorString(st)}");
        try
        {
            Check(cudnnSetStream(handle, stream), "cudnnSetStream");
        }
        catch (Exception setupFailure)
        {
            int cleanupStatus = cudnnDestroy(handle);
            if (cleanupStatus != CUDNN_STATUS_SUCCESS)
            {
                throw new AggregateException(
                    "cuDNN convolution setup and handle rollback both failed.", setupFailure,
                    new CudnnStatusException(cleanupStatus, "cudnnDestroy after setup failure"));
            }
            throw;
        }
        _handle = handle;
        _stream = stream;
    }

    private const long UidX = 1, UidW = 2, UidY = 3;

    /// <summary>Runs X[n,c,h,w] ⊛ W[k,c,r,s] → Y[n,k,outH,outW]. All pointers are device buffers of <paramref name="dataType"/> (CUDNN_DATA_HALF / CUDNN_DATA_BFLOAT16), contiguous NCHW. W-padding may be asymmetric (<paramref name="padWPre"/> zeros before, <paramref name="padWPost"/> after) — the backend graph API keeps PRE/POST paddings as separate attributes, which lets causal (left-padded) 1D convs run without an explicit pad-copy.</summary>
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

    /// <summary>Forward convolution over CHANNELS-LAST buffers, 2 or 3 spatial dims: X <c>[N, spatial…, C]</c>,
    /// W <c>[K, kernel…, C]</c>, Y <c>[N, outSpatial…, K]</c>, symmetric padding. cuDNN's tensor-core engines are built
    /// for this layout: at 3×3, 256-512 channels and BF16 they run ~155-165 TFLOPS on a 4090, where the same conv over
    /// NCHW gets ~40. The dims arrays are in logical NC… order (<c>[N, C, spatial…]</c>, <c>[K, C, kernel…]</c>).</summary>
    public void ExecuteChannelsLast(ulong x, ulong w, ulong y, long[] xDim, long[] wDim, long[] yDim,
        long[] strides, long[] pads, int dataType)
    {
        string key = $"cl|{string.Join(',', xDim)}|{string.Join(',', wDim)}|{string.Join(',', strides)}|{string.Join(',', pads)}|{dataType}";
        Plan plan = _plans.GetOrAdd(key, _ =>
        {
            long[] dil = new long[strides.Length];
            Array.Fill(dil, 1L);
            return BuildPlanNd(backwardData: false, xDim, ChannelsLastStrides(xDim), wDim, ChannelsLastStrides(wDim),
                yDim, ChannelsLastStrides(yDim), strides, pads, pads, dil, dataType);
        });
        Run(plan, x, w, y);
    }

    /// <summary>Element strides of a channels-last buffer for logical dims <c>[N, C, s1, …, sd]</c>.</summary>
    private static long[] ChannelsLastStrides(long[] dims)
    {
        long[] strides = new long[dims.Length];
        long inner = dims[1];                    // channels are innermost
        strides[1] = 1;
        for (int i = dims.Length - 1; i >= 2; i--)
        {
            strides[i] = inner;
            inner *= dims[i];
        }
        strides[0] = inner;
        return strides;
    }

    /// <summary>Transposed convolution as cuDNN convolution-backward-data: DY[n,k,h,wIn] (the transpose-conv input) ⊛ W[k,c,r,s] → DX[n,c,outH,outW]. Geometry attributes describe the corresponding FORWARD conv (DX is the conv input), so the pads crop the full transposed output: outW = (wIn−1)·strideW + dilationW·(s−1) + 1 − padWPre − padWPost.</summary>
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
    private Plan BuildPlan(bool backwardData, long n, long c, long h, long wIn, long k, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padWPre, long padWPost, int dataType,
        long dilationH = 1, long dilationW = 1)
        => BuildPlanNd(backwardData,
            [n, c, h, wIn], [c * h * wIn, h * wIn, wIn, 1],
            [k, c, r, s], [c * r * s, r * s, s, 1],
            [n, k, outH, outW], [k * outH * outW, outH * outW, outW, 1],
            [strideH, strideW], [padH, padWPre], [padH, padWPost], [dilationH, dilationW], dataType);

    /// <summary>One convolution op over N spatial dims (dims/strides are rank N+2, the rest rank N).</summary>
    private unsafe Plan BuildPlanNd(bool backwardData, long[] xDim, long[] xStr, long[] wDim, long[] wStr,
        long[] yDim, long[] yStr, long[] strides, long[] prePads, long[] postPads, long[] dilations, int dataType)
    {
        List<nint> owned = new();
        try
        {
            nint tX = Tensor(owned, UidX, xDim, xStr, dataType);
            nint tW = Tensor(owned, UidW, wDim, wStr, dataType);
            nint tY = Tensor(owned, UidY, yDim, yStr, dataType);

            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, out nint conv), "conv desc create");
            owned.Add(conv);
            long spatial = strides.Length;
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
            int comp = CUDNN_DATA_FLOAT;
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp);
            int mode = CUDNN_CROSS_CORRELATION;
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
            fixed (long* dil = dilations, str = strides, pre = prePads, post = postPads)
            {
                SetAttr(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, spatial, dil);
                SetAttr(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, spatial, str);
                SetAttr(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, spatial, pre);
                SetAttr(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, spatial, post);
            }
            Check(cudnnBackendFinalize(conv), "conv desc finalize");

            int opDescType = backwardData ? CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
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

            (nint exec, long wsBytes) = CudnnPlanSearch.BuildExecutionPlan(_handle, graph, owned, MaxWorkspaceBytes, "conv");
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


    private unsafe nint Tensor(List<nint> owned, long uid, long[] dims, long[] strides, int dtype)
    {
        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, out nint t), "tensor create");
        owned.Add(t);
        int dt = dtype;
        SetAttr(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
        fixed (long* d = dims, st = strides)
        {
            SetAttr(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dims.Length, d);
            SetAttr(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, strides.Length, st);
        }
        long id = uid;
        SetAttr(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id);
        long align = 16;
        SetAttr(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
        byte v = 0;
        SetAttr(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &v);
        Check(cudnnBackendFinalize(t), "tensor finalize");
        return t;
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
