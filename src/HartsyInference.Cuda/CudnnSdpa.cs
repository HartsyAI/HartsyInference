using System.Collections.Concurrent;
using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>Fused scaled-dot-product attention via cuDNN's flash-attention engine (backend graph API).
/// Replaces the materialized cuBLAS QKᵀ→softmax→PV path (which allocates a [B·H, Sq, Skv] score matrix)
/// with a single fused kernel that never materializes the scores — ~34× faster at Krea2 self-attention
/// shape (B=1,H=24,S=4608,D=128: 62.7ms → 1.8ms/call, workspace 0), measured on RTX 4090.
///
/// The graph mirrors what NVIDIA's cudnn-frontend emits for plain inference SDPA on cuDNN ≥ 9.21:
///   bmm1: S = Q @ Kᵀ  (matmul, fp32 accum) → scale: Ss = S·attn_scale (pointwise mul)
///   → softmax: P = softmax(Ss)  (the UNIFIED backend SOFTMAX op — the decomposed
///     max/sub/exp/sum/div graph does NOT match the fused engine) → bmm2: O = P @ V.
/// Tensors are 4D [B,H,S,D], fp16 I/O + fp32 compute. Q/K/V/O and the scale scalar are non-virtual I/O;
/// S/Ss/P are virtual so the engine keeps them in registers/shared memory.
///
/// Execution plans are expensive to build (heuristics + JIT) and cheap to run, so they are cached by
/// shape, exact scale bits, and bias layout. Instances are created per <see cref="CudaBackend"/> (one cuDNN
/// handle bound to the compute stream).</summary>
internal sealed class CudnnSdpa : IDisposable
{
    private readonly nint _handle;
    private readonly ConcurrentDictionary<PlanKey, Lazy<Plan>> _plans = new();
    private bool _disposed;

    /// <summary>Memory layout of the Q/K/V/O device buffers. <c>TokenMajor</c> is [b,s,h,d] addressed purely by
    /// strides — what a fused QKV projection already produces, so callers can skip the permute on both sides.</summary>
    internal enum SdpaLayout { HeadMajor, TokenMajor }

    /// <summary>Exact identity of a cached execution plan and its immutable device scale scalar.</summary>
    internal readonly record struct PlanKey(
        long B, long H, long Sq, long Sk, long D, int ScaleBits, bool HasBias, long BiasB, SdpaLayout Layout);

    private sealed class Plan
    {
        public nint Execution;         // execution-plan descriptor (kept alive; reused every call)
        public ulong Workspace;        // device workspace buffer (0 for the fused engine)
        public long WorkspaceBytes;
        public ulong ScaleBuf;         // 4-byte device scalar holding the scale encoded in this plan's key
    }

    public CudnnSdpa(nint stream)
    {
        int st = cudnnCreate(out nint handle);
        if (st != CUDNN_STATUS_SUCCESS)
            throw new InvalidOperationException($"cudnnCreate failed: {ErrorString(st)}");
        try
        {
            Check(cudnnSetStream(handle, stream), "cudnnSetStream");
        }
        catch
        {
            cudnnDestroy(handle);
            throw;
        }
        _handle = handle;
    }

    /// <summary>D values the fused engine may support (head dim): multiples of 8 in [64, 128] (the documented
    /// flash-fprop envelope on SM80+ — covers 64/96/112/120/128, e.g. Boogu's 120) plus 256 (Ideogram 4,
    /// build/arch-dependent). The caller falls back per-D on rejection (<c>_cudnnSdpaDeadDims</c>), so an
    /// unsupported D costs one warning and the materialized path — never a session kill.</summary>
    public static bool ShapeSupported(long d) => d == 256 || (d >= 64 && d <= 128 && d % 8 == 0);

    /// <summary>
    /// Run fused attention. All pointers are device fp16 buffers laid out contiguously as [B,H,S,D]
    /// (Q/O with Sq rows, K/V with Skv rows). <paramref name="scale"/> is the softmax pre-scale (1/√D typically).
    /// <paramref name="biasF32"/> (optional, 0 = none) is a device fp32 additive attention bias/mask laid out as
    /// [biasB,1,Sq,Skv], broadcast over heads (and over batch when biasB==1 &lt; b), added to the scaled scores
    /// before softmax — the cudnn-frontend Bias score-modifier pattern, which still hits the fused engine.
    /// </summary>
    public unsafe void Execute(ulong qF16, ulong kF16, ulong vF16, ulong oF16,
                               long b, long h, long sq, long sk, long d, float scale,
                               ulong biasF32 = 0, long biasB = 1)
        => Execute(qF16, kF16, vF16, oF16, b, h, sq, sk, d, scale, SdpaLayout.HeadMajor, biasF32, biasB);

    /// <summary>Same as the head-major overload but lets the caller pick the Q/K/V/O buffer layout.</summary>
    internal unsafe void Execute(ulong qF16, ulong kF16, ulong vF16, ulong oF16,
                                 long b, long h, long sq, long sk, long d, float scale,
                                 SdpaLayout layout, ulong biasF32 = 0, long biasB = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        bool hasBias = biasF32 != 0;
        PlanKey key = new PlanKey(
            b, h, sq, sk, d, BitConverter.SingleToInt32Bits(scale), hasBias, biasB, layout);
        Lazy<Plan> candidate = new(
            () => BuildPlan(b, h, sq, sk, d, scale, hasBias, biasB, layout),
            LazyThreadSafetyMode.ExecutionAndPublication);
        Lazy<Plan> cached = _plans.GetOrAdd(key, candidate);
        Plan plan;
        try
        {
            plan = cached.Value;
        }
        catch
        {
            // Failed plans must be retryable after transient allocation pressure clears.
            if (_plans.TryGetValue(key, out Lazy<Plan>? current) && ReferenceEquals(current, cached))
                _plans.TryRemove(key, out _);
            throw;
        }

        // Variant pack: bind the actual device pointers to the graph's tensor UIDs, then execute.
        nint vp = 0;
        try
        {
            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, out vp), "variant pack create");
            long n = hasBias ? 6 : 5;
            long* uids = stackalloc long[6] { UidQ, UidK, UidV, UidO, UidScale, UidBias };
            void** ptrs = stackalloc void*[6]
            {
                (void*)qF16, (void*)kF16, (void*)vF16, (void*)oF16, (void*)plan.ScaleBuf, (void*)biasF32
            };
            void* ws = (void*)plan.Workspace;
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, n, uids);
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, n, ptrs);
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws);
            Check(cudnnBackendFinalize(vp), "variant pack finalize");
            Check(cudnnBackendExecute(_handle, plan.Execution, vp), "cudnnBackendExecute");
        }
        finally
        {
            if (vp != 0) cudnnBackendDestroyDescriptor(vp);
        }
    }

    // ── graph tensor UIDs ───────────────────────────────────────────────
    private const long UidQ = 1, UidK = 2, UidV = 3, UidO = 4, UidScale = 5, UidBias = 6;
    private const long UidS = 100, UidSS = 101, UidP = 102, UidSB = 103;

    private unsafe Plan BuildPlan(
        long b, long h, long sq, long sk, long d, float scale, bool hasBias, long biasB, SdpaLayout layout)
    {
        List<nint> owned = new();   // build-time descriptors to destroy once the plan is finalized
        try
        {
            // ── tensors (4D [b,h,s,d]); only the head/sequence strides differ between the two layouts ──
            bool tokenMajor = layout == SdpaLayout.TokenMajor;
            long qHeadStr = tokenMajor ? d : sq * d, qSeqStr = tokenMajor ? h * d : d;
            long kHeadStr = tokenMajor ? d : sk * d, kSeqStr = tokenMajor ? h * d : d;
            long* qDim = stackalloc long[4] { b, h, sq, d };
            long* qStr = stackalloc long[4] { h * sq * d, qHeadStr, qSeqStr, 1 };
            nint tQ = Tensor(owned, UidQ, qDim, qStr, CUDNN_DATA_HALF, false);
            // Kᵀ: view the [b,h,sk,d] buffer as [b,h,d,sk] by swapping the last two dims/strides.
            long* ktDim = stackalloc long[4] { b, h, d, sk };
            long* ktStr = stackalloc long[4] { h * sk * d, kHeadStr, 1, kSeqStr };
            nint tKt = Tensor(owned, UidK, ktDim, ktStr, CUDNN_DATA_HALF, false);
            long* vDim = stackalloc long[4] { b, h, sk, d };
            long* vStr = stackalloc long[4] { h * sk * d, kHeadStr, kSeqStr, 1 };
            nint tV = Tensor(owned, UidV, vDim, vStr, CUDNN_DATA_HALF, false);
            long* oDim = stackalloc long[4] { b, h, sq, d };
            long* oStr = stackalloc long[4] { h * sq * d, qHeadStr, qSeqStr, 1 };
            nint tO = Tensor(owned, UidO, oDim, oStr, CUDNN_DATA_HALF, false);
            long* one = stackalloc long[4] { 1, 1, 1, 1 };
            nint tScale = Tensor(owned, UidScale, one, one, CUDNN_DATA_FLOAT, false);
            long* sDim = stackalloc long[4] { b, h, sq, sk };
            long* sStr = stackalloc long[4] { h * sq * sk, sq * sk, sk, 1 };
            nint tS = Tensor(owned, UidS, sDim, sStr, CUDNN_DATA_FLOAT, true);
            nint tSS = Tensor(owned, UidSS, sDim, sStr, CUDNN_DATA_FLOAT, true);
            nint tP = Tensor(owned, UidP, sDim, sStr, CUDNN_DATA_HALF, true);

            // ── ops: bmm1 → scale → [+ bias] → softmax → bmm2 ──
            nint* ops = stackalloc nint[5];
            long opCount = 0;
            ops[opCount++] = MatmulOp(owned, tQ, tKt, tS);
            ops[opCount++] = PointwiseOp(owned, CUDNN_POINTWISE_MUL, tS, tScale, tSS);
            nint softmaxIn = tSS;
            if (hasBias)
            {
                // Additive fp32 attention bias [biasB,1,sq,sk], broadcast over heads (dim-1 axes broadcast in
                // pointwise ops) — the cudnn-frontend Bias score-modifier. -1e30 masked entries are safe: the
                // add runs in fp32 and the fused softmax subtracts the row max (fully-masked rows go uniform,
                // matching the materialized path's convention).
                long* bDim = stackalloc long[4] { biasB, 1, sq, sk };
                long* bStr = stackalloc long[4] { sq * sk, sq * sk, sk, 1 };
                nint tBias = Tensor(owned, UidBias, bDim, bStr, CUDNN_DATA_FLOAT, false);
                nint tSB = Tensor(owned, UidSB, sDim, sStr, CUDNN_DATA_FLOAT, true);
                ops[opCount++] = PointwiseOp(owned, CUDNN_POINTWISE_ADD, tSS, tBias, tSB);
                softmaxIn = tSB;
            }
            ops[opCount++] = SoftmaxOp(owned, softmaxIn, tP);
            ops[opCount++] = MatmulOp(owned, tP, tV, tO);

            nint graph;
            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, out graph), "graph create");
            owned.Add(graph);
            void* hp = (void*)_handle;
            SetAttr(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &hp);
            SetAttr(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, opCount, ops);
            Check(cudnnBackendFinalize(graph), "graph finalize");

            (nint exec, long wsBytes) = BuildExecutionPlan(graph, owned);

            Plan plan = new()
            {
                Execution = exec,
                WorkspaceBytes = wsBytes,
            };
            try
            {
                // Plans outlive individual stream operations and DestroyPlan releases these buffers with
                // synchronous cuMemFree. Keep the allocator pair symmetric: stream-pool allocations must be
                // paired with FreeAsync, while session-persistent plans use AllocatePersistent + Free.
                plan.Workspace = wsBytes > 0 ? CudaMemory.AllocatePersistent((nuint)wsBytes) : 0;
                plan.ScaleBuf = CudaMemory.AllocatePersistent(sizeof(float));
                CudaMemory.CopyHostToDevice(plan.ScaleBuf, &scale, sizeof(float));
                return plan;
            }
            catch
            {
                DestroyPlan(plan);
                throw;
            }
        }
        finally
        {
            // The finalized execution plan owns its own state; the intermediate descriptors can go.
            foreach (nint d0 in owned)
                cudnnBackendDestroyDescriptor(d0);
        }
    }

    private unsafe (nint exec, long wsBytes) BuildExecutionPlan(nint graph, List<nint> owned)
    {
        // Ask heuristics (mode A = recommended runtime-compiled fused engines first) for engine configs,
        // then finalize the first that yields a valid plan. Fall back to FALLBACK mode if A is empty.
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
            // TryPlan can throw (SetAttr failure, e.g. a transient host-allocation error) instead of
            // returning ok=false — without this try/finally, an exception mid-loop would skip every
            // remaining cfgs[i]'s destroy call (both the current index and every index not yet reached),
            // leaking up to 32 backend descriptors per throw. Under exactly the resource-pressure
            // conditions that cause such a throw, that leak would make the underlying pressure worse with
            // every retry — tracked per-index so the normal (non-throwing) destroy calls below aren't
            // double-freed here.
            bool[] destroyed = new bool[maxCfgs];
            try
            {
                for (int i = 0; i < maxCfgs; i++)
                {
                    if (i < returned)
                    {
                        (nint exec, long ws, bool ok) = TryPlan(cfgs[i]);
                        if (ok)
                        {
                            // free the unused cfg descriptors, keep going only to release; exec plan is retained
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
        throw new InvalidOperationException("cuDNN SDPA: no engine config produced a valid execution plan");
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
            // SetAttr throws directly on failure (doesn't return a status) — without this, a thrown
            // exception here would leak the just-created execution-plan descriptor `p`.
            cudnnBackendDestroyDescriptor(p);
            throw;
        }
        if (cudnnBackendFinalize(p) != CUDNN_STATUS_SUCCESS)
        {
            cudnnBackendDestroyDescriptor(p);
            return (0, 0, false);
        }
        long ws = 0;
        try
        {
            Check(cudnnBackendGetAttribute(
                p, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, out _, &ws),
                "workspace size get");
        }
        catch
        {
            cudnnBackendDestroyDescriptor(p);
            throw;
        }
        return (p, ws, true);
    }

    // ── descriptor builders ─────────────────────────────────────────────
    private unsafe nint Tensor(List<nint> owned, long uid, long* dims, long* strides, int dtype, bool virt)
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
        byte v = (byte)(virt ? 1 : 0);
        SetAttr(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &v);
        Check(cudnnBackendFinalize(t), "tensor finalize");
        return t;
    }

    private unsafe nint MatmulOp(List<nint> owned, nint a, nint bt, nint c)
    {
        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR, out nint mm), "matmul desc create");
        owned.Add(mm);
        int comp = CUDNN_DATA_FLOAT;
        SetAttr(mm, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp);
        Check(cudnnBackendFinalize(mm), "matmul desc finalize");

        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, out nint op), "matmul op create");
        owned.Add(op);
        void* ap = (void*)a, bp = (void*)bt, cp = (void*)c, mp = (void*)mm;
        SetAttr(op, CUDNN_ATTR_OPERATION_MATMUL_ADESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ap);
        SetAttr(op, CUDNN_ATTR_OPERATION_MATMUL_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bp);
        SetAttr(op, CUDNN_ATTR_OPERATION_MATMUL_CDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cp);
        SetAttr(op, CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &mp);
        Check(cudnnBackendFinalize(op), "matmul op finalize");
        return op;
    }

    private unsafe nint PointwiseOp(List<nint> owned, int mode, nint x, nint bTensor, nint y)
    {
        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, out nint pw), "pw desc create");
        owned.Add(pw);
        int prec = CUDNN_DATA_FLOAT;
        SetAttr(pw, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode);
        SetAttr(pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &prec);
        Check(cudnnBackendFinalize(pw), "pw desc finalize");

        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, out nint op), "pw op create");
        owned.Add(op);
        void* pwp = (void*)pw, xp = (void*)x, bp = (void*)bTensor, yp = (void*)y;
        SetAttr(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pwp);
        SetAttr(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xp);
        SetAttr(op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bp);
        SetAttr(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yp);
        Check(cudnnBackendFinalize(op), "pw op finalize");
        return op;
    }

    private unsafe nint SoftmaxOp(List<nint> owned, nint x, nint y)
    {
        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR, out nint op), "softmax op create");
        owned.Add(op);
        void* xp = (void*)x, yp = (void*)y;
        SetAttr(op, CUDNN_ATTR_OPERATION_SOFTMAX_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xp);
        SetAttr(op, CUDNN_ATTR_OPERATION_SOFTMAX_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yp);
        Check(cudnnBackendFinalize(op), "softmax op finalize");
        return op;
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

    private static Exception? DestroyPlan(Plan plan)
    {
        Exception? firstError = null;
        if (plan.ScaleBuf != 0)
        {
            try { CudaMemory.Free(plan.ScaleBuf); }
            catch (Exception error) { firstError ??= error; }
        }
        if (plan.Workspace != 0)
        {
            try { CudaMemory.Free(plan.Workspace); }
            catch (Exception error) { firstError ??= error; }
        }
        if (plan.Execution != 0)
        {
            try
            {
                int status = cudnnBackendDestroyDescriptor(plan.Execution);
                if (status != CUDNN_STATUS_SUCCESS)
                    throw new CudnnStatusException(status, "execution plan destroy");
            }
            catch (Exception error) { firstError ??= error; }
        }
        return firstError;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Exception? firstError = null;
        foreach (Lazy<Plan> cached in _plans.Values)
        {
            if (!cached.IsValueCreated) continue;
            try
            {
                Exception? planError = DestroyPlan(cached.Value);
                firstError ??= planError;
            }
            catch (Exception error)
            {
                // A faulted Lazy should normally have been removed by Execute. It must not stop teardown of the
                // other independently owned plans or the cuDNN handle if it is still present here during a race.
                firstError ??= error;
            }
        }
        _plans.Clear();
        if (_handle != 0)
        {
            try
            {
                int status = cudnnDestroy(_handle);
                if (status != CUDNN_STATUS_SUCCESS)
                    throw new CudnnStatusException(status, "cudnnDestroy");
            }
            catch (Exception error) { firstError ??= error; }
        }
        if (firstError is not null)
            throw new InvalidOperationException("One or more cuDNN SDPA resources failed to release.", firstError);
    }
}
