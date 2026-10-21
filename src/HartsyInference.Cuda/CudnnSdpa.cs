using System.Collections.Concurrent;
using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>Fused scaled-dot-product attention via cuDNN's flash-attention engine (backend graph API). Replaces the materialized cuBLAS QKᵀ→softmax→PV path (which allocates a [B·H, Sq, Skv] score matrix) with a single fused kernel that never materializes the scores — ~34× faster at Krea2 self-attention shape (B=1,H=24,S=4608,D=128: 62.7ms → 1.8ms/call, workspace 0), measured on RTX 4090. The graph mirrors what NVIDIA's cudnn-frontend emits for plain inference SDPA on cuDNN ≥ 9.21: bmm1: S = Q @ Kᵀ (matmul, fp32 accum) → scale: Ss = S·attn_scale (pointwise mul) → [bias add] → softmax: P = softmax(Ss) (the UNIFIED backend SOFTMAX op — the decomposed max/sub/exp/sum/div graph does NOT match the fused engine) → bmm2: O = P @ V. Tensors are 4D [B,H,S,D] with fp16 or bf16 I/O and fp32 compute; K/V carry their own head count (grouped-query attention needs no widening). Q/K/V/O and the bias are non-virtual I/O, the scale is a by-value host scalar bound on every call, and S/Ss/P are virtual so the engine keeps them in registers/shared memory. Execution plans are expensive to build (heuristics + JIT) and cheap to run, so they are cached by shape, dtype, head counts and bias layout — not by scale. Instances are created per <see cref="CudaBackend"/> (one cuDNN handle bound to the compute stream).</summary>
internal sealed class CudnnSdpa : IDisposable
{
    private readonly nint _handle;
    private readonly ConcurrentDictionary<PlanKey, Lazy<Plan>> _plans = new();
    private bool _disposed;

    /// <summary>Memory layout of the Q/K/V/O device buffers. <c>TokenMajor</c> is [b,s,h,d] addressed purely by strides — what a fused QKV projection already produces, so callers can skip the permute on both sides.</summary>
    internal enum SdpaLayout { HeadMajor, TokenMajor }

    /// <summary>I/O dtype of Q/K/V/O; scores and bias stay fp32 and P follows this dtype.</summary>
    internal enum SdpaDtype { F16, BF16 }

    /// <summary>Identity of a cached plan; the scale is bound per call, so it is not in the key. Bias is fp32 [BiasB, BiasH, BiasSq, Sk], <c>BiasSq == 0</c> for none.</summary>
    internal readonly record struct PlanKey(long B, long H, long Hkv, long Sq, long Sk, long D, SdpaDtype Dtype, SdpaLayout Layout,
        long BiasB = 0, long BiasH = 0, long BiasSq = 0)
    {
        public bool HasBias => BiasSq > 0;

        internal int CudnnDtype => Dtype == SdpaDtype.BF16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_HALF;

        internal void Validate()
        {
            if (B <= 0 || H <= 0 || Hkv <= 0 || Sq <= 0 || Sk <= 0 || D <= 0)
                throw new ArgumentException($"SDPA dims must be positive: {this}.");
            if (H % Hkv != 0)
                throw new ArgumentException($"SDPA K/V heads {Hkv} must divide query heads {H}.");
            if (HasBias && ((BiasB != 1 && BiasB != B) || (BiasH != 1 && BiasH != H) || (BiasSq != 1 && BiasSq != Sq)))
                throw new ArgumentException($"SDPA bias [{BiasB},{BiasH},{BiasSq},{Sk}] does not broadcast to [{B},{H},{Sq},{Sk}].");
        }
    }

    private sealed class Plan
    {
        public nint Execution;         // execution-plan descriptor (kept alive; reused every call)
        public ulong Workspace;        // device workspace buffer (0 for the fused engine)
        public long WorkspaceBytes;
    }

    /// <summary>Execution plans built so far.</summary>
    internal int PlanCount => _plans.Count;

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

    /// <summary>Head dims the fused engine runs: multiples of 8 from 32 to 256 (cuDNN 9.24 on SM 86 and 89). A build that
    /// refuses one falls back for that head dim only.</summary>
    public static bool ShapeSupported(long d) => d >= MinHeadDim && d <= MaxHeadDim && d % 8 == 0;

    internal const int MinHeadDim = 32, MaxHeadDim = 256;

    /// <summary>Run fused attention over device buffers of the key's dtype: Q/O [B,H,Sq,D], K/V [B,Hkv,Sk,D] (head-major) or [B,S,heads,D] (token-major). <paramref name="biasF32"/> (0 = none, matching <see cref="PlanKey.HasBias"/>) is an fp32 additive bias [BiasB,BiasH,BiasSq,Sk]; extents of 1 broadcast, and BiasSq 1 is a key-only bias.</summary>
    public unsafe void Execute(ulong q, ulong k, ulong v, ulong o, in PlanKey key, float scale, ulong biasF32 = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (key.HasBias != (biasF32 != 0))
            throw new ArgumentException("SDPA bias pointer and plan key disagree about whether a bias is present.");
        key.Validate();
        PlanKey planKey = key;
        Lazy<Plan> candidate = new(() => BuildPlan(planKey), LazyThreadSafetyMode.ExecutionAndPublication);
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
            long n = key.HasBias ? 6 : 5;
            float scaleValue = scale;   // by-value tensors bind a host pointer that cuDNN must read inside cudnnBackendExecute
            long* uids = stackalloc long[6] { UidQ, UidK, UidV, UidO, UidScale, UidBias };
            void** ptrs = stackalloc void*[6]
            {
                (void*)q, (void*)k, (void*)v, (void*)o, &scaleValue, (void*)biasF32
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

    private unsafe Plan BuildPlan(PlanKey key)
    {
        List<nint> owned = new();   // build-time descriptors to destroy once the plan is finalized
        try
        {
            long b = key.B, h = key.H, hKv = key.Hkv, sq = key.Sq, sk = key.Sk, d = key.D;
            int io = key.CudnnDtype;
            // ── tensors (4D [b,heads,s,d]); only the head/sequence strides differ between the two layouts ──
            bool tokenMajor = key.Layout == SdpaLayout.TokenMajor;
            long qHeadStr = tokenMajor ? d : sq * d, qSeqStr = tokenMajor ? h * d : d;
            long kHeadStr = tokenMajor ? d : sk * d, kSeqStr = tokenMajor ? hKv * d : d;
            long* qDim = stackalloc long[4] { b, h, sq, d };
            long* qStr = stackalloc long[4] { h * sq * d, qHeadStr, qSeqStr, 1 };
            nint tQ = Tensor(owned, UidQ, qDim, qStr, io, false);
            // Kᵀ: view the [b,hKv,sk,d] buffer as [b,hKv,d,sk] by swapping the last two dims/strides.
            long* ktDim = stackalloc long[4] { b, hKv, d, sk };
            long* ktStr = stackalloc long[4] { hKv * sk * d, kHeadStr, 1, kSeqStr };
            nint tKt = Tensor(owned, UidK, ktDim, ktStr, io, false);
            long* vDim = stackalloc long[4] { b, hKv, sk, d };
            long* vStr = stackalloc long[4] { hKv * sk * d, kHeadStr, kSeqStr, 1 };
            nint tV = Tensor(owned, UidV, vDim, vStr, io, false);
            long* oDim = stackalloc long[4] { b, h, sq, d };
            long* oStr = stackalloc long[4] { h * sq * d, qHeadStr, qSeqStr, 1 };
            nint tO = Tensor(owned, UidO, oDim, oStr, io, false);
            long* one = stackalloc long[4] { 1, 1, 1, 1 };
            nint tScale = Tensor(owned, UidScale, one, one, CUDNN_DATA_FLOAT, false, byValue: true);
            long* sDim = stackalloc long[4] { b, h, sq, sk };
            long* sStr = stackalloc long[4] { h * sq * sk, sq * sk, sk, 1 };
            nint tS = Tensor(owned, UidS, sDim, sStr, CUDNN_DATA_FLOAT, true);
            nint tSS = Tensor(owned, UidSS, sDim, sStr, CUDNN_DATA_FLOAT, true);
            nint tP = Tensor(owned, UidP, sDim, sStr, io, true);

            // ── ops: bmm1 → scale → [+ bias] → softmax → bmm2 ──
            nint* ops = stackalloc nint[5];
            long opCount = 0;
            ops[opCount++] = MatmulOp(owned, tQ, tKt, tS);
            ops[opCount++] = PointwiseOp(owned, CUDNN_POINTWISE_MUL, tS, tScale, tSS);
            nint softmaxIn = tSS;
            if (key.HasBias)
            {
                // -1e30 masked entries are safe: the add runs in fp32 and softmax subtracts the row max (fully masked rows go uniform).
                long* bDim = stackalloc long[4] { key.BiasB, key.BiasH, key.BiasSq, sk };
                long* bStr = stackalloc long[4] { key.BiasH * key.BiasSq * sk, key.BiasSq * sk, sk, 1 };
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

            string shape = $"SDPA batch={b} heads={h} kvHeads={hKv} seqQ={sq} seqKV={sk} headdim={d} dtype={CudnnPlanSearch.DtypeName(io)}"
                + (key.HasBias ? $" bias=[{key.BiasB},{key.BiasH},{key.BiasSq},{sk}]" : "");
            (nint exec, long wsBytes) = CudnnPlanSearch.BuildExecutionPlan(_handle, graph, owned, long.MaxValue, "SDPA", shape);

            Plan plan = new()
            {
                Execution = exec,
                WorkspaceBytes = wsBytes,
            };
            try
            {
                // Plans outlive individual stream operations and DestroyPlan releases the workspace with
                // synchronous cuMemFree. Keep the allocator pair symmetric: stream-pool allocations must be
                // paired with FreeAsync, while session-persistent plans use AllocatePersistent + Free.
                plan.Workspace = wsBytes > 0 ? CudaMemory.AllocatePersistent((nuint)wsBytes) : 0;
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


    // ── descriptor builders ─────────────────────────────────────────────
    private unsafe nint Tensor(List<nint> owned, long uid, long* dims, long* strides, int dtype, bool virt, bool byValue = false)
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
        if (byValue)
        {
            byte byVal = 1;
            SetAttr(t, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &byVal);
        }
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


    private static Exception? DestroyPlan(Plan plan)
    {
        Exception? firstError = null;
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
