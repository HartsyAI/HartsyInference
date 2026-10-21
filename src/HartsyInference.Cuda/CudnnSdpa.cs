using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>Fused scaled-dot-product attention via cuDNN's flash-attention engine (backend graph API). Replaces the materialized cuBLAS QKᵀ→softmax→PV path (which allocates a [B·H, Sq, Skv] score matrix) with a single fused kernel that never materializes the scores — ~34× faster at Krea2 self-attention shape (B=1,H=24,S=4608,D=128: 62.7ms → 1.8ms/call, workspace 0), measured on RTX 4090. The graph mirrors what NVIDIA's cudnn-frontend emits for plain inference SDPA on cuDNN ≥ 9.21: bmm1: S = Q @ Kᵀ (matmul, fp32 accum) → scale: Ss = S·attn_scale (pointwise mul) → [bias add] → softmax: P = softmax(Ss) (the UNIFIED backend SOFTMAX op — the decomposed max/sub/exp/sum/div graph does NOT match the fused engine) → bmm2: O = P @ V. Tensors are 4D [B,H,S,D] with fp16 or bf16 I/O and fp32 compute; K/V carry their own head count (grouped-query attention needs no widening). Q/K/V/O and the bias are non-virtual I/O, the scale is a by-value host scalar bound on every call, and S/Ss/P are virtual so the engine keeps them in registers/shared memory. Execution plans are expensive to build (heuristics + JIT) and cheap to run, so they are cached by dtype, head dim and every tensor's extents and strides — not by scale or offset. Instances are created per <see cref="CudaBackend"/> (one cuDNN handle bound to the compute stream).</summary>
internal sealed class CudnnSdpa : IDisposable
{
    private readonly nint _handle;
    private readonly ConcurrentDictionary<PlanKey, Lazy<Plan>> _plans = new();
    private bool _disposed;

    /// <summary>I/O dtype of Q/K/V/O; scores and bias stay fp32 and P follows this dtype.</summary>
    internal enum SdpaDtype { F16, BF16 }

    /// <summary>Elements in the 16 bytes the graph declares for every tensor, at a 16-bit I/O dtype.</summary>
    internal const long AlignElements = 8;

    /// <summary>Element offsets added to the device pointers at execute time; they are not part of the plan.</summary>
    internal readonly record struct Offsets(long Q, long K, long V, long O, long Bias = 0)
    {
        public static Offsets Of(in AttentionCall c) => new(c.Query.Offset, c.Key.Offset, c.Value.Offset, c.Output.Offset);
    }

    /// <summary>One tensor as <c>[B, H, S, D]</c> by element strides; D is the key's head dim and its stride must be 1. For the bias, S is
    /// the query extent (1 for key-only) and D the key count. <c>S == 0</c> is no tensor.</summary>
    internal readonly record struct View(long B, long H, long S, long BatchStride, long HeadStride, long TokenStride, long DimStride = 1)
    {
        public static View HeadMajor(long b, long h, long s, long d) => new(b, h, s, h * s * d, s * d, d);

        public static View TokenMajor(long b, long h, long s, long d) => new(b, h, s, s * h * d, d, h * d);

        public static View Of(in AttentionOperand x) => new(x.Batch, x.Heads, x.Tokens, x.BatchStride, x.HeadStride, x.TokenStride);

        /// <summary>A contiguous fp32 bias of <paramref name="bias"/>'s dims over <paramref name="keys"/> key columns.</summary>
        public static View Bias(CudnnBias bias, long keys) => bias.HasBias
            ? new(bias.B, bias.H, bias.Sq, bias.H * bias.Sq * keys, bias.Sq * keys, keys) : default;

        /// <summary>Whether this is the whole-tensor head-major or token-major view of its own extents at head dim <paramref name="d"/>.</summary>
        public bool IsNatural(long d, bool tokenMajor) => this == (tokenMajor ? TokenMajor(B, H, S, d) : HeadMajor(B, H, S, d));

        /// <summary>What cuDNN can read at head dim <paramref name="d"/>: positive extents, a contiguous head dim, 16-byte aligned strides, and
        /// neighbouring heads and rows that do not overlap a head dim run.</summary>
        public bool Takes(long d) => B > 0 && H > 0 && S > 0 && d > 0 && DimStride == 1 && d % AlignElements == 0
            && BatchStride > 0 && HeadStride > 0 && TokenStride > 0
            && BatchStride % AlignElements == 0 && HeadStride % AlignElements == 0 && TokenStride % AlignElements == 0
            && (H == 1 || HeadStride >= d) && (S == 1 || TokenStride >= d);
    }

    /// <summary>True unless Q, K, V and the output are all whole-tensor head-major or all whole-tensor token-major views at no offset.</summary>
    internal static bool IsStrided(long d, in View q, in View k, in View v, in View o, in Offsets offsets) =>
        offsets != default || !((q.IsNatural(d, false) && k.IsNatural(d, false) && v.IsNatural(d, false) && o.IsNatural(d, false))
            || (q.IsNatural(d, true) && k.IsNatural(d, true) && v.IsNatural(d, true) && o.IsNatural(d, true)));

    /// <summary>Identity of a cached plan: dtype, head dim and every tensor's extents and strides. The scale is bound per call and the
    /// offsets per execute, so neither is in the key. The bias is fp32; no bias is a default <see cref="View"/>.</summary>
    internal readonly record struct PlanKey(SdpaDtype Dtype, long D, View Q, View K, View V, View O, View Bias = default)
    {
        public long B => Q.B;

        public long H => Q.H;

        public long Hkv => K.H;

        public long Sq => Q.S;

        public long Sk => K.S;

        public bool HasBias => Bias.S > 0;

        public CudnnSdpaMask Mask => HasBias ? CudnnSdpaMask.Bias : CudnnSdpaMask.None;

        public static PlanKey Of(SdpaDtype dtype, long d, in View q, in View k, in View v, in View o, CudnnBias bias) =>
            new(dtype, d, q, k, v, o, View.Bias(bias, k.S));

        internal int CudnnDtype => Dtype == SdpaDtype.BF16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_HALF;

        internal void Validate()
        {
            if (B <= 0 || H <= 0 || Hkv <= 0 || Sq <= 0 || Sk <= 0 || D <= 0)
                throw new ArgumentException($"SDPA dims must be positive: {this}.");
            if (H % Hkv != 0)
                throw new ArgumentException($"SDPA K/V heads {Hkv} must divide query heads {H}.");
            if (K.B != B || V.B != B || O.B != B || V.H != Hkv || O.H != H || V.S != Sk || O.S != Sq)
                throw new ArgumentException($"SDPA operands disagree on batch, heads or rows: {this}.");
            if (!Q.Takes(D) || !K.Takes(D) || !V.Takes(D) || !O.Takes(D))
                throw new ArgumentException($"SDPA operands need a contiguous head dim and strides aligned to {AlignElements} elements: {this}.");
            if (HasBias && ((Bias.B != 1 && Bias.B != B) || (Bias.H != 1 && Bias.H != H) || (Bias.S != 1 && Bias.S != Sq)))
                throw new ArgumentException($"SDPA bias [{Bias.B},{Bias.H},{Bias.S},{Sk}] does not broadcast to [{B},{H},{Sq},{Sk}].");
            if (HasBias && Bias.DimStride != 1)
                throw new ArgumentException($"SDPA bias needs a contiguous key axis: {this}.");
        }
    }

    /// <summary>One tensor of the graph: extents and strides as cuDNN is given them.</summary>
    internal readonly record struct GraphTensor(long Uid, long[] Dims, long[] Strides, int Dtype, bool Virtual, bool ByValue = false);

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

    /// <summary>Run fused attention over device buffers of the key's dtype, each viewed by its key strides from the buffer plus its offset in <paramref name="offsets"/>. <paramref name="biasF32"/> (0 = none, matching <see cref="PlanKey.HasBias"/>) is an fp32 additive bias; extents of 1 broadcast, and a bias S of 1 is a key-only bias.</summary>
    public unsafe void Execute(ulong q, ulong k, ulong v, ulong o, in PlanKey key, float scale, ulong biasF32 = 0, in Offsets offsets = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (key.HasBias != (biasF32 != 0))
            throw new ArgumentException("SDPA bias pointer and plan key disagree about whether a bias is present.");
        key.Validate();
        if (offsets.Q < 0 || offsets.K < 0 || offsets.V < 0 || offsets.O < 0 || offsets.Bias < 0
            || offsets.Q % AlignElements != 0 || offsets.K % AlignElements != 0 || offsets.V % AlignElements != 0 || offsets.O % AlignElements != 0
            || offsets.Bias % (AlignElements / 2) != 0)
            throw new ArgumentException($"SDPA offsets must be non-negative and 16-byte aligned: {offsets}.");
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
            const ulong halfBytes = sizeof(ushort), floatBytes = sizeof(float);
            void** ptrs = stackalloc void*[6]
            {
                (void*)(q + (ulong)offsets.Q * halfBytes), (void*)(k + (ulong)offsets.K * halfBytes), (void*)(v + (ulong)offsets.V * halfBytes),
                (void*)(o + (ulong)offsets.O * halfBytes), &scaleValue, (void*)(biasF32 == 0 ? 0 : biasF32 + (ulong)offsets.Bias * floatBytes)
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
            Dictionary<long, nint> tensors = new();
            foreach (GraphTensor t in GraphTensors(key))
                tensors[t.Uid] = Tensor(owned, t);
            nint tQ = tensors[UidQ], tKt = tensors[UidK], tV = tensors[UidV], tO = tensors[UidO], tScale = tensors[UidScale];
            nint tS = tensors[UidS], tSS = tensors[UidSS], tP = tensors[UidP];

            // ── ops: bmm1 → scale → [+ bias] → softmax → bmm2 ──
            nint* ops = stackalloc nint[5];
            long opCount = 0;
            ops[opCount++] = MatmulOp(owned, tQ, tKt, tS);
            ops[opCount++] = PointwiseOp(owned, CUDNN_POINTWISE_MUL, tS, tScale, tSS);
            nint softmaxIn = tSS;
            if (key.HasBias)
            {
                // -1e30 masked entries are safe: the add runs in fp32 and softmax subtracts the row max (fully masked rows go uniform).
                nint tSB = tensors[UidSB];
                ops[opCount++] = PointwiseOp(owned, CUDNN_POINTWISE_ADD, tSS, tensors[UidBias], tSB);
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
                + (key.HasBias ? $" bias=[{key.Bias.B},{key.Bias.H},{key.Bias.S},{sk}]" : "")
                + $" strides=q{Strides(key.Q)}k{Strides(key.K)}v{Strides(key.V)}o{Strides(key.O)}";
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


    /// <summary>The graph's tensors in creation order. Kᵀ views the [b,hKv,sk,d] buffer as [b,hKv,d,sk] by swapping the last two dims and strides.</summary>
    internal static GraphTensor[] GraphTensors(in PlanKey key)
    {
        long b = key.B, h = key.H, hKv = key.Hkv, sq = key.Sq, sk = key.Sk, d = key.D;
        int io = key.CudnnDtype;
        View q = key.Q, k = key.K, v = key.V, o = key.O;
        long[] sDim = [b, h, sq, sk], sStr = [h * sq * sk, sq * sk, sk, 1], one = [1, 1, 1, 1];
        List<GraphTensor> t =
        [
            new(UidQ, [b, h, sq, d], [q.BatchStride, q.HeadStride, q.TokenStride, q.DimStride], io, false),
            new(UidK, [b, hKv, d, sk], [k.BatchStride, k.HeadStride, k.DimStride, k.TokenStride], io, false),
            new(UidV, [b, hKv, sk, d], [v.BatchStride, v.HeadStride, v.TokenStride, v.DimStride], io, false),
            new(UidO, [b, h, sq, d], [o.BatchStride, o.HeadStride, o.TokenStride, o.DimStride], io, false),
            new(UidScale, one, one, CUDNN_DATA_FLOAT, false, ByValue: true),
            new(UidS, sDim, sStr, CUDNN_DATA_FLOAT, true),
            new(UidSS, sDim, sStr, CUDNN_DATA_FLOAT, true),
            new(UidP, sDim, sStr, io, true),
        ];
        if (key.HasBias)
        {
            View bias = key.Bias;
            t.Add(new(UidBias, [bias.B, bias.H, bias.S, sk], [bias.BatchStride, bias.HeadStride, bias.TokenStride, bias.DimStride], CUDNN_DATA_FLOAT, false));
            t.Add(new(UidSB, sDim, sStr, CUDNN_DATA_FLOAT, true));
        }
        return [.. t];
    }

    private static string Strides(in View v) => $"[{v.BatchStride},{v.HeadStride},{v.TokenStride}]";

    // ── descriptor builders ─────────────────────────────────────────────
    private unsafe nint Tensor(List<nint> owned, in GraphTensor g)
    {
        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, out nint t), "tensor create");
        owned.Add(t);
        int dt = g.Dtype;
        SetAttr(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
        fixed (long* dims = g.Dims)
            SetAttr(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims);
        fixed (long* strides = g.Strides)
            SetAttr(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides);
        long id = g.Uid;
        SetAttr(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id);
        long align = 16;
        SetAttr(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
        byte v = (byte)(g.Virtual ? 1 : 0);
        SetAttr(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &v);
        if (g.ByValue)
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
