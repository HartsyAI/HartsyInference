using System.Buffers;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Mixture-of-Experts feed-forward block: a router picks the top-k experts per token, each selected expert (a SwiGLU FFN) runs only on the tokens routed to it (gather → expert GEMM → weighted scatter-add), plus an optional always-on shared expert applied to every token, replacing the dense SwiGLU on MoE layers. Covers Qwen2-MoE/Qwen3-MoE/Mixtral (softmax routing) and DeepSeek (sigmoid routing); experts run through the same quant-aware <see cref="GenericTransformer.Project"/> path as the dense FFN.</summary>
/// <remarks><b>Routing residency:</b> the router logits are read to the host to pick top-k experts and build the per-expert token groups (one D2H per MoE layer); the expert GEMMs and the gather/scatter combine stay device-resident. A fully on-device top-k + dispatch is a follow-up; correctness is unaffected.</remarks>
public sealed class MoeFeedForward(MoeConfig moe, int hiddenSize, bool lowVram)
{
    private readonly MoeConfig _moe = moe;
    private readonly int _hidden = hiddenSize;
    private readonly bool _lowVram = lowVram;

    private Tensor _routerW = null!;                 // [E, hidden]
    private Tensor[] _gateW = null!, _upW = null!, _downW = null!;   // per expert
    private Tensor? _shGateW, _shUpW, _shDownW, _shGateScoreW;       // shared expert (optional)
    private float[]? _correctionBias;                // DeepSeek-V3 e_score_correction_bias [E] (selection only)

    /// <summary>Largest batch the device-resident routed stage serves: every (token, slot) pair reads its expert's rows once, which only
    /// beats the grouped per-expert path while few tokens share an expert.</summary>
    internal const int IndexedMaxTokens = 16;

    private bool? _indexedShapeOk;                   // dtype/shape/routing eligibility, fixed once the weights are loaded
    private bool? _groupedShapeOk;
    private int[] _groupedOffsets = [];              // host copy of the dispatch offsets, reused across calls

    // Opt-in heterogeneous runtime state (CPU-only slice). Created on first use; the direct path never touches it.
    private static readonly ForcedPlacementPolicy CpuOnlyPolicy = new(static _ => ExpertPlacement.Cpu);
    private HostExpertCache? _hostCache;
    private F32ExpertWeights?[]? _hostWeights;
    private ExpertAssignment[]? _lastPlan;
    private int _lastPlanCount;

    // Expert offload (attached by the engine when the placement planner chose it). Scratch is reused from call to call.
    private MoeExpertOffload? _offload;
    private int _offloadLayer = -1;
    private int[] _offIds = [];
    private int[] _offCounts = [];
    private bool[] _offResident = [];
    private ExpertKey[] _offKeys = [];
    private ExpertAssignment[] _offPlan = [];
    private List<ExpertKey> _offMisses = [];
    private int[] _offHostPlan = [];
    private int[] _offHostOffset = [];
    private readonly ExpertLease _offLease = new();

    /// <summary>
    /// Routes this layer's experts through <paramref name="offload"/>: the experts its device cache holds run on the device as before,
    /// and the others run on the CPU, or on the device from a copy uploaded for one large batch. Registers the layer's experts with
    /// the offload's cache. A later <see cref="LoadWeights"/> detaches it, since the cache would hold the old weights.
    /// </summary>
    /// <param name="offload">The model's offload.</param>
    /// <param name="layer">This block's layer index, the bank layer its experts are keyed by.</param>
    public void AttachOffload(MoeExpertOffload offload, int layer)
    {
        ArgumentNullException.ThrowIfNull(offload);
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        int e = _moe.NumExperts;
        offload.RegisterLayer(layer, e, ResolveExpert);
        _offIds = new int[e * Math.Max(1, _moe.NumExpertsPerTok)];
        _offCounts = new int[e];
        _offResident = new bool[e];
        _offKeys = new ExpertKey[e];
        _offPlan = new ExpertAssignment[e];
        _offMisses = new List<ExpertKey>(e);
        _offHostPlan = new int[e];
        _offHostOffset = new int[e];
        _offloadLayer = layer;
        _offload = offload;
    }

    /// <summary>Where this block's routed experts run: the offload attached to it, or null when every expert runs on the device.</summary>
    public MoeExpertOffload? Offload => _offload;

    /// <summary>
    /// Opt-in (default off): routes the routed experts through the heterogeneous runtime. The scheduler plans every expert on
    /// the CPU, <see cref="HostExpertCache"/> holds the layer's F32 weights without copying them, and
    /// <see cref="HeterogeneousExpertExecutor"/> runs the F32 reference. The combine is the same weighted scatter-add as the
    /// direct path. Requires F32 expert weights.
    /// </summary>
    public bool UseHostExpertRuntime { get; set; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // The opt-in runtime caches the layer's weights on first use; a reload must not keep serving the old copies.
        _hostCache?.Dispose();
        _hostCache = null;
        _hostWeights = null;
        _offload = null;
        _offloadLayer = -1;
        _routerW = w[$"{prefix}.mlp.gate.weight"];
        // DeepSeek-V3 / Kimi-K2 router correction bias (added to the selection scores only). Optional: V2-Lite and
        // every softmax-routed MoE lack it. Read once to a host array (it is a tiny [E] vector used per token).
        // llama.cpp keeps router biases F32, so a direct host copy is exact.
        if (w.TryGetValue($"{prefix}.mlp.gate.e_score_correction_bias", out Tensor? cb))
        {
            if (cb.DType != DType.F32) throw new NotSupportedException("MoE correction bias must be F32.");
            _correctionBias = HostCopy(cb, _moe.NumExperts);
        }
        else if (_moe.Scoring == MoeScoring.SigmoidLogitAdd)
        {
            throw new InvalidDataException("SigmoidLogitAdd routing requires an e_score_correction_bias tensor.");
        }
        int e = _moe.NumExperts;
        _gateW = new Tensor[e];
        _upW = new Tensor[e];
        _downW = new Tensor[e];
        for (int i = 0; i < e; i++)
        {
            string ep = $"{prefix}.mlp.experts.{i}";
            _gateW[i] = w[$"{ep}.gate_proj.weight"];
            _upW[i] = w[$"{ep}.up_proj.weight"];
            _downW[i] = w[$"{ep}.down_proj.weight"];
        }
        if (_moe.SharedExpertIntermediateSize > 0)
        {
            _shGateW = w[$"{prefix}.mlp.shared_expert.gate_proj.weight"];
            _shUpW = w[$"{prefix}.mlp.shared_expert.up_proj.weight"];
            _shDownW = w[$"{prefix}.mlp.shared_expert.down_proj.weight"];
            // The shared-expert sigmoid gate ([1, hidden]) is optional (Qwen2-MoE has it; some variants don't).
            w.TryGetValue($"{prefix}.mlp.shared_expert_gate.weight", out Tensor? sg);
            _shGateScoreW = sg;
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _routerW;
        for (int i = 0; i < _gateW.Length; i++) { yield return _gateW[i]; yield return _upW[i]; yield return _downW[i]; }
        if (_shGateW is not null) { yield return _shGateW; yield return _shUpW!; yield return _shDownW!; }
        if (_shGateScoreW is not null) yield return _shGateScoreW;
    }

    /// <summary>The routed experts' projections as three groups, every expert's gate, then up, then down, each in expert
    /// order. The members are the tensors <see cref="EnumerateWeights"/> yields, grouped so a backend can place each
    /// projection's experts in one device allocation (<see cref="IBackend.PreloadWeightGroups"/>).</summary>
    public IEnumerable<IReadOnlyList<Tensor>> EnumerateExpertGroups()
    {
        yield return _gateW;
        yield return _upW;
        yield return _downW;
    }

    /// <summary>Router logits (<c>x @ router_weight</c>) as their own <c>[1, N, E]</c> tensor, for callers that need to compute them from a DIFFERENT input than the expert FFN's (Gemma-4: the router reads a separately normalized view of the attention output) before passing them to <see cref="Forward"/>.</summary>
    public Tensor ComputeRouterLogits(IBackend backend, Tensor x, int n)
    {
        Tensor logits = new(new TensorShape(1, n, _moe.NumExperts), DType.F32);
        GenericTransformer.Project(backend, logits, x, _routerW, null, lowVram: false);
        return logits;
    }

    /// <summary>MoE FFN: <paramref name="x"/> <c>[1, N, hidden]</c> → <c>[1, N, hidden]</c>; <paramref name="routerLogits"/> overrides the internal <c>x @ router_weight</c> projection when provided (Gemma-4: the router reads a differently-normalized view of the attention output, computed by the caller) — same <c>[1, N, E]</c> shape either way, not disposed by this call.</summary>
    public unsafe Tensor Forward(IBackend backend, Tensor x, int n, Tensor? routerLogits = null)
    {
        int e = _moe.NumExperts;
        int topK = _moe.NumExpertsPerTok;
        int inter = _moe.MoeIntermediateSize;

        if (n <= IndexedMaxTokens && CanRunIndexed(backend)) return ForwardIndexed(backend, x, n, routerLogits);
        if (n > IndexedMaxTokens && CanRunGrouped(backend)) return ForwardGrouped(backend, x, n, routerLogits);

        // 1. Router logits, read to host for top-k selection.
        Tensor? ownRouterLogits = null;
        Tensor routerLogitsT = routerLogits ?? (ownRouterLogits = new(new TensorShape(1, n, e), DType.F32));
        if (routerLogits is null) GenericTransformer.Project(backend, routerLogitsT, x, _routerW, null, lowVram: false);
        float[] logits = new float[(long)n * e];
        fixed (float* dst = logits)
        {
            float* src = (float*)routerLogitsT.DataPointer;   // D2H sync on CUDA
            Buffer.MemoryCopy(src, dst, logits.Length * 4L, logits.Length * 4L);
        }
        ownRouterLogits?.Dispose();

        // 2. Per-token top-k routing → per-expert token groups + weights.
        List<int>[] expertTokens = new List<int>[e];
        List<float>[] expertWeights = new List<float>[e];
        for (int i = 0; i < e; i++) { expertTokens[i] = []; expertWeights[i] = []; }
        if (_moe.ExpertGroupCount > 0)
            RouteGroupLimited(logits, n, e, topK, expertTokens, expertWeights);
        else
            Route(logits, n, e, topK, expertTokens, expertWeights);

        // 3. Output accumulator, seeded with the shared expert (if any).
        Tensor output = new(new TensorShape(1, n, _hidden), DType.F32);
        backend.Fill(output, 0f);
        if (_shGateW is not null)
        {
            Tensor shared = SwiGlu(backend, x, n, _shGateW, _shUpW!, _shDownW!, _moe.SharedExpertIntermediateSize);
            int[] identity = new int[n];
            for (int i = 0; i < n; i++) identity[i] = i;
            float[] gateVals;
            if (_shGateScoreW is not null)
            {
                // sigmoid(x @ shared_expert_gate) scales the shared expert per token. The gate is a per-token
                // scalar already going to the host (as a scatter scale), so apply the sigmoid there — no need
                // for a device Sigmoid op.
                Tensor g = new(new TensorShape(1, n, 1), DType.F32);
                GenericTransformer.Project(backend, g, x, _shGateScoreW, null, lowVram: false);
                gateVals = HostCopy(g, n);
                g.Dispose();
                for (int i = 0; i < n; i++) gateVals[i] = 1f / (1f + MathF.Exp(-gateVals[i]));
            }
            else
            {
                gateVals = new float[n];
                Array.Fill(gateVals, 1f);
            }
            // Identity scatter folds the per-token gate in and seeds the accumulator in one device op.
            backend.ScatterAddWeightedRows(output, shared, identity, gateVals);
            shared.Dispose();
        }

        // 4. Routed experts: gather → expert SwiGLU → weighted scatter-add.
        if (_offload is not null)
        {
            RunRoutedWithOffload(backend, x, n, output, expertTokens, expertWeights);
            return output;
        }
        if (UseHostExpertRuntime)
        {
            RunRoutedThroughRuntime(backend, x, n, output, expertTokens, expertWeights);
            return output;
        }
        for (int ex = 0; ex < e; ex++)
        {
            int ne = expertTokens[ex].Count;
            if (ne == 0) continue;
            int[] idx = [.. expertTokens[ex]];
            float[] wts = [.. expertWeights[ex]];

            Tensor gathered = new(new TensorShape(1, ne, _hidden), DType.F32);
            backend.GatherRows(gathered, x, idx);
            Tensor expOut = SwiGlu(backend, gathered, ne, _gateW[ex], _upW[ex], _downW[ex], inter);
            gathered.Dispose();
            backend.ScatterAddWeightedRows(output, expOut, idx, wts);
            expOut.Dispose();
        }
        return output;
    }

    /// <summary>Routed experts under expert offload. The plan pins the experts the device cache holds; each runs on the device exactly as
    /// the direct path runs it, its projections finding the cache's copies. A miss serving a large batch runs the same way from a copy
    /// uploaded for this call. The remaining misses run on the CPU: their rows are gathered on the device and read back in one copy
    /// BEFORE any device expert is queued, since a host read waits for the queue; the device experts are then queued and the CPU
    /// experts computed while they run. Each CPU expert's output is combined with its own weighted scatter-add, as on the direct path:
    /// the combine kernel requires distinct rows within one call.</summary>
    private unsafe void RunRoutedWithOffload(IBackend backend, Tensor x, int n, Tensor output, List<int>[] expertTokens,
        List<float>[] expertWeights)
    {
        MoeExpertOffload offload = _offload!;
        int e = _moe.NumExperts;
        int h = _hidden;
        int pairs = 0;
        for (int ex = 0; ex < e; ex++) pairs += expertTokens[ex].Count;
        if (_offIds.Length < pairs) _offIds = new int[Math.Max(pairs, _offIds.Length * 2)];
        int fill = 0;
        for (int ex = 0; ex < e; ex++)
            for (int j = 0; j < expertTokens[ex].Count; j++) _offIds[fill++] = ex;

        int distinct = ExpertScheduler.Plan(offload.Cache, _offIds.AsSpan(0, pairs), _offloadLayer, 0, e, offload.Policy, _offCounts,
            _offResident, _offKeys, _offPlan, _offMisses, _offLease);
        float[]? hostIn = null, hostOut = null;
        int[]? hostIndex = null;
        try
        {
            // Host experts: those planned on the CPU that serve too few rows to be worth an upload.
            int hostExperts = 0, hostRows = 0;
            for (int i = 0; i < distinct; i++)
            {
                if (_offPlan[i].Placement == ExpertPlacement.Cpu && !offload.Streams(_offPlan[i].Rows))
                {
                    hostExperts++;
                    hostRows += _offPlan[i].Rows;
                }
            }

            // 1. Gather every host row on the device and read it back once, before the device experts are queued.
            if (hostExperts > 0)
            {
                hostIndex = ArrayPool<int>.Shared.Rent(hostRows);
                int at = 0;
                for (int i = 0; i < distinct; i++)
                {
                    if (_offPlan[i].Placement != ExpertPlacement.Cpu || offload.Streams(_offPlan[i].Rows)) continue;
                    foreach (int token in expertTokens[_offPlan[i].Key.Expert]) hostIndex[at++] = token;
                }
                using Tensor gathered = new(new TensorShape(1, hostRows, h), DType.F32);
                backend.GatherRows(gathered, x, hostIndex.AsSpan(0, hostRows));
                hostIn = ArrayPool<float>.Shared.Rent(hostRows * h);
                hostOut = ArrayPool<float>.Shared.Rent(hostRows * h);
                gathered.AsReadOnlySpan<float>().CopyTo(hostIn);   // D2H sync, with nothing else queued yet
            }

            // 2. Device experts: resident ones, and misses large enough to stream.
            int inter = _moe.MoeIntermediateSize;
            for (int i = 0; i < distinct; i++)
            {
                ExpertAssignment assignment = _offPlan[i];
                bool streamed = assignment.Placement == ExpertPlacement.Cpu && offload.Streams(assignment.Rows);
                if (assignment.Placement != ExpertPlacement.Gpu && !streamed) continue;
                int ex = assignment.Key.Expert;
                ReadOnlySpan<int> idx = CollectionsMarshal.AsSpan(expertTokens[ex]);
                Tensor gatheredRows = new(new TensorShape(1, idx.Length, h), DType.F32);
                backend.GatherRows(gatheredRows, x, idx);
                Tensor expOut = SwiGlu(backend, gatheredRows, idx.Length, _gateW[ex], _upW[ex], _downW[ex], inter);
                gatheredRows.Dispose();
                backend.ScatterAddWeightedRows(output, expOut, idx, CollectionsMarshal.AsSpan(expertWeights[ex]));
                expOut.Dispose();
                offload.Record(assignment.Placement, streamed, assignment.Rows);
            }

            // 3. Host experts, computed while the device runs, then combined one expert at a time.
            if (hostExperts > 0)
            {
                ExpertProgram program = _moe.Activation == ActivationKind.GeluTanh ? ExpertProgram.GeGlu : ExpertProgram.Swiglu;
                int[] hostPlan = _offHostPlan;
                int[] hostOffset = _offHostOffset;
                int k = 0, rowAt = 0;
                for (int i = 0; i < distinct; i++)
                {
                    if (_offPlan[i].Placement != ExpertPlacement.Cpu || offload.Streams(_offPlan[i].Rows)) continue;
                    hostPlan[k] = i;
                    hostOffset[k] = rowAt;
                    rowAt += _offPlan[i].Rows;
                    k++;
                }
                float[] inBuf = hostIn!, outBuf = hostOut!;
                ExpertAssignment[] plan = _offPlan;
                IExpertHostRunner runner = offload.Host;
                // One expert at a time: the runner spreads each expert's rows across every core, which a layer's three or four
                // single-row experts at decode could not fill on their own.
                for (int j = 0; j < hostExperts; j++)
                {
                    ExpertAssignment a = plan[hostPlan[j]];
                    int start = hostOffset[j] * h, length = a.Rows * h;
                    runner.Run(program, a.Key, inBuf.AsSpan(start, length), a.Rows, outBuf.AsSpan(start, length));
                }
                // One host tensor carries every CPU expert's output; each expert's rows are a view of it, combined on their own.
                using Tensor allOut = new(new TensorShape(hostRows, h), DType.F32);
                outBuf.AsSpan(0, hostRows * h).CopyTo(allOut.AsSpan<float>());
                for (int j = 0; j < hostExperts; j++)
                {
                    ExpertAssignment a = plan[hostPlan[j]];
                    int ex = a.Key.Expert;
                    using Tensor expOut = allOut.SliceRows(hostOffset[j], a.Rows);
                    backend.ScatterAddWeightedRows(output, expOut, CollectionsMarshal.AsSpan(expertTokens[ex]),
                        CollectionsMarshal.AsSpan(expertWeights[ex]));
                    offload.Record(ExpertPlacement.Cpu, streamed: false, a.Rows);
                }
            }
        }
        finally
        {
            _offLease.Dispose();
            if (hostIndex is not null) ArrayPool<int>.Shared.Return(hostIndex);
            if (hostIn is not null) ArrayPool<float>.Shared.Return(hostIn);
            if (hostOut is not null) ArrayPool<float>.Shared.Return(hostOut);
        }
        offload.AfterLayer(_offMisses);
    }


    /// <summary>True when the routed stage can run entirely on <paramref name="backend"/>: the experts' quant type has an indexed GEMV, the routing is
    /// one the device router reproduces, and no host runtime or offload owns the experts.</summary>
    internal bool CanRunIndexed(IBackend backend)
    {
        if (!HartsyInference.Core.Configuration.EngineKnobs.MoeIndexed.Value) return false;
        if (UseHostExpertRuntime || _offload is not null) return false;
        if (_indexedShapeOk is null) _indexedShapeOk = IndexedShapeOk();
        return _indexedShapeOk.Value && backend.SupportsMoeExpertIndexed(_gateW[0].DType) && backend.SupportsMoeExpertIndexed(_downW[0].DType)
            && backend.MoeExpertsResident(_gateW, _upW, _downW);
    }

    private bool IndexedShapeOk()
    {
        // Group-limited and logit-biased routing keep the host router.
        if (_moe.ExpertGroupCount > 0 || _moe.Scoring == MoeScoring.SigmoidLogitAdd) return false;
        if (_moe.NumExperts > MoeRouteArgs.MaxExperts || _moe.NumExpertsPerTok > _moe.NumExperts) return false;
        if (_gateW.Length != _moe.NumExperts || _upW.Length != _moe.NumExperts || _downW.Length != _moe.NumExperts) return false;
        DType gateType = _gateW[0].DType, downType = _downW[0].DType;
        if (!gateType.IsQuantized || !downType.IsQuantized || _upW[0].DType != gateType) return false;
        for (int i = 1; i < _gateW.Length; i++)
        {
            if (_gateW[i].DType != gateType || _upW[i].DType != gateType || _downW[i].DType != downType) return false;
        }
        int align = gateType == DType.Q8_0 ? 32 : 256;
        int downAlign = downType == DType.Q8_0 ? 32 : 256;
        return _hidden % align == 0 && _moe.MoeIntermediateSize % downAlign == 0
            && _gateW[0].Shape[0] == _moe.MoeIntermediateSize && _gateW[0].Shape[1] == _hidden
            && _downW[0].Shape[0] == _hidden && _downW[0].Shape[1] == _moe.MoeIntermediateSize;
    }

    /// <summary>True when a large batch can take the device-routed grouped path: routing the device router reproduces, one quantized type per
    /// projection, and a backend that runs the grouped expert GEMMs for those types.</summary>
    internal bool CanRunGrouped(IBackend backend)
    {
        if (!HartsyInference.Core.Configuration.EngineKnobs.MoeIndexed.Value) return false;
        if (UseHostExpertRuntime || _offload is not null) return false;
        if (_groupedShapeOk is null) _groupedShapeOk = GroupedShapeOk();
        return _groupedShapeOk.Value && backend.SupportsMoeExpertsGrouped(_gateW[0].DType) && backend.SupportsMoeExpertsGrouped(_downW[0].DType)
            && backend.MoeExpertsResident(_gateW, _upW, _downW);
    }

    private bool GroupedShapeOk()
    {
        if (_moe.ExpertGroupCount > 0 || _moe.Scoring == MoeScoring.SigmoidLogitAdd) return false;
        if (_moe.NumExperts > MoeRouteArgs.MaxExperts || _moe.NumExpertsPerTok > _moe.NumExperts) return false;
        if (_gateW.Length != _moe.NumExperts || _upW.Length != _moe.NumExperts || _downW.Length != _moe.NumExperts) return false;
        DType gateType = _gateW[0].DType, downType = _downW[0].DType;
        if (!gateType.IsQuantized || !downType.IsQuantized || _upW[0].DType != gateType) return false;
        for (int i = 1; i < _gateW.Length; i++)
        {
            if (_gateW[i].DType != gateType || _upW[i].DType != gateType || _downW[i].DType != downType) return false;
        }
        return _gateW[0].Shape[0] == _moe.MoeIntermediateSize && _gateW[0].Shape[1] == _hidden
            && _downW[0].Shape[0] == _hidden && _downW[0].Shape[1] == _moe.MoeIntermediateSize;
    }

    /// <summary>The routed stage of a large batch with the routing on the device: <see cref="IBackend.MoeRoute"/>, <see cref="IBackend.MoeBuildDispatch"/>
    /// into expert-major order, one read of the E + 1 offsets, the per-expert GEMMs over contiguous row ranges
    /// (<see cref="IBackend.MoeExpertsGrouped"/>), and one combine that folds in the shared expert.</summary>
    private Tensor ForwardGrouped(IBackend backend, Tensor x, int n, Tensor? routerLogits)
    {
        int e = _moe.NumExperts;
        int k = _moe.NumExpertsPerTok;
        Tensor? ownLogits = null;
        Tensor logits = routerLogits ?? (ownLogits = new(new TensorShape(1, n, e), DType.F32));
        if (routerLogits is null) GenericTransformer.Project(backend, logits, x, _routerW, null, lowVram: false);

        Tensor topkIdx = new(new TensorShape(n, k), DType.I32);
        Tensor topkWeight = new(new TensorShape(n, k), DType.F32);
        backend.MoeRoute(topkIdx, topkWeight, logits, RouteArgs);
        ownLogits?.Dispose();

        Tensor counts = new(new TensorShape(e), DType.I32);
        Tensor offsets = new(new TensorShape(e + 1), DType.I32);
        Tensor perm = new(new TensorShape(n * k), DType.I32);
        Tensor pairSlot = new(new TensorShape(n * k), DType.I32);
        backend.MoeBuildDispatch(counts, offsets, perm, pairSlot, topkIdx, e);
        topkIdx.Dispose();
        counts.Dispose();
        if (_groupedOffsets.Length != e + 1) _groupedOffsets = new int[e + 1];
        offsets.AsReadOnlySpan<int>().CopyTo(_groupedOffsets);   // the layer's one device-to-host read
        offsets.Dispose();

        int rows = _groupedOffsets[e];
        Tensor expertOut = new(new TensorShape(1, rows, _hidden), DType.F32);
        backend.MoeExpertsGrouped(expertOut, x, perm, _groupedOffsets, _gateW, _upW, _downW, gelu: _moe.Activation == ActivationKind.GeluTanh);
        perm.Dispose();

        Tensor? shared = null, sharedGate = null;
        if (_shGateW is not null)
        {
            shared = SwiGlu(backend, x, n, _shGateW, _shUpW!, _shDownW!, _moe.SharedExpertIntermediateSize);
            if (_shGateScoreW is not null)
            {
                sharedGate = new(new TensorShape(1, n, 1), DType.F32);
                GenericTransformer.Project(backend, sharedGate, x, _shGateScoreW, null, lowVram: false);
            }
        }
        Tensor output = new(new TensorShape(1, n, _hidden), DType.F32);
        backend.MoeCombinePairs(output, expertOut, pairSlot, topkWeight, shared, sharedGate, k);
        expertOut.Dispose();
        pairSlot.Dispose();
        topkWeight.Dispose();
        shared?.Dispose();
        sharedGate?.Dispose();
        return output;
    }

    private MoeRouteArgs RouteArgs => new(_moe.NumExperts, _moe.NumExpertsPerTok,
        _moe.Scoring == MoeScoring.Softmax ? MoeRouteScoring.Softmax : MoeRouteScoring.Sigmoid,
        Renormalize: _moe.NormTopKProb, Scale: _moe.RoutedScalingFactor);

    /// <summary>The routed stage with nothing leaving the device: router logits, <see cref="IBackend.MoeRoute"/>, the expert-indexed gate/up and down
    /// GEMVs, then one combine that also folds in the shared expert. Capturable in a CUDA graph.</summary>
    private Tensor ForwardIndexed(IBackend backend, Tensor x, int n, Tensor? routerLogits)
    {
        int e = _moe.NumExperts;
        int k = _moe.NumExpertsPerTok;
        Tensor? ownLogits = null;
        Tensor logits = routerLogits ?? (ownLogits = new(new TensorShape(1, n, e), DType.F32));
        if (routerLogits is null) GenericTransformer.Project(backend, logits, x, _routerW, null, lowVram: false);

        Tensor topkIdx = new(new TensorShape(n, k), DType.I32);
        Tensor topkWeight = new(new TensorShape(n, k), DType.F32);
        backend.MoeRoute(topkIdx, topkWeight, logits, RouteArgs);
        ownLogits?.Dispose();

        int rows = n * k;
        Tensor act = new(new TensorShape(1, rows, _moe.MoeIntermediateSize), DType.F32);
        backend.MoeExpertGateUp(act, x, _gateW, _upW, topkIdx, k, gelu: _moe.Activation == ActivationKind.GeluTanh);
        Tensor slotOut = new(new TensorShape(1, rows, _hidden), DType.F32);
        backend.MoeExpertDown(slotOut, act, _downW, topkIdx, k);
        act.Dispose();
        topkIdx.Dispose();

        Tensor? shared = null, sharedGate = null;
        if (_shGateW is not null)
        {
            shared = SwiGlu(backend, x, n, _shGateW, _shUpW!, _shDownW!, _moe.SharedExpertIntermediateSize);
            if (_shGateScoreW is not null)
            {
                sharedGate = new(new TensorShape(1, n, 1), DType.F32);
                GenericTransformer.Project(backend, sharedGate, x, _shGateScoreW, null, lowVram: false);
            }
        }
        Tensor output = new(new TensorShape(1, n, _hidden), DType.F32);
        backend.MoeCombineSlots(output, slotOut, topkWeight, shared, sharedGate, k);
        slotOut.Dispose();
        topkWeight.Dispose();
        shared?.Dispose();
        sharedGate?.Dispose();
        return output;
    }

    /// <summary>Gated FFN over <paramref name="rows"/> tokens: down(act(gate(x)) * up(x)) — SiLU (SwiGLU, the default) or tanh-GELU (GeGLU, Gemma-4's <see cref="MoeConfig.Activation"/>).</summary>
    private Tensor SwiGlu(IBackend backend, Tensor x, int rows, Tensor gateW, Tensor upW, Tensor downW, int inter)
    {
        TensorShape ff = new(1, rows, inter);
        Tensor gate = new(ff, DType.F32);
        GenericTransformer.Project(backend, gate, x, gateW, null, _lowVram);
        Tensor act = new(ff, DType.F32);
        if (_moe.Activation == ActivationKind.GeluTanh) backend.Gelu(act, gate);
        else backend.Silu(act, gate);
        gate.Dispose();
        Tensor up = new(ff, DType.F32);
        GenericTransformer.Project(backend, up, x, upW, null, _lowVram);
        Tensor comb = new(ff, DType.F32);
        backend.Mul(comb, act, up);
        act.Dispose(); up.Dispose();
        Tensor outp = new(new TensorShape(1, rows, _hidden), DType.F32);
        GenericTransformer.Project(backend, outp, comb, downW, null, _lowVram);
        comb.Dispose();
        return outp;
    }

    /// <summary>Per-token top-k expert selection + routing weights (softmax, sigmoid, or Kolibri's biased-logit
    /// selection with unbiased sigmoid weights; optional renorm).</summary>
    internal void Route(float[] logits, int n, int e, int topK, List<int>[] expertTokens, List<float>[] expertWeights)
    {
        float[] selection = new float[e];
        float[] weight = new float[e];
        int[] pick = new int[topK];
        for (int t = 0; t < n; t++)
        {
            long baseOff = (long)t * e;
            if (_moe.Scoring == MoeScoring.Softmax)
            {
                float max = float.NegativeInfinity;
                for (int i = 0; i < e; i++) max = MathF.Max(max, logits[baseOff + i]);
                float sum = 0f;
                for (int i = 0; i < e; i++) { float v = MathF.Exp(logits[baseOff + i] - max); weight[i] = v; sum += v; }
                for (int i = 0; i < e; i++) selection[i] = weight[i] /= sum;
            }
            else
            {
                for (int i = 0; i < e; i++)
                {
                    float logit = logits[baseOff + i];
                    weight[i] = 1f / (1f + MathF.Exp(-logit));
                    selection[i] = _moe.Scoring == MoeScoring.SigmoidLogitAdd
                        ? logit + _correctionBias![i]
                        : weight[i];
                }
            }

            // Top-k by score (k is small; a partial selection scan is fine).
            float wsum = 0f;
            for (int kk = 0; kk < topK; kk++)
            {
                int best = -1;
                float bestVal = float.NegativeInfinity;
                for (int i = 0; i < e; i++)
                {
                    bool already = false;
                    for (int j = 0; j < kk; j++) if (pick[j] == i) { already = true; break; }
                    if (already) continue;
                    if (selection[i] > bestVal) { bestVal = selection[i]; best = i; }
                }
                pick[kk] = best;
                wsum += weight[best];
            }
            for (int kk = 0; kk < topK; kk++)
            {
                int ex = pick[kk];
                float wt = weight[ex];
                if (_moe.NormTopKProb) wt /= wsum;
                // llama.cpp scales the routed weights by w_scale unconditionally (1.0 is a no-op for Qwen/Mixtral).
                wt *= _moe.RoutedScalingFactor;
                expertTokens[ex].Add(t);
                expertWeights[ex].Add(wt);
            }
        }
    }

    /// <summary>Routed experts through the heterogeneous runtime. Plans on the CPU, gathers the routed rows expert-major, runs the
    /// F32 reference, then combines each expert's rows with the same weighted scatter-add the direct path uses.</summary>
    private void RunRoutedThroughRuntime(IBackend backend, Tensor x, int n, Tensor output, List<int>[] expertTokens,
        List<float>[] expertWeights)
    {
        int e = _moe.NumExperts;
        int h = _hidden;
        HostExpertCache cache = EnsureHostRuntime();

        // The planner counts (token, slot) pairs per expert, so one id per routed pair is enough.
        int pairs = 0;
        for (int ex = 0; ex < e; ex++) pairs += expertTokens[ex].Count;
        int[] ids = new int[pairs];
        int fill = 0;
        for (int ex = 0; ex < e; ex++)
            for (int j = 0; j < expertTokens[ex].Count; j++) ids[fill++] = ex;

        ExpertAssignment[] plan = new ExpertAssignment[e];
        int[] countScratch = new int[e];
        bool[] residentScratch = new bool[e];
        ExpertKey[] keyScratch = new ExpertKey[e];
        List<ExpertKey> misses = new(e);
        using ExpertLease lease = new();
        int distinct = ExpertScheduler.Plan(cache, ids, 0, 0, e, CpuOnlyPolicy, countScratch, residentScratch, keyScratch,
            plan, misses, lease);
        _lastPlan = plan;
        _lastPlanCount = distinct;

        // Expert-major gather: rows of each planned expert are contiguous, in the router's token order.
        float[] gathered = new float[pairs * h];
        ReadOnlySpan<float> xs = x.AsReadOnlySpan<float>();
        int offset = 0;
        for (int i = 0; i < distinct; i++)
        {
            List<int> tokens = expertTokens[plan[i].Key.Expert];
            for (int j = 0; j < tokens.Count; j++)
            {
                xs.Slice(tokens[j] * h, h).CopyTo(gathered.AsSpan(offset, h));
                offset += h;
            }
        }

        float[] expertOut = new float[gathered.Length];
        ExpertProgram program = _moe.Activation == ActivationKind.GeluTanh ? ExpertProgram.GeGlu : ExpertProgram.Swiglu;
        HeterogeneousExpertExecutor.Execute(program, plan.AsSpan(0, distinct), gathered, h, expertOut, ResolveHostWeights, device: null);

        // Combine exactly as the direct path: one weighted scatter-add per expert, rows in the router's token order.
        offset = 0;
        for (int i = 0; i < distinct; i++)
        {
            int ex = plan[i].Key.Expert;
            int rows = plan[i].Rows;
            Tensor expOut = new(new TensorShape(1, rows, h), DType.F32);
            expertOut.AsSpan(offset, rows * h).CopyTo(expOut.AsSpan<float>());
            backend.ScatterAddWeightedRows(output, expOut, expertTokens[ex].ToArray(), expertWeights[ex].ToArray());
            expOut.Dispose();
            offset += rows * h;
        }
    }

    /// <summary>Plan of the last routed call through the heterogeneous runtime; for tests.</summary>
    internal ExpertAssignment[] LastPlanForTest() => _lastPlan is null ? [] : _lastPlan[.._lastPlanCount];

    /// <summary>Builds the host cache once: every expert is registered and made resident. Entries reference the layer's F32 tensors,
    /// so nothing is copied.</summary>
    private HostExpertCache EnsureHostRuntime()
    {
        if (_hostCache is not null) return _hostCache;
        int e = _moe.NumExperts;
        long elements = 0;
        for (int i = 0; i < e; i++)
        {
            RequireF32(_gateW[i]);
            RequireF32(_upW[i]);
            RequireF32(_downW[i]);
            elements += _gateW[i].ElementCount + _upW[i].ElementCount + _downW[i].ElementCount;
        }
        ExpertBank bank = new(0, e, ResolveExpert, 0);
        HostExpertCache cache = new(elements * sizeof(float), [bank]);
        ExpertKey[] keys = new ExpertKey[e];
        for (int i = 0; i < e; i++) keys[i] = bank.Key(i);
        cache.Acquire(keys).Dispose();
        _hostWeights = new F32ExpertWeights?[e];
        _hostCache = cache;
        return cache;
    }

    private ExpertWeights ResolveExpert(ExpertKey key) =>
        new(key, new ExpertMatrix(_gateW[key.Expert]), new ExpertMatrix(_downW[key.Expert]), new ExpertMatrix(_upW[key.Expert]));

    private F32ExpertWeights ResolveHostWeights(ExpertKey key)
    {
        F32ExpertWeights? weights = _hostWeights![key.Expert];
        if (weights is null)
        {
            weights = new F32ExpertWeights(_hidden, _moe.MoeIntermediateSize, _gateW[key.Expert].AsReadOnlySpan<float>().ToArray(),
                _upW[key.Expert].AsReadOnlySpan<float>().ToArray(), _downW[key.Expert].AsReadOnlySpan<float>().ToArray()).Validated();
            _hostWeights[key.Expert] = weights;
        }
        return weights;
    }

    private static void RequireF32(Tensor t)
    {
        if (t.DType != DType.F32) throw new NotSupportedException("The host expert runtime needs F32 expert weights; dequantize the layer at load.");
    }

    /// <summary>DeepSeek-V3/Kimi-K2 node-limited (group-limited) routing (HF <c>noaux_tc</c>): sigmoid scores + per-expert correction bias form the SELECTION score; experts are partitioned into groups scored by the sum of their top-2 selection scores; only the top <c>ExpertGroupUsedCount</c> groups are eligible; the per-token top-k is taken over the kept groups; the routing WEIGHT is the raw sigmoid score (no bias), optionally renormalized, then scaled by <c>RoutedScalingFactor</c>.</summary>
    private void RouteGroupLimited(float[] logits, int n, int e, int topK, List<int>[] expertTokens, List<float>[] expertWeights)
    {
        int nGroup = _moe.ExpertGroupCount;
        int topkGroup = _moe.ExpertGroupUsedCount;
        int perGroup = e / nGroup;
        float scale = _moe.RoutedScalingFactor;
        float[] sig = new float[e];        // raw sigmoid → the routing weight
        float[] choice = new float[e];     // sigmoid + correction bias → the selection score
        float[] tmp = new float[e];        // choice masked to 0 outside kept groups (HF masked_fill)
        float[] groupScore = new float[nGroup];
        bool[] groupKept = new bool[nGroup];
        int[] pick = new int[topK];
        for (int t = 0; t < n; t++)
        {
            long baseOff = (long)t * e;
            for (int i = 0; i < e; i++)
            {
                float s = 1f / (1f + MathF.Exp(-logits[baseOff + i]));
                sig[i] = s;
                choice[i] = s + (_correctionBias is not null ? _correctionBias[i] : 0f);
            }
            // Group score = sum of the top-2 selection scores in the group.
            for (int g = 0; g < nGroup; g++)
            {
                int gb = g * perGroup;
                float top1 = float.NegativeInfinity, top2 = float.NegativeInfinity;
                for (int j = 0; j < perGroup; j++)
                {
                    float v = choice[gb + j];
                    if (v > top1) { top2 = top1; top1 = v; }
                    else if (v > top2) top2 = v;
                }
                groupScore[g] = top1 + top2;
                groupKept[g] = false;
            }
            // Keep the top `topkGroup` groups (ties → lower index, matching torch.topk).
            for (int kk = 0; kk < topkGroup; kk++)
            {
                int bestG = -1; float bestV = float.NegativeInfinity;
                for (int g = 0; g < nGroup; g++)
                {
                    if (groupKept[g]) continue;
                    if (groupScore[g] > bestV) { bestV = groupScore[g]; bestG = g; }
                }
                if (bestG >= 0) groupKept[bestG] = true;
            }
            for (int i = 0; i < e; i++) tmp[i] = groupKept[i / perGroup] ? choice[i] : 0f;
            // Per-token top-k over the masked selection scores.
            float wsum = 0f;
            for (int kk = 0; kk < topK; kk++)
            {
                int best = -1; float bestVal = float.NegativeInfinity;
                for (int i = 0; i < e; i++)
                {
                    bool already = false;
                    for (int j = 0; j < kk; j++) if (pick[j] == i) { already = true; break; }
                    if (already) continue;
                    if (tmp[i] > bestVal) { bestVal = tmp[i]; best = i; }
                }
                pick[kk] = best;
                wsum += sig[best];
            }
            for (int kk = 0; kk < topK; kk++)
            {
                int ex = pick[kk];
                float wt = sig[ex];
                if (_moe.NormTopKProb) wt /= wsum + 1e-20f;
                wt *= scale;
                expertTokens[ex].Add(t);
                expertWeights[ex].Add(wt);
            }
        }
    }

    private static unsafe float[] HostCopy(Tensor t, int count)
    {
        float[] r = new float[count];
        fixed (float* dst = r)
        {
            float* src = (float*)t.DataPointer;   // D2H sync on CUDA
            Buffer.MemoryCopy(src, dst, count * 4L, count * 4L);
        }
        return r;
    }
}
