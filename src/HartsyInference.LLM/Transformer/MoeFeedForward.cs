using System.Buffers;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Experts;
using HartsyInference.Core.Backends.Routing;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Mixture-of-Experts feed-forward block: a router picks the top-k experts per token, each selected expert (a SwiGLU FFN) runs only on the tokens routed to it (gather → expert GEMM → weighted scatter-add), plus an optional always-on shared expert applied to every token, replacing the dense SwiGLU on MoE layers. Covers Qwen2-MoE/Qwen3-MoE/Mixtral (softmax routing) and DeepSeek (sigmoid routing); experts run through the same quant-aware <see cref="GenericTransformer.Project"/> path as the dense FFN.</summary>
/// <remarks><b>Routing residency:</b> the router logits are read to the host to pick top-k experts and build the per-expert token groups (one D2H per MoE layer); the expert GEMMs and the gather/scatter combine stay device-resident. A fully on-device top-k + dispatch is a follow-up; correctness is unaffected.</remarks>
public sealed class MoeFeedForward(MoeConfig moe, int hiddenSize, bool lowVram)
{
    private readonly MoeConfig _moe = moe;
    private readonly int _modelHidden = hiddenSize;
    private readonly int _hidden = moe.ExpertWidth(hiddenSize);   // the width the routed experts read and write
    private readonly bool _lowVram = lowVram;

    private Tensor _routerW = null!;                 // [E, hidden]
    private Tensor? _routerW2;                       // second router [E, hidden] of split-feature routing
    private Tensor[] _gateW = null!, _upW = null!, _downW = null!;   // per expert
    private Tensor? _shGateW, _shUpW, _shDownW, _shGateScoreW;       // shared expert (optional)
    private float[]? _correctionBias;                // e_score_correction_bias [E] (selection only)
    private Tensor? _correctionBiasT;                // the same bias as a tensor for the device router
    private float[]? _routedScale;                   // learned per-expert routed scale [E]
    private Tensor? _routedScaleT;
    private Tensor? _latentDown, _latentNorm, _latentUp;            // latent entry/exit (optional)
    private Tensor? _postExpertNorm, _postExpertOnes;               // Muse Spark per-expert output norm (optional)

    /// <summary>Largest batch the device-resident routed stage serves: every (token, slot) pair reads its expert's rows once, which only
    /// beats the grouped per-expert path while few tokens share an expert.</summary>
    internal const int IndexedMaxTokens = 16;

    private bool? _deviceShapeOk;                    // dtype/shape/routing eligibility, fixed once the weights are loaded
    private bool? _indexedAlignOk;                   // block alignment the indexed GEMVs need
    private int[] _groupedOffsets = [];              // host copy of the dispatch offsets, reused across calls

    // Opt-in heterogeneous runtime state (CPU-only slice). Created on first use; the direct path never touches it.
    private static readonly ForcedPlacementPolicy CpuOnlyPolicy = new(static _ => ExpertPlacement.Cpu);
    private HostExpertCache? _hostCache;
    private F32ExpertWeights?[]? _hostWeights;
    private ExpertAssignment[]? _lastPlan;
    private int _lastPlanCount;
    private bool _useHostExpertRuntime;

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
        // Offload may plan any expert on the host, so an activation the host runtime cannot run is refused here, not mid-forward.
        _ = HostExpertProgram(_moe.Activation);
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

    /// <summary>Routed experts in this block.</summary>
    public int NumExperts => _moe.NumExperts;

    /// <summary>Inner width of one routed expert.</summary>
    public int IntermediateSize => _moe.MoeIntermediateSize;

    /// <summary>
    /// Opt-in (default off): routes the routed experts through the heterogeneous runtime. The scheduler plans every expert on
    /// the CPU, <see cref="HostExpertCache"/> holds the layer's F32 weights without copying them, and
    /// <see cref="HeterogeneousExpertExecutor"/> runs the F32 reference. The combine is the same weighted scatter-add as the
    /// direct path. Requires F32 expert weights and an activation the host runtime runs; enabling it otherwise throws.
    /// </summary>
    public bool UseHostExpertRuntime
    {
        get => _useHostExpertRuntime;
        set
        {
            if (value) _ = HostExpertProgram(_moe.Activation);
            _useHostExpertRuntime = value;
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // The opt-in runtime caches the layer's weights on first use; a reload must not keep serving the old copies.
        _hostCache?.Dispose();
        _hostCache = null;
        _hostWeights = null;
        _offload = null;
        _offloadLayer = -1;
        _deviceShapeOk = null;
        _indexedAlignOk = null;
        ValidateRoutingOptions();
        string block = $"{prefix}.mlp";
        _routerW = w[$"{block}.gate.weight"];
        _routerW2 = _moe.SplitFeatureRouting ? w[$"{block}.gate.weight_1"] : null;
        // DeepSeek-V3 / Kimi-K2 router correction bias (added to the selection scores only). Optional: V2-Lite and
        // every softmax-routed MoE lack it. Read once to a host array (it is a tiny [E] vector used per token).
        // llama.cpp keeps router biases F32, so a direct host copy is exact.
        _correctionBias = null;
        _correctionBiasT = null;
        if (w.TryGetValue($"{block}.gate.e_score_correction_bias", out Tensor? cb))
        {
            _correctionBias = HostVector(cb, _moe.NumExperts, "correction bias");
            _correctionBiasT = cb;
        }
        else if (_moe.Scoring == MoeScoring.SigmoidLogitAdd)
        {
            throw new InvalidDataException("SigmoidLogitAdd routing requires an e_score_correction_bias tensor.");
        }
        _routedScaleT = _moe.LearnedRoutedScale ? w[$"{block}.gate.routed_scaling_factor_param"] : null;
        _routedScale = _routedScaleT is null ? null : HostVector(_routedScaleT, _moe.NumExperts, "routed scale");
        LoadLatentAndExpertNorm(w, block);
        int e = _moe.NumExperts;
        _gateW = new Tensor[e];
        _upW = new Tensor[e];
        _downW = new Tensor[e];
        for (int i = 0; i < e; i++)
        {
            string ep = $"{block}.experts.{i}";
            _gateW[i] = w[$"{ep}.gate_proj.weight"];
            _upW[i] = w[$"{ep}.up_proj.weight"];
            _downW[i] = w[$"{ep}.down_proj.weight"];
        }
        _shGateW = _shUpW = _shDownW = _shGateScoreW = null;
        if (_moe.SharedExpertIntermediateSize > 0)
        {
            string sp = $"{block}.shared_expert";
            _shGateW = w[$"{sp}.gate_proj.weight"];
            _shUpW = w[$"{sp}.up_proj.weight"];
            _shDownW = w[$"{sp}.down_proj.weight"];
            // The shared-expert sigmoid gate ([1, hidden]) is optional (Qwen2-MoE has it; some variants don't).
            w.TryGetValue($"{block}.shared_expert_gate.weight", out Tensor? sg);
            _shGateScoreW = sg;
            // A latent block's shared expert reads and writes the model width, not the expert width.
            if (_latentDown is not null
                && (_shGateW.Shape[1] != _modelHidden || _shUpW.Shape[1] != _modelHidden || _shDownW.Shape[0] != _modelHidden))
                throw new InvalidDataException($"{block}: the shared expert of a latent block must read and write the model width {_modelHidden}.");
        }
    }

    /// <summary>Refuses option combinations the host router cannot reproduce.</summary>
    private void ValidateRoutingOptions()
    {
        bool ernieRouting = _moe.SplitFeatureRouting || _moe.LearnedRoutedScale;
        if (ernieRouting && _moe.ExpertGroupCount > 0)
            throw new NotSupportedException("Split-feature routing and a learned routed scale are not supported with group-limited routing.");
        if (_moe.SplitFeatureRouting && _moe.Scoring != MoeScoring.Sigmoid)
            throw new NotSupportedException($"Split-feature routing sums sigmoid scores; {_moe.Scoring} scoring is not supported with it.");
    }

    /// <summary>Loads the optional latent entry/exit projections and the per-expert output norm the config asks for.</summary>
    private void LoadLatentAndExpertNorm(IReadOnlyDictionary<string, Tensor> w, string block)
    {
        _latentDown = _latentNorm = _latentUp = _postExpertNorm = _postExpertOnes = null;
        if (_moe.LatentSize > 0)
        {
            // The latent norm weight keeps its Muse Spark name wherever MoeConfig.LatentNorm places it.
            _latentDown = w[$"{block}.pre_expert_proj.weight"];
            _latentNorm = _moe.LatentNorm == MoeLatentNorm.None ? null : GenericTransformer.EnsureF32(w[$"{block}.pre_expert_norm.weight"]);
            _latentUp = w[$"{block}.post_expert_proj.weight"];
            if (_latentDown.Shape[0] != _hidden || _latentUp.Shape[1] != _hidden)
                throw new InvalidDataException($"{block}: latent projections do not match the configured latent size {_hidden}.");
        }
        if (_moe.PostExpertNorm)
        {
            _postExpertNorm = GenericTransformer.EnsureF32(w[$"{block}.post_expert_norm.weight"]);
            _postExpertOnes = GenericTransformer.Ones(_hidden);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _routerW;
        if (_routerW2 is not null) yield return _routerW2;
        for (int i = 0; i < _gateW.Length; i++) { yield return _gateW[i]; yield return _upW[i]; yield return _downW[i]; }
        if (_shGateW is not null) { yield return _shGateW; yield return _shUpW!; yield return _shDownW!; }
        if (_shGateScoreW is not null) yield return _shGateScoreW;
        if (_correctionBiasT is not null) yield return _correctionBiasT;
        if (_routedScaleT is not null) yield return _routedScaleT;
        if (_latentDown is not null) yield return _latentDown;
        if (_latentNorm is not null) yield return _latentNorm;
        if (_latentUp is not null) yield return _latentUp;
        if (_postExpertNorm is not null) { yield return _postExpertNorm; yield return _postExpertOnes!; }
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
    /// <remarks>Under <see cref="MoeConfig.SplitFeatureRouting"/> the tensor already holds the routing scores
    /// <c>sigmoid(x @ gate) + sigmoid(x @ gate_1)</c>, which is what <see cref="Forward"/> then expects.</remarks>
    public Tensor ComputeRouterLogits(IBackend backend, Tensor x, int n)
    {
        TensorShape shape = new(1, n, _moe.NumExperts);
        Tensor logits = new(shape, DType.F32);
        GenericTransformer.Project(backend, logits, x, _routerW, null, lowVram: false);
        if (_routerW2 is null) return logits;
        Tensor scores = new(shape, DType.F32);
        try
        {
            GenericTransformer.Project(backend, scores, x, _routerW2, null, lowVram: false);
            using Tensor s1 = new(shape, DType.F32);
            backend.Sigmoid(s1, logits);
            backend.Sigmoid(logits, scores);
            backend.Add(scores, s1, logits);
        }
        catch
        {
            scores.Dispose();
            throw;
        }
        finally
        {
            logits.Dispose();
        }
        return scores;
    }

    /// <summary>MoE FFN: <paramref name="x"/> <c>[1, N, hidden]</c> → <c>[1, N, hidden]</c>; <paramref name="routerLogits"/> overrides the internal <c>x @ router_weight</c> projection when provided (Gemma-4: the router reads a differently-normalized view of the attention output, computed by the caller) — same <c>[1, N, E]</c> shape either way, not disposed by this call. Under
    /// <see cref="MoeConfig.SplitFeatureRouting"/> it must hold the summed sigmoid scores <see cref="ComputeRouterLogits"/> returns, not raw
    /// logits.</summary>
    /// <remarks>With <see cref="MoeConfig.LatentSize"/> set the router and the shared expert read the full-width <paramref name="x"/>. The
    /// experts read <c>fc1(x)</c>, normalized first under <see cref="MoeLatentNorm.BeforeExperts"/> (Muse Spark); their weighted sum is
    /// normalized under <see cref="MoeLatentNorm.AfterCombine"/> (ERNIE-5.5) and leaves through <c>fc2</c>; the shared expert's output is
    /// added after <c>fc2</c>.</remarks>
    public Tensor Forward(IBackend backend, Tensor x, int n, Tensor? routerLogits = null)
    {
        if (_latentDown is null) return ForwardRouted(backend, x, n, routerLogits);
        Tensor? ownLogits = routerLogits is null ? ComputeRouterLogits(backend, x, n) : null;
        Tensor? expertSum = null;
        try
        {
            using (Tensor latent = new(new TensorShape(1, n, _hidden), DType.F32))
            {
                GenericTransformer.Project(backend, latent, x, _latentDown, null, _lowVram);
                if (_moe.LatentNorm == MoeLatentNorm.BeforeExperts)
                {
                    using Tensor latentNormed = LatentRmsNorm(backend, latent);
                    expertSum = ForwardRouted(backend, latentNormed, n, routerLogits ?? ownLogits);
                }
                else
                {
                    expertSum = ForwardRouted(backend, latent, n, routerLogits ?? ownLogits);
                }
            }
            if (_moe.LatentNorm == MoeLatentNorm.AfterCombine)
            {
                Tensor normed = LatentRmsNorm(backend, expertSum);
                expertSum.Dispose();
                expertSum = normed;
            }
            Tensor output = new(new TensorShape(1, n, _modelHidden), DType.F32);
            GenericTransformer.Project(backend, output, expertSum, _latentUp!, null, _lowVram);
            if (_shGateW is not null) AddSharedExpert(backend, output, x, n);
            return output;
        }
        finally
        {
            expertSum?.Dispose();
            ownLogits?.Dispose();
        }
    }

    private Tensor LatentRmsNorm(IBackend backend, Tensor input)
    {
        Tensor normed = new(input.Shape, DType.F32);
        backend.RmsNorm(normed, input, _latentNorm!, _moe.LatentNormEps);
        return normed;
    }

    /// <summary>True when the routed stage folds the shared expert into its own combine. A latent block's routed stage runs at the expert
    /// width, so its shared expert joins in <see cref="Forward"/> instead, on the full-width input.</summary>
    private bool FoldsSharedExpert => _shGateW is not null && _latentDown is null;

    /// <summary>Adds the shared expert of <paramref name="x"/>, scaled by its optional per-token sigmoid gate, into
    /// <paramref name="output"/>.</summary>
    private void AddSharedExpert(IBackend backend, Tensor output, Tensor x, int n)
    {
        Tensor shared = SwiGlu(backend, x, n, _shGateW!, _shUpW!, _shDownW!, _moe.SharedExpertIntermediateSize, _modelHidden);
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
        // Identity scatter folds the per-token gate in and accumulates in one device op.
        backend.ScatterAddRows(output, shared, identity, gateVals);
        shared.Dispose();
    }

    /// <summary>The routed stage at the expert width: <paramref name="x"/> <c>[1, N, expert width]</c> in, the weighted expert sum (plus any shared expert) out.</summary>
    private unsafe Tensor ForwardRouted(IBackend backend, Tensor x, int n, Tensor? routerLogits)
    {
        int e = _moe.NumExperts;
        int topK = _moe.NumExpertsPerTok;

        if (n <= IndexedMaxTokens && CanRunIndexed(backend)) return ForwardIndexed(backend, x, n, routerLogits);
        if (n > IndexedMaxTokens && CanRunGrouped(backend)) return ForwardGrouped(backend, x, n, routerLogits);

        // 1. Router logits, read to host for top-k selection.
        Tensor? ownRouterLogits = routerLogits is null ? ComputeRouterLogits(backend, x, n) : null;
        Tensor routerLogitsT = routerLogits ?? ownRouterLogits!;
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
        if (FoldsSharedExpert) AddSharedExpert(backend, output, x, n);

        // 4. Routed experts: gather → expert SwiGLU → weighted scatter-add.
        if (_offload is not null || UseHostExpertRuntime)
        {
            bool routed = false;
            try
            {
                if (_offload is not null) RunRoutedWithOffload(backend, x, n, output, expertTokens, expertWeights);
                else RunRoutedThroughRuntime(backend, x, n, output, expertTokens, expertWeights);
                routed = true;
                return output;
            }
            finally
            {
                if (!routed) output.Dispose();
            }
        }
        for (int ex = 0; ex < e; ex++)
        {
            int ne = expertTokens[ex].Count;
            if (ne == 0) continue;
            int[] idx = [.. expertTokens[ex]];
            float[] wts = [.. expertWeights[ex]];

            Tensor gathered = new(new TensorShape(1, ne, _hidden), DType.F32);
            backend.GatherRows(gathered, x, idx);
            Tensor expOut = RoutedExpert(backend, gathered, ne, ex);
            gathered.Dispose();
            backend.ScatterAddRows(output, expOut, idx, wts);
            expOut.Dispose();
        }
        return output;
    }

    /// <summary>One routed expert over <paramref name="rows"/> gathered tokens, including the per-expert output norm when configured.</summary>
    private Tensor RoutedExpert(IBackend backend, Tensor x, int rows, int expert) =>
        NormExpertRows(backend, SwiGlu(backend, x, rows, _gateW[expert], _upW[expert], _downW[expert], _moe.MoeIntermediateSize, _hidden));

    /// <summary>Muse Spark's per-expert output norm: scales each routed row by <c>post_expert_norm</c>, then RMS-normalizes it. Consumes
    /// <paramref name="rows"/> and returns the normalized rows; returns <paramref name="rows"/> itself when the block has no such norm.</summary>
    private Tensor NormExpertRows(IBackend backend, Tensor rows)
    {
        if (_postExpertNorm is null) return rows;
        Tensor scaled = new(rows.Shape, DType.F32);
        backend.AffineBroadcastLastDim(scaled, rows, _postExpertNorm, null);
        rows.Dispose();
        Tensor normed = new(scaled.Shape, DType.F32);
        backend.RmsNorm(normed, scaled, _postExpertOnes!, _moe.PostExpertNormEps);
        scaled.Dispose();
        return normed;
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
            for (int i = 0; i < distinct; i++)
            {
                ExpertAssignment assignment = _offPlan[i];
                bool streamed = assignment.Placement == ExpertPlacement.Cpu && offload.Streams(assignment.Rows);
                if (assignment.Placement != ExpertPlacement.Gpu && !streamed) continue;
                int ex = assignment.Key.Expert;
                ReadOnlySpan<int> idx = CollectionsMarshal.AsSpan(expertTokens[ex]);
                Tensor gatheredRows = new(new TensorShape(1, idx.Length, h), DType.F32);
                backend.GatherRows(gatheredRows, x, idx);
                Tensor expOut = RoutedExpert(backend, gatheredRows, idx.Length, ex);
                gatheredRows.Dispose();
                backend.ScatterAddRows(output, expOut, idx, CollectionsMarshal.AsSpan(expertWeights[ex]));
                expOut.Dispose();
                offload.Record(assignment.Placement, streamed, assignment.Rows);
            }

            // 3. Host experts, computed while the device runs, then combined one expert at a time.
            if (hostExperts > 0)
            {
                ExpertProgram program = HostExpertProgram(_moe.Activation);
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
                Tensor rawOut = new(new TensorShape(hostRows, h), DType.F32);
                outBuf.AsSpan(0, hostRows * h).CopyTo(rawOut.AsSpan<float>());
                using Tensor allOut = NormExpertRows(backend, rawOut);
                for (int j = 0; j < hostExperts; j++)
                {
                    ExpertAssignment a = plan[hostPlan[j]];
                    int ex = a.Key.Expert;
                    using Tensor expOut = allOut.SliceRows(hostOffset[j], a.Rows);
                    backend.ScatterAddRows(output, expOut, CollectionsMarshal.AsSpan(expertTokens[ex]), CollectionsMarshal.AsSpan(expertWeights[ex]));
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
        if (!DeviceRoutingEligible()) return false;
        _indexedAlignOk ??= IndexedAlignmentOk();
        return _indexedAlignOk.Value && backend.SupportsMoeExpertIndexed(_gateW[0].DType) && backend.SupportsMoeExpertIndexed(_downW[0].DType)
            && backend.MoeExpertsResident(_gateW, _upW, _downW);
    }

    /// <summary>True when a large batch can take the device-routed grouped path: routing the device router reproduces, one quantized type per
    /// projection, and a backend that runs the grouped expert GEMMs for those types.</summary>
    internal bool CanRunGrouped(IBackend backend) =>
        DeviceRoutingEligible() && backend.SupportsMoeExpertsGrouped(_gateW[0].DType) && backend.SupportsMoeExpertsGrouped(_downW[0].DType)
        && backend.MoeExpertsResident(_gateW, _upW, _downW);

    /// <summary>What both device-routed paths need whatever the backend: the knob, experts owned by this layer (no host runtime or offload), and a
    /// routing/dtype/shape combination the device router and expert kernels reproduce.</summary>
    private bool DeviceRoutingEligible()
    {
        if (!HartsyInference.Core.Configuration.EngineKnobs.MoeIndexed.Value) return false;
        if (UseHostExpertRuntime || _offload is not null) return false;
        _deviceShapeOk ??= DeviceRoutedShapeOk();
        return _deviceShapeOk.Value;
    }

    private bool DeviceRoutedShapeOk()
    {
        // Group-limited, logit-biased, split-feature and learned-scale routing keep the host router: MoeRouteArgs carries none of them.
        // The device expert kernels take only the SiLU or tanh-GELU gate, so every other activation runs the generic path.
        if (_moe.ExpertGroupCount > 0 || _moe.Scoring == MoeScoring.SigmoidLogitAdd) return false;
        if (_moe.SplitFeatureRouting || _moe.LearnedRoutedScale) return false;
        if (_moe.Activation is not (ActivationKind.Silu or ActivationKind.GeluTanh)) return false;
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

    /// <summary>The indexed GEMVs read whole quant blocks along the input length, so the hidden and intermediate sizes must be block multiples.</summary>
    private bool IndexedAlignmentOk() =>
        _hidden % _gateW[0].DType.BlockElementCount == 0 && _moe.MoeIntermediateSize % _downW[0].DType.BlockElementCount == 0;

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
        backend.MoeRoute(topkIdx, topkWeight, logits, RouteArgs, _correctionBiasT);
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
        backend.MoeExpertsGrouped(expertOut, x, perm, _groupedOffsets, _gateW, _upW, _downW, gelu: DeviceGateIsGelu);
        expertOut = NormExpertRows(backend, expertOut);
        perm.Dispose();

        Tensor? shared = null, sharedGate = null;
        if (FoldsSharedExpert)
        {
            shared = SwiGlu(backend, x, n, _shGateW!, _shUpW!, _shDownW!, _moe.SharedExpertIntermediateSize, _modelHidden);
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

    private MoeRouteArgs RouteArgs => _moe.SplitFeatureRouting || _moe.LearnedRoutedScale
        ? throw new NotSupportedException("The device MoE router takes neither split-feature routing nor a learned per-expert routed scale.")
        : new(_moe.NumExperts, _moe.NumExpertsPerTok, _moe.Scoring == MoeScoring.Softmax ? MoeRouteScoring.Softmax : MoeRouteScoring.Sigmoid,
            Renormalize: _moe.NormTopKProb, Scale: _moe.RoutedScalingFactor);

    /// <summary>The device expert kernels' gate flag: tanh-GELU or SiLU, and nothing else.</summary>
    private bool DeviceGateIsGelu => _moe.Activation switch
    {
        ActivationKind.Silu => false,
        ActivationKind.GeluTanh => true,
        _ => throw new NotSupportedException($"The device MoE expert kernels do not run the {_moe.Activation} expert activation."),
    };

    /// <summary>The routed stage with nothing leaving the device: router logits, <see cref="IBackend.MoeRoute"/>, the expert-indexed gate/up and down
    /// GEMVs, then one combine that also folds in the shared expert. Capturable in a CUDA graph.</summary>
    private Tensor ForwardIndexed(IBackend backend, Tensor x, int n, Tensor? routerLogits)
    {
        using MoeCombineParts parts = ForwardIndexedParts(backend, x, n, routerLogits);
        Tensor output = new(new TensorShape(1, n, _hidden), DType.F32);
        backend.MoeCombineSlots(output, parts.SlotOut, parts.TopkWeight, parts.Shared, parts.SharedGate, parts.TopK);
        return output;
    }

    /// <summary>Single-token device-routed MoE up to (not including) the combine, for a caller that fuses the combine into the next op
    /// (<see cref="IBackend.MoeCombineAddRmsNormEmitQ8"/>). Null when this layer does not take the device-routed path; the caller then
    /// runs <see cref="Forward"/>.</summary>
    public MoeCombineParts? TryForwardDeferred(IBackend backend, Tensor x)
        => _latentDown is null && CanRunIndexed(backend) ? ForwardIndexedParts(backend, x, 1, routerLogits: null) : null;

    private MoeCombineParts ForwardIndexedParts(IBackend backend, Tensor x, int n, Tensor? routerLogits)
    {
        int e = _moe.NumExperts;
        int k = _moe.NumExpertsPerTok;
        Tensor? ownLogits = null;
        Tensor logits = routerLogits ?? (ownLogits = new(new TensorShape(1, n, e), DType.F32));
        if (routerLogits is null) GenericTransformer.Project(backend, logits, x, _routerW, null, lowVram: false);

        Tensor topkIdx = new(new TensorShape(n, k), DType.I32);
        Tensor topkWeight = new(new TensorShape(n, k), DType.F32);
        backend.MoeRoute(topkIdx, topkWeight, logits, RouteArgs, _correctionBiasT);
        ownLogits?.Dispose();

        int rows = n * k;
        Tensor act = new(new TensorShape(1, rows, _moe.MoeIntermediateSize), DType.F32);
        backend.MoeExpertGateUp(act, x, _gateW, _upW, topkIdx, k, gelu: DeviceGateIsGelu);
        Tensor slotOut = new(new TensorShape(1, rows, _hidden), DType.F32);
        backend.MoeExpertDown(slotOut, act, _downW, topkIdx, k);
        slotOut = NormExpertRows(backend, slotOut);
        act.Dispose();
        topkIdx.Dispose();

        Tensor? shared = null, sharedGate = null;
        if (FoldsSharedExpert)
        {
            shared = SwiGlu(backend, x, n, _shGateW!, _shUpW!, _shDownW!, _moe.SharedExpertIntermediateSize, _modelHidden);
            if (_shGateScoreW is not null)
            {
                sharedGate = new(new TensorShape(1, n, 1), DType.F32);
                GenericTransformer.Project(backend, sharedGate, x, _shGateScoreW, null, lowVram: false);
            }
        }
        return new MoeCombineParts(slotOut, topkWeight, shared, sharedGate, k);
    }

    /// <summary>Gated FFN over <paramref name="rows"/> tokens to <paramref name="width"/> outputs: down(act(gate(x)) * up(x)) under
    /// <see cref="MoeConfig.Activation"/>: SiLU, tanh-GELU, ReLU or ReLU², the clamped alpha-SwiGLU, or SiTU-GLU. Any other activation throws.</summary>
    private Tensor SwiGlu(IBackend backend, Tensor x, int rows, Tensor gateW, Tensor upW, Tensor downW, int inter, int width)
    {
        TensorShape ff = new(1, rows, inter);
        Tensor gate = new(ff, DType.F32);
        GenericTransformer.Project(backend, gate, x, gateW, null, _lowVram);
        Tensor comb;
        if (_moe.Activation is ActivationKind.SwigluOai or ActivationKind.Situ)
        {
            Tensor gatedUp = new(ff, DType.F32);
            GenericTransformer.Project(backend, gatedUp, x, upW, null, _lowVram);
            comb = GatedFfn.SwiGlu(backend, _moe.Activation, gate, gatedUp, _moe.SwigluLimit, _moe.SwigluAlpha,
                _moe.ActivationSituBeta, _moe.ActivationSituLinearBeta);
        }
        else
        {
            Tensor act = new(ff, DType.F32);
            GatedFfn.Activate(backend, _moe.Activation, act, gate);
            gate.Dispose();
            Tensor up = new(ff, DType.F32);
            GenericTransformer.Project(backend, up, x, upW, null, _lowVram);
            comb = new(ff, DType.F32);
            backend.Mul(comb, act, up);
            act.Dispose(); up.Dispose();
        }
        Tensor outp = new(new TensorShape(1, rows, width), DType.F32);
        GenericTransformer.Project(backend, outp, comb, downW, null, _lowVram);
        comb.Dispose();
        return outp;
    }

    /// <summary>Whether the host expert runtime, and so expert offload, runs routed experts with <paramref name="activation"/>.</summary>
    public static bool HostRunsExpertActivation(ActivationKind activation) => activation is ActivationKind.Silu or ActivationKind.GeluTanh;

    /// <summary>The host expert runtime's program for an expert activation; refuses the ones it does not run.</summary>
    private static ExpertProgram HostExpertProgram(ActivationKind activation)
    {
        if (!HostRunsExpertActivation(activation))
            throw new NotSupportedException($"The host expert runtime does not run the {activation} expert activation.");
        return activation == ActivationKind.GeluTanh ? ExpertProgram.GeGlu : ExpertProgram.Swiglu;
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
                for (int i = 0; i < e; i++) selection[i] = (weight[i] /= sum) + (_correctionBias?[i] ?? 0f);
            }
            else if (_moe.SplitFeatureRouting)
            {
                // The router already summed the two sigmoids, so the input holds the scores themselves.
                for (int i = 0; i < e; i++)
                {
                    weight[i] = logits[baseOff + i];
                    selection[i] = weight[i] + (_correctionBias?[i] ?? 0f);
                }
            }
            else
            {
                for (int i = 0; i < e; i++)
                {
                    float logit = logits[baseOff + i];
                    weight[i] = 1f / (1f + MathF.Exp(-logit));
                    // A selection-only bias moves which experts win, never their weights (HF noaux_tc, llama.cpp exp_probs_b).
                    selection[i] = _moe.Scoring == MoeScoring.SigmoidLogitAdd
                        ? logit + _correctionBias![i]
                        : weight[i] + (_correctionBias?[i] ?? 0f);
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
            // The clamp only matters when every picked score underflows; above it the divisor is wsum itself.
            float denom = MathF.Max(wsum, 1e-20f);
            for (int kk = 0; kk < topK; kk++)
            {
                int ex = pick[kk];
                float wt = weight[ex];
                if (_moe.NormTopKProb) wt /= denom;
                // llama.cpp scales the routed weights by w_scale unconditionally (1.0 is a no-op for Qwen/Mixtral);
                // a learned per-expert scale replaces it.
                wt *= _routedScale is null ? _moe.RoutedScalingFactor : _routedScale[ex];
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
        ExpertProgram program = HostExpertProgram(_moe.Activation);
        HeterogeneousExpertExecutor.Execute(program, plan.AsSpan(0, distinct), gathered, h, expertOut, ResolveHostWeights, device: null);

        // Combine exactly as the direct path: one weighted scatter-add per expert, rows in the router's token order.
        offset = 0;
        for (int i = 0; i < distinct; i++)
        {
            int ex = plan[i].Key.Expert;
            int rows = plan[i].Rows;
            Tensor rawOut = new(new TensorShape(1, rows, h), DType.F32);
            expertOut.AsSpan(offset, rows * h).CopyTo(rawOut.AsSpan<float>());
            Tensor expOut = NormExpertRows(backend, rawOut);
            backend.ScatterAddRows(output, expOut, expertTokens[ex].ToArray(), expertWeights[ex].ToArray());
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

    /// <summary>A per-expert F32 router vector read once to the host; refuses any other dtype or length.</summary>
    private static float[] HostVector(Tensor t, int experts, string what)
    {
        if (t.DType != DType.F32) throw new NotSupportedException($"MoE {what} must be F32; got {t.DType}.");
        if (t.ElementCount != experts) throw new InvalidDataException($"MoE {what} has {t.ElementCount} entries; expected {experts}.");
        return HostCopy(t, experts);
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

/// <summary>A device-routed MoE output before its combine: the per-slot expert rows, their routing weights and the optional shared-expert
/// row and gate logit. Disposing it frees all four.</summary>
public sealed record MoeCombineParts(Tensor SlotOut, Tensor TopkWeight, Tensor? Shared, Tensor? SharedGate, int TopK) : IDisposable
{
    public void Dispose()
    {
        SlotOut.Dispose();
        TopkWeight.Dispose();
        Shared?.Dispose();
        SharedGate?.Dispose();
    }
}
