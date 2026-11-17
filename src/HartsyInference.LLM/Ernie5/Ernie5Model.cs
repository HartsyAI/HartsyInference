using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Heads;
using HartsyInference.LLM.Transformer.Mhc;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Host reference model of ERNIE-5.5 (vLLM-ERNIE <c>Ernie5_5ForCausalLM</c>): the token embedding broadcast to <c>hc_mult</c> residual
/// streams, the decoder layers (HCA or DSA attention, dense or latent-MoE SiTU FFN, each under mHC), the mHC head read-out with the final
/// RMS norm, the LM head, and multimax. With <see cref="Ernie5LoadOptions.Mtp"/> it also runs the multi-token-prediction layer
/// (vLLM-ERNIE <c>Ernie5_5MTP</c>) that drafts the token after next.</summary>
/// <remarks>
/// <para>Every op runs through <see cref="IBackend"/>, so the model runs on the CPU, CUDA or Vulkan backend. The residual stream, the
/// sublayer glue and the attention caches stay host arrays, and each backend op takes them as host tensors. On a GPU backend,
/// <c>Ernie5ModelGpuParityTests</c> covers one prefill chunk and one decode step with weights rebuilt per forward, and with resident weights on CUDA and Vulkan; MTP is not covered on a device.</para>
/// <para>FP8 linears stay E4M3 with their block-scale grid on the weight, so each runs <c>IBackend.Fp8BlockLinear</c>; MoE experts run per
/// expert through the same linear. <see cref="Ernie5LoadOptions.ResidentWeights"/> keeps every built layer; otherwise each layer is built
/// and dropped per forward. Not thread-safe: run one sequence at a time.</para>
/// <para>A GPU linear reads its weight from the backend's weight cache, or uploads it for the one call. With
/// <see cref="Ernie5LoadOptions.ResidentWeights"/> the load preloads every layer's per-token weights (<see cref="Ernie5Layer.EnumerateDenseWeights"/>)
/// and the LM head, in layer order, up to the device's free memory less a headroom kept for activations and routed-expert uploads; routed
/// experts stay mapped on the host and the backend may keep the ones it sees reused. The MTP layer is not preloaded, so with
/// <see cref="Ernie5LoadOptions.Mtp"/> its weights upload per call. Streamed layers are new tensors on every forward, so nothing of theirs
/// stays on the device.</para>
/// <para>With <see cref="Ernie5LoadOptions.Placement"/> the layers are split into ranges, one backend per range (a pipeline over several
/// devices), and each range preloads onto its own device; the head read-out, LM head and MTP layer run on the last stage. With
/// <see cref="Ernie5LoadOptions.PreloadExperts"/> the routed experts and every block-scale grid are preloaded too, and a stage whose weights do
/// not fit its device refuses the load.</para>
/// </remarks>
public sealed class Ernie5Model : IDisposable, IExpertOffloadTarget
{
    // The last stage's backend: the head read-out, the LM head and the MTP layer run on it.
    private readonly IBackend _backend;
    private readonly LlmPlacement _placement;
    private readonly IBackend[] _layerBackends;
    // Every weight the load preloaded, with the backend that holds it, so disposal frees each on its own device.
    private readonly List<(IBackend Backend, List<Tensor> Weights)> _preloaded = [];
    private readonly IErnie5WeightSource _source;
    private readonly IDisposable? _ownedSource;
    private readonly Ernie5LoadOptions _options;
    private readonly DeepSeekV41RopeTable _hcaRope;
    private readonly DeepSeekV41RopeTable _dsaRope;
    private readonly Ernie5Layer?[] _layers;
    private readonly HyperConnection _head;
    private readonly float[] _headFn;
    private readonly float[] _headScale;
    private readonly float[] _headBase;
    private readonly float[] _finalNorm;
    private readonly float[]? _multimaxRanges;
    private readonly float[]? _multimaxGains;
    private readonly float[]? _mtpHeadFn;
    private readonly float[]? _mtpHeadScale;
    private readonly float[]? _mtpHeadBase;
    private readonly float[]? _mtpNorm;
    private Ernie5MtpLayer? _mtp;
    private Tensor? _embedding;
    private Tensor? _lmHead;
    private int _disposed;

    /// <param name="backend">The CPU, CUDA or Vulkan backend; with <see cref="Ernie5LoadOptions.Placement"/>, its last stage's backend.</param>
    /// <param name="config">The checkpoint's config.</param>
    /// <param name="source">The weights; every key must map (<see cref="Ernie5WeightMap"/>). Not owned unless loaded through
    /// <see cref="Load"/>.</param>
    /// <param name="options">Sequence length and weight residency.</param>
    /// <exception cref="NotSupportedException">A backend is not CPU, CUDA or Vulkan, the config names a feature the model does not implement,
    /// <see cref="Ernie5LoadOptions.Mtp"/> asks for MTP on a checkpoint with other than one MTP layer, a multi-stage placement comes without
    /// <see cref="Ernie5LoadOptions.ResidentWeights"/>, or with <see cref="Ernie5LoadOptions.PreloadExperts"/> a stage's weights do not fit
    /// its device.</exception>
    /// <exception cref="ArgumentException">The placement does not cover exactly the decoder layers, or <paramref name="backend"/> is not its last
    /// stage's backend.</exception>
    /// <exception cref="InvalidDataException">A checkpoint key maps to nothing, or a parameter has no key.</exception>
    public Ernie5Model(IBackend backend, Ernie5Config config, IErnie5WeightSource source, Ernie5LoadOptions options)
        : this(backend, config, source, options, ownedSource: null)
    {
    }

    private Ernie5Model(IBackend backend, Ernie5Config config, IErnie5WeightSource source, Ernie5LoadOptions options, IDisposable? ownedSource)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxSequenceTokens, 1);
        LlmPlacement placement = options.Placement ?? LlmPlacement.Single(backend, config.NumHiddenLayers);
        if (placement.LayerCount != config.NumHiddenLayers)
            throw new ArgumentException($"The placement covers {placement.LayerCount} layers; the checkpoint has {config.NumHiddenLayers}.", nameof(options));
        if (!ReferenceEquals(placement.LastBackend, backend))
            throw new ArgumentException("The model's backend must be the placement's last stage backend: the head read-out, the LM head and the MTP layer run on it.",
                nameof(backend));
        if (!placement.IsSingle && !options.ResidentWeights)
            throw new NotSupportedException("A multi-stage ERNIE-5.5 placement needs ResidentWeights: a streamed layer is rebuilt on every forward, so splitting it "
                + "over devices keeps nothing on them.");
        foreach (LlmStage stage in placement.Stages)
            if (stage.Backend.Device.IsGpu && !stage.Backend.Device.IsCuda && !stage.Backend.Device.IsVulkan)
                throw new NotSupportedException($"ERNIE-5.5 has no path on the {stage.Backend.Device} backend; it runs on the CPU, CUDA and Vulkan backends.");
        IReadOnlyList<string> missing = config.UnimplementedFeatures(options.Mtp);
        if (missing.Count > 0)
            throw new NotSupportedException($"This ERNIE-5.5 config names features the model does not implement: {string.Join("; ", missing)}.");
        if (options.Mtp && config.NumNextnPredictLayers != 1)
            throw new NotSupportedException($"This ERNIE-5.5 config has {config.NumNextnPredictLayers} multi-token-prediction layers; "
                + "MTP is wired for exactly one.");
        Coverage = Ernie5WeightMap.Check(config, source.Names, options.Mtp);
        Coverage.RequireComplete();

        _backend = backend;
        _placement = placement;
        _layerBackends = new IBackend[config.NumHiddenLayers];
        foreach (LlmStage stage in placement.Stages)
            for (int i = stage.StartLayer; i < stage.EndLayer; i++) _layerBackends[i] = stage.Backend;
        Config = config;
        _source = source;
        _ownedSource = ownedSource;
        _options = options;
        int max = options.MaxSequenceTokens;
        _hcaRope = DeepSeekV41RopeTable.Build(config.QkRopeHeadDim, max, config.CompressRopeTheta, config.HcaRopeScaling);
        _dsaRope = DeepSeekV41RopeTable.Build(config.MlaQkRopeHeadDim, max, config.RopeTheta, null);
        _layers = new Ernie5Layer?[config.NumHiddenLayers];
        int hc = config.HcMult, flat = hc * config.HiddenSize, mixes = (2 + hc) * hc;
        // the head read-out uses only the formula config; its own projection and gates are passed to Head
        _head = new HyperConnection(backend, Ernie5Layer.HcConfig(config), new float[mixes * flat], new float[3], new float[mixes]);
        _headFn = Ernie5Layer.Transpose(Ernie5Layer.HostF32(source, Ernie5WeightNames.HeadFn), flat, hc);
        _headScale = Ernie5Layer.HostF32(source, Ernie5WeightNames.HeadScale);
        _headBase = Ernie5Layer.HostF32(source, Ernie5WeightNames.HeadBase);
        _finalNorm = Ernie5Layer.HostF32(source, Ernie5WeightNames.FinalNorm);
        if (config.Multimax is not null)
        {
            _multimaxRanges = Ernie5Layer.HostF32(source, Ernie5WeightNames.MultimaxRanges);
            _multimaxGains = Ernie5Layer.HostF32(source, Ernie5WeightNames.MultimaxGains);
        }
        if (options.Mtp)
        {
            int mtp = config.NumHiddenLayers;
            float[] stored = Ernie5Layer.HostF32(source, Ernie5WeightNames.Mtp(mtp, Ernie5WeightNames.MtpHeadFn));
            if (stored.Length != (long)flat * hc)
                throw new InvalidDataException($"MTP layer {mtp} {Ernie5WeightNames.MtpHeadFn} holds {stored.Length} values, not {flat} x {hc}.");
            _mtpHeadFn = Ernie5Layer.Transpose(stored, flat, hc);
            _mtpHeadScale = Ernie5Layer.HostF32(source, Ernie5WeightNames.Mtp(mtp, Ernie5WeightNames.MtpHeadScale));
            _mtpHeadBase = Ernie5Layer.HostF32(source, Ernie5WeightNames.Mtp(mtp, Ernie5WeightNames.MtpHeadBase));
            _mtpNorm = Ernie5Layer.HostF32(source, Ernie5WeightNames.Mtp(mtp, Ernie5WeightNames.MtpFinalNorm));
        }
        try
        {
            _embedding = source.Read(Ernie5WeightNames.Embedding);
            _lmHead = source.Read(Ernie5WeightNames.LmHead);
            // Embed casts rows without a block grid, so an FP8 table would read wrong values. The real checkpoint stores it BF16.
            if (_embedding.DType == DType.F8E4M3)
                throw new NotSupportedException($"'{Ernie5WeightNames.Embedding}' is FP8; the embedding gather reads plain values, so it must be BF16 or F32.");
            RequireMatrix(_embedding, config.VocabSize, config.HiddenSize, Ernie5WeightNames.Embedding);
            RequireMatrix(_lmHead, config.VocabSize, config.HiddenSize, Ernie5WeightNames.LmHead);
            if (options.ResidentWeights)
            {
                for (int i = 0; i < _layers.Length; i++) _layers[i] = new Ernie5Layer(_layerBackends[i], config, i, source, _hcaRope, _dsaRope);
                if (options.Mtp) _mtp = new Ernie5MtpLayer(backend, config, config.NumHiddenLayers, source, _hcaRope, _dsaRope);
                foreach (LlmStage stage in placement.Stages)
                    if (stage.Backend.Device.IsGpu) PreloadStage(stage);
            }
        }
        catch
        {
            DisposeWeights();
            throw;
        }
    }

    /// <summary>Opens a safetensors directory and loads it; the model owns the checkpoint.</summary>
    public static Ernie5Model Load(string directory, IBackend backend, Ernie5LoadOptions options)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        Stopwatch clock = Stopwatch.StartNew();
        Logs.Info($"[ernie5] load start: dir={directory} ResidentWeights={options.ResidentWeights} "
            + $"MaxSequenceTokens={options.MaxSequenceTokens}");
        Ernie5Config config = Ernie5Config.Load(Path.Combine(directory, "config.json"));
        Ernie5CheckpointWeights weights = Ernie5CheckpointWeights.Open(directory, config.QuantBlock is [int block, _] ? block : 128);
        Logs.Info($"[ernie5] index read: {weights.Names.Count()} tensors");
        try
        {
            Ernie5Model model = new(backend, config, weights, options, weights);
            Logs.Info($"[ernie5] load returned in {clock.Elapsed.TotalSeconds:F1} s");
            return model;
        }
        catch
        {
            weights.Dispose();
            throw;
        }
    }

    /// <summary>The checkpoint's config.</summary>
    public Ernie5Config Config { get; }

    /// <summary>How the checkpoint's keys mapped onto the model at load.</summary>
    public Ernie5WeightCoverage Coverage { get; }

    /// <summary>A new, empty sequence state of at most <see cref="Ernie5LoadOptions.MaxSequenceTokens"/> tokens.</summary>
    public Ernie5SequenceState CreateState(int capacity)
    {
        if (capacity > _options.MaxSequenceTokens)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity,
                $"The model was loaded for sequences of up to {_options.MaxSequenceTokens} tokens.");
        return new Ernie5SequenceState(Config, capacity, _options.Mtp);
    }

    /// <summary>Appends <paramref name="tokenIds"/> to <paramref name="state"/> and returns the logits, <c>[rows, vocab]</c> after multimax: the last
    /// row only, or every row when <paramref name="allRows"/>.</summary>
    public float[] Forward(ReadOnlySpan<int> tokenIds, Ernie5SequenceState state, bool allRows = false)
    {
        int rows = allRows ? tokenIds.Length : 1;
        return Logits(ForwardHidden(tokenIds, state, allRows), rows);
    }

    /// <summary>Appends <paramref name="tokenIds"/> to <paramref name="state"/> and returns the final normed hidden rows, <c>[rows, hidden]</c>: the
    /// last row only, or every row when <paramref name="allRows"/>.</summary>
    public float[] ForwardHidden(ReadOnlySpan<int> tokenIds, Ernie5SequenceState state, bool allRows = false)
    {
        float[] stream = ForwardStream(tokenIds, state);
        int tokens = tokenIds.Length, rows = allRows ? tokens : 1, flat = Config.HcMult * Config.HiddenSize;
        return ReadOut(stream.AsSpan((tokens - rows) * flat, rows * flat));
    }

    /// <summary>Appends <paramref name="tokenIds"/> to <paramref name="state"/> and returns the residual stream after the last decoder layer, before
    /// the head read-out, for every appended token, <c>[tokens, hc, hidden]</c>: the target hidden state <see cref="MtpForward"/> reads.</summary>
    public float[] ForwardStream(ReadOnlySpan<int> tokenIds, Ernie5SequenceState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int tokens = tokenIds.Length, d = Config.HiddenSize, hc = Config.HcMult, startPos = state.Length;
        RequireAppendable(tokenIds, state, startPos, state.Faulted, nameof(Forward));

        float[] stream = Embed(tokenIds, d, hc);
        string? dumpDir = Ernie5LayerDump.DirectoryFromEnvironment();
        bool completed = false;
        long forwardStart = Stopwatch.GetTimestamp();
        double buildMs = 0, runMs = 0;
        try
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                long layerStart = Stopwatch.GetTimestamp();
                // A start with no matching end line is how a stall inside a layer shows up.
                Logs.Debug($"[ernie5] layer {i + 1}/{_layers.Length} start");
                Ernie5Layer layer = _layers[i] ?? new Ernie5Layer(_layerBackends[i], Config, i, _source, _hcaRope, _dsaRope);
                double layerBuildMs = Stopwatch.GetElapsedTime(layerStart).TotalMilliseconds;
                long runStart = Stopwatch.GetTimestamp();
                try
                {
                    layer.Forward(stream, tokens, startPos, state);
                    if (dumpDir is not null) Ernie5LayerDump.Write(dumpDir, i, startPos, tokens, hc, d, stream);
                }
                finally
                {
                    if (!ReferenceEquals(layer, _layers[i])) layer.Dispose();
                }
                double layerRunMs = Stopwatch.GetElapsedTime(runStart).TotalMilliseconds;
                buildMs += layerBuildMs;
                runMs += layerRunMs;
                Logs.Info($"[ernie5] layer {i + 1}/{_layers.Length}: build {layerBuildMs:F0} ms, forward {layerRunMs:F0} ms "
                    + $"({(_layers[i] is null ? "streamed" : "resident")})");
            }
            completed = true;
        }
        finally
        {
            // Layers that ran already advanced their caches; Length no longer matches them.
            if (!completed) state.MarkFaulted();
        }
        state.Length = startPos + tokens;
        Logs.Info($"[ernie5] forward of {tokens} token(s) at position {startPos}: {Stopwatch.GetElapsedTime(forwardStart).TotalMilliseconds:F0} ms "
            + $"(layer build {buildMs:F0} ms, layer forward {runMs:F0} ms){DeviceSummary()}");
        return stream;
    }

    // One stage's weights into its backend's weight cache. The budget is the device's free memory less a headroom for activations and the
    // per-call uploads, checked here rather than left to the backend's preload, which fails the whole load on an out-of-memory error where the
    // driver does not report a full device first.
    // Without PreloadExperts: each layer's per-token weights in order until the next set would cross the budget (that layer and every later
    // one of the stage stay on the host), then the LM head on the last stage if all its layers fit; routed experts upload as tokens pick them.
    // With PreloadExperts: every weight of the stage, routed experts and block-scale grids included, or the load is refused before anything
    // is placed, naming the stage and what it needs.
    private void PreloadStage(LlmStage stage)
    {
        long start = Stopwatch.GetTimestamp();
        IBackend backend = stage.Backend;
        bool last = ReferenceEquals(stage, _placement.Stages[^1]);
        DeviceMemoryUsage usage = backend.GetDeviceMemoryUsage();
        long budget = PreloadBudget(usage.FreeBytes, usage.TotalBytes), used = 0;
        List<Tensor> placed = [];
        _preloaded.Add((backend, placed));
        int count = stage.EndLayer - stage.StartLayer, layers = 0;
        if (_options.PreloadExperts)
        {
            long need = StageBytes(stage, last);
            if (need > budget)
                throw new NotSupportedException($"ERNIE-5.5 layers [{stage.StartLayer}, {stage.EndLayer}){(last ? " and the LM head" : "")} need "
                    + $"{need / (1 << 20)} MB on {backend.Device}, but its preload budget is {Math.Max(budget, 0) / (1 << 20)} MB "
                    + $"({usage.FreeBytes / (1 << 20)} MB free less headroom). Spread the layers over more devices, or load without PreloadExperts.");
        }
        for (int i = stage.StartLayer; i < stage.EndLayer; i++)
        {
            List<Tensor> weights = [.. _layers[i]!.EnumerateDenseWeights()];
            long bytes = Bytes(weights);
            if (used + bytes > budget) break;
            backend.PreloadWeights(weights);
            placed.AddRange(weights);
            used += bytes;
            layers++;
        }
        bool head = last && layers == count && used + Bytes([_lmHead!]) <= budget;
        if (head)
        {
            backend.PreloadWeights([_lmHead!]);
            placed.Add(_lmHead!);
            used += Bytes([_lmHead!]);
        }
        long experts = 0;
        if (_options.PreloadExperts)
        {
            for (int i = stage.StartLayer; i < stage.EndLayer; i++)
            {
                List<IReadOnlyList<Tensor>> groups = [.. _layers[i]!.EnumerateExpertPreloadGroups()];
                backend.PreloadWeightGroups(groups);
                foreach (IReadOnlyList<Tensor> group in groups)
                {
                    placed.AddRange(group);
                    experts += Bytes(group);
                }
            }
            used += experts;
        }
        Logs.Info($"[ernie5] stage layers [{stage.StartLayer}, {stage.EndLayer}) on {backend.Device}: preloaded {layers}/{count} layers' per-token "
            + $"weights{(head ? ", the LM head" : "")}{(_options.PreloadExperts ? $" and {experts / (1 << 20)} MB of routed experts" : "")}, "
            + $"{used / (1 << 20)} MB of a {Math.Max(budget, 0) / (1 << 20)} MB budget, in {Stopwatch.GetElapsedTime(start).TotalSeconds:F1} s"
            + DeviceSummary(backend));
    }

    // Every weight a stage holds with PreloadExperts: its layers' per-token weights and routed experts with their block-scale grids, and the LM
    // head on the last stage.
    private long StageBytes(LlmStage stage, bool last)
    {
        long bytes = last ? Bytes([_lmHead!]) : 0;
        for (int i = stage.StartLayer; i < stage.EndLayer; i++)
        {
            bytes += Bytes(_layers[i]!.EnumerateDenseWeights());
            foreach (IReadOnlyList<Tensor> group in _layers[i]!.EnumerateExpertPreloadGroups()) bytes += Bytes(group);
        }
        return bytes;
    }

    /// <summary>Device bytes the preload may take: <paramref name="free"/> less a headroom of 15% of <paramref name="total"/>, at least 3 GiB.
    /// Zero when the backend does not report its memory.</summary>
    internal static long PreloadBudget(long free, long total)
    {
        if (free <= 0 || total <= 0) return 0;
        long headroom = Math.Max(3L << 30, total * 15 / 100);
        return Math.Max(free - headroom, 0);
    }

    private static long Bytes(IEnumerable<Tensor> weights) => weights.Sum(t => t.DType.ComputeByteCount(t.ElementCount));

    private string DeviceSummary()
    {
        if (_placement.IsSingle) return DeviceSummary(_backend);
        return string.Concat(_placement.Stages.Select(s => s.Backend).Distinct(ReferenceEqualityComparer.Instance).Cast<IBackend>().Select(DeviceSummary));
    }

    private static string DeviceSummary(IBackend backend)
    {
        if (!backend.Device.IsGpu) return "";
        DeviceMemoryUsage usage = backend.GetDeviceMemoryUsage();
        return $"; {backend.Device} weights {usage.WeightBytes / (1 << 20)} MB, activations {usage.ActivationBytes / (1 << 20)} MB, "
            + $"free {usage.FreeBytes / (1 << 20)} of {usage.TotalBytes / (1 << 20)} MB";
    }

    /// <inheritdoc />
    /// <remarks>The routed experts read the latent, not the model width.</remarks>
    public int ExpertHiddenSize => Config.MoeLatentSize > 0 ? Config.MoeLatentSize : Config.HiddenSize;

    /// <inheritdoc />
    public int ExpertIntermediateSize => Config.MoeIntermediateSize;

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">The model streams its layers (<see cref="Ernie5LoadOptions.ResidentWeights"/> off).</exception>
    public IEnumerable<Tensor> EnumerateExpertWeights()
    {
        RequireResidentForOffload();
        // Every slot is built once ResidentWeights holds.
        return _layers.Select(l => l?.Moe).OfType<MoeFeedForward>().SelectMany(m => m.EnumerateExpertWeights());
    }

    /// <inheritdoc />
    /// <remarks>Attach before the first forward: a backend that keeps reused weights on its own may already hold experts the cache then
    /// cannot evict. The MTP layer's experts are not routed through the offload.</remarks>
    /// <exception cref="InvalidOperationException">The model streams its layers (<see cref="Ernie5LoadOptions.ResidentWeights"/> off): a
    /// streamed layer's experts are new tensors on every forward, so a cache keyed by them would never hit.</exception>
    /// <exception cref="NotSupportedException">The layers are placed over several stages: one offload serves one device.</exception>
    public IReadOnlyList<(int Layer, int ExpertCount)> AttachExpertOffload(MoeExpertOffload offload)
    {
        ArgumentNullException.ThrowIfNull(offload);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        RequireResidentForOffload();
        if (!_placement.IsSingle)
            throw new NotSupportedException("An expert offload serves one device; this model's layers are placed over several stages.");
        List<(int Layer, int ExpertCount)> layers = [];
        foreach (Ernie5Layer? layer in _layers)
        {
            if (layer?.Moe is not { } moe) continue;
            moe.AttachOffload(offload, layer.Index);
            layers.Add((layer.Index, moe.NumExperts));
        }
        return layers;
    }

    private void RequireResidentForOffload()
    {
        if (!_options.ResidentWeights)
            throw new InvalidOperationException("Expert offload needs ResidentWeights: streamed layers build new expert tensors on every forward.");
    }

    /// <summary>The mHC head read-out and final RMS norm of residual-stream rows from <see cref="ForwardStream"/>, <c>[rows, hidden]</c>.</summary>
    public float[] ReadOut(ReadOnlySpan<float> stream)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int rows = StreamRows(stream, nameof(stream));
        float[] hidden = new float[rows * Config.HiddenSize];
        _head.Head(stream, rows, _headFn, _headScale, _headBase, _finalNorm, hidden);
        return hidden;
    }

    /// <summary>
    /// Runs the multi-token-prediction layer on <paramref name="tokenIds"/> at positions from <see cref="Ernie5SequenceState.MtpLength"/>,
    /// appending to its cache. Row <c>i</c> mixes <paramref name="previous"/> row <c>i</c> (the target stream of the token at that position,
    /// from <see cref="ForwardStream"/>, or an earlier draft step's output) with the embedding of <paramref name="tokenIds"/>[i], the token after
    /// it. Returns the layer's output stream, <c>[tokens, hc, hidden]</c>: <see cref="MtpHidden"/> reads a draft from it, and a further draft
    /// step takes it as <paramref name="previous"/>.
    /// </summary>
    /// <remarks>Row <c>i</c> runs at rope position <c>MtpLength + i</c>, which must be the position of the target token whose stream it
    /// reads, as in vLLM, where the draft takes the target's positions. Keeping <see cref="Ernie5SequenceState.MtpLength"/> in step with
    /// <see cref="Ernie5SequenceState.Length"/> is the caller's job: the first draft covers the positions the target has committed, and each
    /// chained step adds one past them. A failure here faults only the MTP cache; the target keeps decoding.</remarks>
    /// <exception cref="InvalidOperationException">The model was loaded without <see cref="Ernie5LoadOptions.Mtp"/>.</exception>
    public float[] MtpForward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<float> previous, Ernie5SequenceState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        RequireMtp();
        if (!state.HasMtp) throw new ArgumentException("The sequence state has no MTP caches; create it with CreateState.", nameof(state));
        int tokens = tokenIds.Length, d = Config.HiddenSize, startPos = state.MtpLength;
        RequireAppendable(tokenIds, state, startPos, state.MtpFaulted, nameof(MtpForward));
        if (StreamRows(previous, nameof(previous)) != tokens)
            throw new ArgumentException($"previous must hold one stream row per token ({tokens}).", nameof(previous));

        float[] embedded = Embed(tokenIds, d, 1);
        Ernie5MtpLayer layer = _mtp ?? new Ernie5MtpLayer(_backend, Config, Config.NumHiddenLayers, _source, _hcaRope, _dsaRope);
        bool completed = false;
        try
        {
            float[] output = layer.Forward(previous, embedded, tokens, startPos, state);
            completed = true;
            state.MtpLength = startPos + tokens;
            return output;
        }
        finally
        {
            if (!ReferenceEquals(layer, _mtp)) layer.Dispose();
            // The MTP cache already advanced; MtpLength no longer matches it. The target caches are untouched.
            if (!completed) state.MarkMtpFaulted();
        }
    }

    /// <summary>The MTP layer's read-out of stream rows from <see cref="MtpForward"/>: its own mHC head and <c>shared_head.norm</c>,
    /// <c>[rows, hidden]</c>. <see cref="Logits"/> turns it into draft logits through the shared LM head and multimax.</summary>
    /// <exception cref="InvalidOperationException">The model was loaded without <see cref="Ernie5LoadOptions.Mtp"/>.</exception>
    public float[] MtpHidden(ReadOnlySpan<float> stream)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        RequireMtp();
        int rows = StreamRows(stream, nameof(stream));
        float[] hidden = new float[rows * Config.HiddenSize];
        _head.Head(stream, rows, _mtpHeadFn!, _mtpHeadScale!, _mtpHeadBase!, _mtpNorm!, hidden);
        return hidden;
    }

    /// <summary>The LM head and multimax over <paramref name="rows"/> hidden rows from <see cref="ForwardHidden"/>, <c>[rows, vocab]</c>.</summary>
    public float[] Logits(ReadOnlySpan<float> hidden, int rows)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int d = Config.HiddenSize, vocab = Config.VocabSize;
        if (rows < 1 || hidden.Length != rows * d)
            throw new ArgumentException($"hidden must hold {rows} rows of {d} values.", nameof(hidden));
        using Tensor hiddenT = HostTensor(hidden, rows, d);
        using Tensor logitsT = new(new TensorShape(rows, vocab), DType.F32);
        _backend.Linear(logitsT, hiddenT, _lmHead!, null);
        float[] logits = logitsT.AsReadOnlySpan<float>().ToArray();
        if (_multimaxRanges is not null) MultimaxHead.Apply(logits, _multimaxRanges, _multimaxGains!);
        return logits;
    }

    // Each embedding row repeated `copies` times. With hc copies this is the first layer's broadcast pre-mix (vLLM mhc_pre_broadcast),
    // which equals the ordinary pre-mix over hc identical copies; the MTP layer takes one copy.
    private float[] Embed(ReadOnlySpan<int> tokenIds, int d, int copies)
    {
        float[] stream = new float[tokenIds.Length * copies * d];
        for (int t = 0; t < tokenIds.Length; t++)
        {
            using Tensor row = _embedding!.SliceRows(tokenIds[t], 1);
            using Tensor f32 = row.DType == DType.F32 ? row.To(row.Device) : row.CastTo(DType.F32);
            ReadOnlySpan<float> values = f32.AsReadOnlySpan<float>();
            for (int c = 0; c < copies; c++) values.CopyTo(stream.AsSpan((t * copies + c) * d, d));
        }
        return stream;
    }

    private void RequireAppendable(ReadOnlySpan<int> tokenIds, Ernie5SequenceState state, int startPos, bool faulted, string call)
    {
        if (faulted) throw new InvalidOperationException($"An earlier {call} on this state failed partway through; call Reset() before reusing it.");
        if (tokenIds.Length < 1) throw new ArgumentException($"{call} needs at least one token.", nameof(tokenIds));
        if (startPos + tokenIds.Length > state.Capacity)
            throw new InvalidOperationException($"The sequence state holds {state.Capacity} tokens; {startPos + tokenIds.Length} were asked for.");
        foreach (int id in tokenIds)
            if ((uint)id >= (uint)Config.VocabSize) throw new ArgumentOutOfRangeException(nameof(tokenIds), id, "Token id is outside the vocabulary.");
    }

    private void RequireMtp()
    {
        if (!_options.Mtp) throw new InvalidOperationException("The model was loaded without Ernie5LoadOptions.Mtp, so it has no MTP layer.");
    }

    private int StreamRows(ReadOnlySpan<float> stream, string name)
    {
        int flat = Config.HcMult * Config.HiddenSize;
        if (stream.Length < flat || stream.Length % flat != 0)
            throw new ArgumentException($"{name} must hold whole residual-stream rows of {flat} values.", name);
        return stream.Length / flat;
    }

    private static Tensor HostTensor(ReadOnlySpan<float> values, int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static void RequireMatrix(Tensor t, int rows, int cols, string name)
    {
        if (t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
            throw new InvalidDataException($"'{name}' is {t.Shape}; the config needs [{rows}, {cols}].");
    }

    private void DisposeWeights()
    {
        // A preloaded weight's device copy is released by name, on the backend that holds it, not by disposing the tensor.
        foreach ((IBackend backend, List<Tensor> weights) in _preloaded) backend.FreeWeights(weights);
        _preloaded.Clear();
        foreach (Ernie5Layer? layer in _layers) layer?.Dispose();
        Array.Clear(_layers);
        Interlocked.Exchange(ref _mtp, null)?.Dispose();
        Interlocked.Exchange(ref _embedding, null)?.Dispose();
        Interlocked.Exchange(ref _lmHead, null)?.Dispose();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        DisposeWeights();
        _ownedSource?.Dispose();
    }
}
