using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Looped;

/// <summary>Adapts a <see cref="LoopedTransformer"/> on one backend to <see cref="IGenerationModel"/>: sequence states carry the loop count they
/// were created for (<see cref="LoopedSequenceState"/>), batched decode groups sequences by loop count, and the single-token step captures as
/// one graph on a backend that supports it.</summary>
/// <remarks>Does not own the wrapped transformer or backend: <see cref="Dispose"/> releases nothing.</remarks>
public sealed class LoopedTransformerModel : IGenerationModel, IGraphDecodable
{
    // Speculative verify batches this many draft tokens per pass; larger only adds rejected work.
    private const int MaxDraftTokens = 16;

    private readonly LoopedTransformer _transformer;
    private readonly IBackend _backend;
    private readonly bool _preloadWeights;

    public GenerationModelInfo Info { get; }

    public GenerationCapabilities Capabilities { get; }

    public IBackend OutputBackend => _backend;

    /// <summary>The architecture driven.</summary>
    public LoopedTransformerConfig Config => _transformer.Config;

    /// <summary>Wraps <paramref name="transformer"/> on <paramref name="backend"/>; <paramref name="preloadWeights"/> uploads every weight that fits, headroom-guarded, now and again before graph capture.</summary>
    public LoopedTransformerModel(LoopedTransformer transformer, IBackend backend, bool preloadWeights = false)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(backend);
        _transformer = transformer;
        _backend = backend;
        _preloadWeights = preloadWeights;
        LoopedTransformerConfig cfg = transformer.Config;
        Info = new GenerationModelInfo(cfg.Architecture ?? nameof(LoopedTransformer), cfg.VocabSize, cfg.HiddenSize, cfg.PhysicalLayers,
            cfg.MaxPositionEmbeddings);
        Capabilities = new GenerationCapabilities
        {
            SupportsSpeculation = true,
            MaxSpeculativeDepth = MaxDraftTokens,
            SupportsBatchDecode = true,
        };
        if (_preloadWeights) PreloadDecodeWeights();
    }

    /// <summary>The loop count <paramref name="options"/> asks for: its own when set, else the checkpoint default; checked against the model's maximum.</summary>
    public int ResolveLoops(in SequenceStateOptions options)
    {
        int loops = options.Loops > 0 ? options.Loops : Config.Loops;
        if (loops > Config.EffectiveMaxLoops)
            throw new ArgumentOutOfRangeException(nameof(options), loops, $"This model runs at most {Config.EffectiveMaxLoops} loops.");
        return loops;
    }

    public ISequenceState CreateSequenceState(SequenceStateOptions options)
    {
        int loops = ResolveLoops(options);
        int slots = LoopedSchedule.MaxSlotCount(Config);
        if (options.Pool is not null)
        {
            if (options.Pool.NumLayers < slots)
                throw new ArgumentException($"The KV pool has {options.Pool.NumLayers} slots; this model needs {slots}.", nameof(options));
            return new LoopedSequenceState(new PagedKvCache(options.Pool), loops);
        }
        DType kvDtype = KvCaches.F16For(_backend) ? DType.F16 : DType.F32;
        return new LoopedSequenceState(new FixedKvCache(slots, 1, Config.NumKvHeads, Config.HeadDim, options.MaxSequenceTokens, kvDtype), loops);
    }

    /// <summary>A retained state serves only a request with the same loop count: the recurrent slots hold loop-specific keys.</summary>
    public bool AcceptsRetainedState(ISequenceState state, in SequenceStateOptions options) =>
        state is LoopedSequenceState looped && looped.Loops == ResolveLoops(options);

    /// <summary>Copies a fixed cache layer by layer on its device, keeping the loop count; null for any other state.</summary>
    public ISequenceState? ResizeSequenceState(ISequenceState state, int capacity) =>
        state is LoopedSequenceState { Inner: FixedKvCache cache } looped ? new LoopedSequenceState(cache.CopyWithCapacity(capacity), looped.Loops) : null;

    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => Prefill(chunk, state, CancellationToken.None);

    /// <summary>Stops between virtual layers once <paramref name="cancel"/> is signalled; see <see cref="IGenerationModel.Prefill(in PrefillChunk, ISequenceState, CancellationToken)"/>.</summary>
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
    {
        LoopedSequenceState looped = AsLooped(state);
        ReadOnlySpan<int> tokenIds = chunk.TokenIds.Span;
        int t = tokenIds.Length;
        Tensor hidden = chunk.Embeds is not null
            ? _transformer.ForwardEmbeds(_backend, chunk.Embeds, t, chunk.PosStart, looped, looped.Loops, cancel)
            : _transformer.Forward(_backend, tokenIds, chunk.PosStart, looped, looped.Loops, cancel);
        if (!chunk.LastRowOnly || t <= 1) return hidden;
        // Sampling reads one row: slice before the head projection so it runs at t = 1 (the fused quantized GEMV path).
        Tensor lastHidden = new(new TensorShape(1, 1, Info.HiddenSize), DType.F32);
        if (!_backend.TryGatherRowsResident(lastHidden, hidden, [t - 1]))
            _backend.GatherRows(lastHidden, hidden, [t - 1]);
        hidden.Dispose();
        return lastHidden;
    }

    /// <summary>One decode token per sequence; sequences created for different loop counts run as separate groups and their rows are
    /// reassembled in the caller's order.</summary>
    public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states)
    {
        int n = tokenIds.Length;
        if (states.Length != n) throw new ArgumentException($"{n} tokens for {states.Length} sequence states.");
        LoopedSequenceState[] looped = new LoopedSequenceState[n];
        for (int b = 0; b < n; b++) looped[b] = AsLooped(states[b]);
        using Tensor embeds = new(new TensorShape(1, n, Info.HiddenSize), DType.F32);
        _transformer.EmbedLookup(embeds, tokenIds);
        bool uniform = true;
        for (int b = 1; b < n && uniform; b++) uniform = looped[b].Loops == looped[0].Loops;
        if (uniform)
        {
            int[] positions = new int[n];
            IKvCache[] caches = new IKvCache[n];
            for (int b = 0; b < n; b++)
            {
                caches[b] = looped[b];
                positions[b] = looped[b].CurrentLength;
            }
            return _transformer.ForwardBatchDecode(_backend, embeds, positions, caches, looped[0].Loops);
        }
        return DecodeGrouped(embeds, looped);
    }

    /// <summary>Runs one forward per distinct loop count over the rows that share it and gathers the results back into batch order.</summary>
    private Tensor DecodeGrouped(Tensor embeds, LoopedSequenceState[] looped)
    {
        int n = looped.Length;
        int hidden = Info.HiddenSize;
        List<int> order = [];
        List<Tensor> outputs = [];
        try
        {
            foreach (int loops in looped.Select(s => s.Loops).Distinct().Order())
            {
                int[] members = Enumerable.Range(0, n).Where(b => looped[b].Loops == loops).ToArray();
                int[] positions = new int[members.Length];
                IKvCache[] caches = new IKvCache[members.Length];
                for (int i = 0; i < members.Length; i++)
                {
                    caches[i] = looped[members[i]];
                    positions[i] = looped[members[i]].CurrentLength;
                }
                using Tensor groupEmbeds = new(new TensorShape(1, members.Length, hidden), DType.F32);
                _backend.GatherRows(groupEmbeds, embeds, members);
                outputs.Add(_transformer.ForwardBatchDecode(_backend, groupEmbeds, positions, caches, loops));
                order.AddRange(members);
            }
            using Tensor joined = new(new TensorShape(1, n, hidden), DType.F32);
            _backend.Concat(joined, [.. outputs], dim: 1);
            // Row r of the joined tensor belongs to sequence order[r]; invert that so result row b is sequence b.
            int[] inverse = new int[n];
            for (int r = 0; r < n; r++) inverse[order[r]] = r;
            Tensor result = new(new TensorShape(1, n, hidden), DType.F32);
            _backend.GatherRows(result, joined, inverse);
            return result;
        }
        finally
        {
            foreach (Tensor t in outputs) t.Dispose();
        }
    }

    public Tensor ProjectLogits(Tensor hidden, int rows) => _transformer.ProjectLogits(_backend, hidden, rows);

    public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => _transformer.EnumerateWeights();

    /// <remarks>Sized for every cache slot the largest loop count uses, so a shorter request is admitted conservatively.</remarks>
    public long EstimateSequenceBytes(int contextTokens)
    {
        long elementBytes = KvCaches.F16For(_backend) ? 2 : 4;
        long perToken = 2L * LoopedSchedule.MaxSlotCount(Config) * Config.NumKvHeads * Config.HeadDim * elementBytes;
        return perToken * contextTokens;
    }

    public CapacitySnapshot Capacity()
    {
        (long free, long total) = _backend.GetVramInfo();
        return new CapacitySnapshot(free, total);
    }

    bool IGraphDecodable.SupportsDeviceSampling(IBackend backend) => backend.DeviceSamplingSupported;

    bool IGraphDecodable.SupportsGraphDecode(IBackend backend) => _transformer.SupportsGraphDecode(backend);

    string? IGraphDecodable.GraphDecodeRefusal(IBackend backend) => _transformer.GraphDecodeRefusal(backend);

    GraphDecodeSession IGraphDecodable.CaptureDecodeGraph(ISequenceState state, int pos, int firstToken, float repetitionPenalty, DeviceSamplerConfig? sampler)
    {
        if (state is not LoopedSequenceState { Inner: FixedKvCache cache } looped)
            throw new ArgumentException("Graph decode captures against a looped sequence over a FixedKvCache.", nameof(state));
        int loops = looped.Loops;

        // A cold model's first capture fails when a weight's lazy first-touch upload lands mid-capture; a real single-token forward and
        // head projection against a throwaway cache moves that allocation outside the capture region.
        using (FixedKvCache warmupCache = new(cache.NumLayers, 1, Config.NumKvHeads, Config.HeadDim, maxSequenceLength: 2))
        using (LoopedSequenceState warmup = new(warmupCache, loops))
        {
            using Tensor warmupHidden = _transformer.Forward(_backend, [firstToken], 0, warmup, loops);
            _transformer.ProjectLogits(_backend, warmupHidden, 1).Dispose();
        }
        // Stragglers uploaded inside the capture bake a memcpy node that replays every token.
        if (_preloadWeights) PreloadDecodeWeights();

        Tensor embedTable = _transformer.EnsureEmbedResidentForGraphDecode(_backend);
        _transformer.EnsureRopeTablesForGraphDecode(_backend, cache.MaxSequenceLength);
        ulong devicePos = _backend.AllocDevicePos();
        ulong deviceTokenId = _backend.AllocDeviceTokenId();
        ulong history = _backend.AllocDeviceHistory(cache.MaxSequenceLength);
        ulong historyCount = _backend.AllocDeviceCounter();
        ulong rng = sampler is { } samplerConfig ? _backend.AllocDeviceRng(samplerConfig.Seed, samplerConfig.TopK) : 0;
        bool stagedRow = _transformer.StagesGraphDecodeEmbedRows(_backend);
        ulong embedRow = 0;
        object? graph = null;
        try
        {
            if (stagedRow) embedRow = _backend.AllocDecodeEmbedRow(Config.HiddenSize);
            _backend.WriteDeviceTokenId(deviceTokenId, firstToken);
            _backend.WriteDevicePos(devicePos, pos + 1, pos);
            _backend.WriteDeviceCounter(historyCount, 0);
            graph = _backend.CaptureGraph(() =>
                _transformer.ForwardGraphDecodeStep(_backend, embedTable, cache, loops, devicePos, deviceTokenId, history, historyCount,
                    repetitionPenalty, sampler, rng, embedRow));
            return new GraphDecodeSession(_backend, graph!, devicePos, deviceTokenId, history, historyCount, pos, rng,
                embedRow, Config.HiddenSize, firstToken, stagedRow ? _transformer : null);
        }
        catch
        {
            if (graph is not null) _backend.DisposeGraph(graph);
            _backend.FreeDecodeEmbedRow(embedRow);
            _backend.FreeDevicePos(devicePos);
            _backend.FreeDeviceTokenId(deviceTokenId);
            _backend.FreeDeviceHistory(history);
            _backend.FreeDeviceCounter(historyCount);
            _backend.FreeDeviceRng(rng);
            throw;
        }
    }

    void IGraphDecodable.CommitReplayedStep(ISequenceState state) => AsLooped(state).AdvanceLength(1);

    public void Dispose()
    {
    }

    private static LoopedSequenceState AsLooped(ISequenceState state) =>
        state as LoopedSequenceState ?? throw new ArgumentException($"{state.GetType().Name} is not a looped sequence state.", nameof(state));

    /// <summary>Uploads every weight that fits while 2 GB (plus graph decode's embed table) stays free; large stragglers stay lazy.</summary>
    private void PreloadDecodeWeights()
    {
        try
        {
            long headroom = 2L << 30;
            if (_transformer.SupportsGraphDecode(_backend)) headroom += _transformer.GraphDecodeEmbedExtraBytes(_backend);
            DecodeWeightPreload.Preload(_backend, _transformer.EnumerateWeights(), headroom);
        }
        catch (Exception ex) { Logs.Warning($"weight preload failed (continuing with lazy residency): {ex.Message}"); }
    }
}
