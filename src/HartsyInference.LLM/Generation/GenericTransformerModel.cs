using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>Adapts a <see cref="GenericTransformer"/> (with its backend and optional layer-split placement) to <see cref="IGenerationModel"/>, owning the staged/plain forward switch, weight preload and graph-capture warmup that drivers used to duplicate.</summary>
/// <remarks>Does not own the wrapped transformer or backend: <see cref="Dispose"/> releases nothing.</remarks>
public sealed class GenericTransformerModel : IGenerationModel, IGraphDecodable
{
    // Speculative verify batches this many draft tokens per pass; larger only adds rejected work.
    private const int MaxDraftTokens = 16;

    private readonly GenericTransformer _transformer;
    private readonly IBackend _backend;
    private readonly LlmPlacement? _placement;
    private readonly bool _preloadWeights;

    public GenerationModelInfo Info { get; }

    public GenerationCapabilities Capabilities { get; }

    public IBackend OutputBackend => _backend;

    private bool Staged => _placement is not null && !_placement.IsSingle;

    /// <summary>Under a sharded <paramref name="placement"/> the output backend is its last stage's; <paramref name="preloadWeights"/> uploads every weight that fits, headroom-guarded, now and again before graph capture.</summary>
    public GenericTransformerModel(GenericTransformer transformer, IBackend backend, LlmPlacement? placement = null,
        bool preloadWeights = false)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(backend);
        _transformer = transformer;
        _placement = placement;
        _backend = placement is not null ? placement.LastBackend : backend;
        _preloadWeights = preloadWeights;
        TransformerConfig cfg = transformer.Config;
        Info = new GenerationModelInfo(nameof(GenericTransformer), cfg.VocabSize, cfg.HiddenSize, cfg.NumLayers,
            cfg.MaxPositionEmbeddings);
        Capabilities = new GenerationCapabilities
        {
            NeedsTokenIdsWithEmbeds = cfg.PerLayerEmbeddingDim > 0,
            SupportsSpeculation = !Staged,
            MaxSpeculativeDepth = MaxDraftTokens,
            SupportsBatchDecode = !Staged,
        };
        if (_preloadWeights) PreloadDecodeWeights();
    }

    public ISequenceState CreateSequenceState(SequenceStateOptions options)
    {
        if (options.Pool is not null) return new PagedKvCache(options.Pool);
        TransformerConfig cfg = _transformer.Config;
        // Gemma-4's local layers are narrower than its global ones; HeadDimsPerLayer is uniform elsewhere.
        DType kvDtype = KvCaches.F16Enabled && !options.FullPrecisionKv ? DType.F16 : DType.F32;
        return new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), options.MaxSequenceTokens, kvDtype);
    }

    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state)
    {
        IKvCache cache = AsKvCache(state);
        ReadOnlySpan<int> tokenIds = chunk.TokenIds.Span;
        int t = tokenIds.Length;
        Tensor hidden;
        if (chunk.Embeds is not null)
        {
            hidden = Staged
                ? _transformer.ForwardEmbedsStaged(_placement!, chunk.Embeds, t, chunk.PosStart, cache, tokenIds)
                : _transformer.ForwardEmbeds(_backend, chunk.Embeds, t, chunk.PosStart, cache, tokenIds: tokenIds);
        }
        else
        {
            hidden = Staged
                ? _transformer.ForwardStaged(_placement!, tokenIds, chunk.PosStart, cache)
                : _transformer.Forward(_backend, tokenIds, chunk.PosStart, cache);
        }
        if (!chunk.LastRowOnly || t <= 1) return hidden;

        // Sampling reads one row, so slicing BEFORE the head projection skips (t-1)/t of the vocab GEMM and makes
        // the projection t=1, which takes the fused quantized GEMV path instead of dequant + cuBLAS.
        Tensor lastHidden = new(new TensorShape(1, 1, Info.HiddenSize), DType.F32);
        _backend.GatherRows(lastHidden, hidden, [t - 1]);
        hidden.Dispose();
        return lastHidden;
    }

    public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states)
    {
        if (Staged) throw new NotSupportedException("Batch decode is not available for a staged placement.");
        int n = tokenIds.Length;
        if (states.Length != n) throw new ArgumentException($"{n} tokens for {states.Length} sequence states.");
        int[] positions = new int[n];
        IKvCache[] caches = new IKvCache[n];
        for (int b = 0; b < n; b++)
        {
            caches[b] = AsKvCache(states[b]);
            positions[b] = caches[b].CurrentLength;
        }
        using Tensor embeds = new(new TensorShape(1, n, Info.HiddenSize), DType.F32);
        _transformer.EmbedLookup(embeds, tokenIds);
        return _transformer.ForwardBatchDecode(_backend, embeds, positions, caches);
    }

    public Tensor ProjectLogits(Tensor hidden, int rows) => _transformer.ProjectLogits(_backend, hidden, rows);

    public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) =>
        _transformer.EnumerateWeights(includeRedundantSplits);

    public long EstimateSequenceBytes(int contextTokens)
    {
        TransformerConfig cfg = _transformer.Config;
        long elementBytes = KvCaches.F16Enabled ? 2 : 4;
        long perToken = 0;
        foreach (int headDim in cfg.HeadDimsPerLayer()) perToken += 2L * cfg.NumKvHeads * headDim * elementBytes;
        return perToken * contextTokens;
    }

    public CapacitySnapshot Capacity()
    {
        (long free, long total) = _backend.GetVramInfo();
        return new CapacitySnapshot(free, total);
    }

    bool IGraphDecodable.SupportsGraphDecode(IBackend backend) => !Staged && _transformer.SupportsGraphDecode(backend);

    GraphDecodeSession IGraphDecodable.CaptureDecodeGraph(ISequenceState state, int pos, int firstToken, float repetitionPenalty)
    {
        if (state is not FixedKvCache cache)
            throw new ArgumentException("Graph decode captures against a FixedKvCache.", nameof(state));
        TransformerConfig cfg = _transformer.Config;

        // A cold model's first capture fails with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED when a weight's lazy
        // first-touch upload lands mid-capture; a real single-token forward and head projection against a
        // throwaway cache moves that allocation outside the capture region.
        using (FixedKvCache warmup = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), maxSequenceLength: 2))
        {
            using Tensor warmupHidden = _transformer.Forward(_backend, [firstToken], 0, warmup);
            _transformer.ProjectLogits(_backend, warmupHidden, 1).Dispose();
        }

        // Stragglers uploaded inside the capture bake a memcpy node that replays every token.
        if (_preloadWeights) PreloadDecodeWeights();

        Tensor embedTable = _transformer.EnsureEmbedResidentForGraphDecode(_backend);
        _transformer.EnsurePleResidentForGraphDecode(_backend);
        (Tensor cosTable, Tensor sinTable) = _transformer.EnsureRopeTableForGraphDecode(_backend, cache.MaxSequenceLength);
        ulong devicePos = _backend.AllocDevicePos();
        ulong deviceTokenId = _backend.AllocDeviceTokenId();
        ulong history = _backend.AllocDeviceHistory(cache.MaxSequenceLength);
        ulong historyCount = _backend.AllocDeviceCounter();
        object? graph = null;
        try
        {
            _backend.WriteDeviceTokenId(deviceTokenId, firstToken);
            _backend.WriteDevicePos(devicePos, pos + 1, pos);
            _backend.WriteDeviceCounter(historyCount, 0);
            graph = _backend.CaptureGraph(() =>
                _transformer.ForwardGraphDecodeStep(_backend, embedTable, cache, cosTable, sinTable, devicePos,
                    deviceTokenId, history, historyCount, repetitionPenalty));
            return new GraphDecodeSession(_backend, graph!, devicePos, deviceTokenId, history, historyCount, pos);
        }
        catch
        {
            if (graph is not null) _backend.DisposeGraph(graph);
            _backend.FreeDevicePos(devicePos);
            _backend.FreeDeviceTokenId(deviceTokenId);
            _backend.FreeDeviceHistory(history);
            _backend.FreeDeviceCounter(historyCount);
            throw;
        }
    }

    void IGraphDecodable.CommitReplayedStep(ISequenceState state) => AsKvCache(state).AdvanceLength(1);

    public void Dispose()
    {
    }

    private static IKvCache AsKvCache(ISequenceState state) =>
        state as IKvCache ?? throw new ArgumentException($"{state.GetType().Name} is not a KV cache.", nameof(state));

    /// <summary>Uploads every transformer weight that fits while leaving 2 GB free (large stragglers stay lazy); idempotent, already-cached tensors are skipped.</summary>
    private void PreloadDecodeWeights()
    {
        try
        {
            // 2 GB base plus the F32 embed table graph decode later materializes, reserved only for graph-eligible
            // models: reserving it for graph-excluded ones halved their eager decode (91.7 -> 48.8 tok/s).
            TransformerConfig cfg = _transformer.Config;
            long headroom = 2L << 30;
            if (_transformer.SupportsGraphDecode(_backend))
                headroom += (long)cfg.VocabSize * cfg.HiddenSize * sizeof(float);
            // No extra prefill slack: it pushed weights out of the budget and unresident weights bake per-token
            // PCIe memcpy nodes into a captured graph (GLM 43.8 -> 27.9 tok/s).
            if (Staged)
            {
                StagedWeightPreload.Preload(_transformer, _placement!, headroom - (2L << 30));
                return;
            }
            if (_backend.FreeMemoryBytes() is long free && free > headroom)
            {
                List<Tensor> toPreload = [];
                long budget = free - headroom;
                foreach (Tensor t in _transformer.EnumerateWeights(includeRedundantSplits: false))
                {
                    long bytes = Tensor.ComputeByteSize(t.Shape, t.DType);
                    if (budget - bytes < 0) continue;
                    budget -= bytes;
                    toPreload.Add(t);
                }
                long skipped = 0;
                int skippedCount = 0;
                foreach (Tensor t in _transformer.EnumerateWeights(includeRedundantSplits: false))
                {
                    long b = Tensor.ComputeByteSize(t.Shape, t.DType);
                    if (!toPreload.Contains(t)) { skipped += b; skippedCount++; }
                }
                Logs.Info($"[preload] free={_backend.FreeMemoryBytes() >> 20}MB headroom={headroom >> 20}MB " +
                    $"kept={toPreload.Count} skipped={skippedCount} ({skipped >> 20}MB left lazy)");
                _backend.PreloadWeights(toPreload);
            }
        }
        catch (Exception ex) { Logs.Warning($"weight preload failed (continuing with lazy residency): {ex.Message}"); }
    }
}
