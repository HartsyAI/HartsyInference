using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Looped;

/// <summary>Config-driven looped (latent-recurrent) normalized-GPT decoder: NVIDIA Nemotron Dory's spine. The physical layers
/// (<see cref="LoopedLayer"/>) run in the order a <see cref="LoopedSchedule"/> unrolls for the request's loop count, the input block's
/// output is re-injected and unit-normed before every pass of the recurrent block, and the logits are the untied head scaled
/// per vocabulary entry. No final norm: every layer leaves the residual on the unit sphere.</summary>
/// <remarks>Built for full device residency like <see cref="GenericTransformer"/>: every op is an <see cref="IBackend"/> call, weights are
/// stable references for <see cref="IBackend.PreloadWeights"/>, and keys/values live in an <see cref="IKvCache"/> whose slots a schedule
/// assigns per attention layer and loop.</remarks>
public sealed unsafe class LoopedTransformer : IDisposable, IGraphDecodeRowSource
{
    /// <summary>Layers a cancellable forward lets the device have in flight (see <see cref="DeviceRunAhead"/>).</summary>
    private const int RunAheadLayers = 2;

    private readonly LoopedTransformerConfig _cfg;
    private readonly LoopedLayer[] _layers;
    private readonly LoopedNormVectors _norms;
    private readonly LoopedSchedule?[] _schedules;
    // Distinct RoPE bases in first-use order, and each physical layer's index into them (-1 for an MLP layer).
    private readonly float[] _thetas;
    private readonly int[] _thetaIndex;
    private readonly bool _usesWindow;
    private Tensor? _embedRaw;         // the loaded table, whatever its dtype
    private Tensor? _embed;            // F32 host table for the gather; never uploaded (the head is untied)
    private Tensor? _embedGatherQuant; // the loaded quantized table, gathered on device by graph decode when the backend can
    private bool _embedRowsStaged;
    private Tensor? _lmHead;
    private Tensor? _logitScale;
    private Tensor?[] _graphCos;
    private Tensor?[] _graphSin;
    private int _graphCapacity;
    private int _disposed;

    /// <summary>The architecture this transformer was built for.</summary>
    public LoopedTransformerConfig Config => _cfg;

    /// <summary>Creates the transformer for <paramref name="cfg"/> (weights loaded separately).</summary>
    public LoopedTransformer(LoopedTransformerConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        cfg.Validate();
        _cfg = cfg;
        _norms = new LoopedNormVectors(cfg);
        _layers = new LoopedLayer[cfg.PhysicalLayers];
        for (int p = 0; p < cfg.PhysicalLayers; p++) _layers[p] = new LoopedLayer(cfg, _norms, p);
        _schedules = new LoopedSchedule?[cfg.EffectiveMaxLoops + 1];
        List<float> thetas = [];
        _thetaIndex = new int[cfg.PhysicalLayers];
        for (int p = 0; p < cfg.PhysicalLayers; p++)
        {
            if (!cfg.IsAttention(p)) { _thetaIndex[p] = -1; continue; }
            int index = thetas.IndexOf(cfg.RopeThetas[p]);
            if (index < 0) { index = thetas.Count; thetas.Add(cfg.RopeThetas[p]); }
            _thetaIndex[p] = index;
            _usesWindow |= cfg.IsSliding(p);
        }
        _thetas = [.. thetas];
        _graphCos = new Tensor?[_thetas.Length];
        _graphSin = new Tensor?[_thetas.Length];
    }

    /// <summary>Loads weights from an HF-style key dict; <paramref name="prefix"/> is everything up to (not including) <c>embed_tokens</c> (e.g. <c>model</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        ThrowIfDisposed();
        Tensor embedRaw = w[$"{prefix}.embed_tokens.weight"];
        if (embedRaw.ElementCount != (long)_cfg.VocabSize * _cfg.HiddenSize)
            throw new ArgumentException($"Embedding table {embedRaw.Shape} does not match vocab {_cfg.VocabSize} × hidden {_cfg.HiddenSize}.");
        _embedRaw = embedRaw;
        _embed = ProjectionOps.EnsureF32(embedRaw);
        if (embedRaw.DType == DType.Q8_0 || embedRaw.DType == DType.Q5_0 || embedRaw.DType == DType.Q4_K || embedRaw.DType == DType.Q6_K)
            _embedGatherQuant = embedRaw;
        // The head is untied, so a quantized table is staged from the host row for graph decode instead of being uploaded.
        _embedRowsStaged = embedRaw.DType.IsQuantized;
        _lmHead = w["lm_head.weight"];
        if (_lmHead.Shape.Rank != 2 || _lmHead.Shape[0] != _cfg.VocabSize || _lmHead.Shape[1] != _cfg.HiddenSize)
            throw new ArgumentException($"lm_head.weight is {_lmHead.Shape}; expected [{_cfg.VocabSize}, {_cfg.HiddenSize}].");
        _logitScale = BakeLogitScale(w["lm_head.logit_scale.weight"]);
        for (int p = 0; p < _layers.Length; p++) _layers[p].LoadWeights(w, $"{prefix}.layers.{p}");
    }

    /// <summary>All weight tensors (stable references) for <see cref="IBackend.PreloadWeights"/>/<see cref="IBackend.FreeWeights"/>; the
    /// host-gathered embedding table is not among them.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_lmHead is not null) yield return _lmHead;
        if (_logitScale is not null) yield return _logitScale;
        foreach (Tensor t in _norms.Enumerate()) yield return t;
        foreach (LoopedLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights()) yield return t;
    }

    /// <summary>Host embedding gather into <paramref name="output"/> <c>[1, T, hidden]</c>.</summary>
    public void EmbedLookup(Tensor output, ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        int h = _cfg.HiddenSize;
        float* op = (float*)output.DataPointer;
        float* ep = (float*)_embed!.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int id = tokenIds[s];
            if ((uint)id >= (uint)_cfg.VocabSize)
                throw new ArgumentException($"token id {id} out of range [0, {_cfg.VocabSize}).");
            Buffer.MemoryCopy(ep + (long)id * h, op + (long)s * h, h * 4, h * 4);
        }
    }

    /// <summary>Prefill or decode over <paramref name="tokenIds"/> at positions <paramref name="posStart"/>.. with <paramref name="loops"/> passes of the
    /// recurrent block, appending to <paramref name="cache"/>; returns the last layer's residual <c>[1, T, hidden]</c> (no final norm).</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int posStart, IKvCache cache, int loops, CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        int t = tokenIds.Length;
        using Tensor embeds = new(new TensorShape(1, t, _cfg.HiddenSize), DType.F32);
        EmbedLookup(embeds, tokenIds);
        return ForwardEmbeds(backend, embeds, t, posStart, cache, loops, cancel);
    }

    /// <summary>As <see cref="Forward"/> from pre-built embeddings <paramref name="embeds"/> <c>[1, T, hidden]</c>, which the caller keeps.</summary>
    public Tensor ForwardEmbeds(IBackend backend, Tensor embeds, int t, int posStart, IKvCache cache, int loops, CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        LoopedSchedule schedule = Schedule(loops);
        RequireSlots(cache, schedule);
        cancel.ThrowIfCancellationRequested();
        int d = _cfg.HeadDim;
        Tensor[] cos = new Tensor[_thetas.Length];
        Tensor[] sin = new Tensor[_thetas.Length];
        DeviceRunAhead runAhead = new(backend, stackalloc nint[RunAheadLayers], cancel.CanBeCanceled);
        Tensor? hidden = null;
        Tensor? inputH = null;
        try
        {
            for (int i = 0; i < _thetas.Length; i++)
            {
                Tensor c = new(new TensorShape(1, t, d), DType.F32);
                Tensor s = new(new TensorShape(1, t, d), DType.F32);
                RopeTables.BuildRope(c, s, t, posStart, d, d, _thetas[i], RopeScaling.None);
                (cos[i], sin[i]) = ProjectionOps.MakeRopeTableResident(backend, c, s);
            }
            hidden = ProjectionOps.MakeResident(backend, embeds);
            for (int v = 0; v < schedule.VirtualLayers; v++)
            {
                cancel.ThrowIfCancellationRequested();
                runAhead.BeforeStep();
                cancel.ThrowIfCancellationRequested();
                hidden = EnterLayer(backend, schedule, v, hidden, ref inputH);
                int p = schedule.Physical(v);
                int theta = _thetaIndex[p];
                Tensor next = _layers[p].Forward(backend, hidden, t, posStart, cache, schedule.CacheSlot(v),
                    theta < 0 ? null : cos[theta], theta < 0 ? null : sin[theta]);
                runAhead.AfterStep();
                hidden = LeaveLayer(backend, schedule, v, hidden, next, ref inputH);
            }
            cache.AdvanceLength(t);
            Tensor result = hidden;
            hidden = null;
            return result;
        }
        finally
        {
            runAhead.Dispose();
            if (hidden is not null && !ReferenceEquals(hidden, inputH)) hidden.Dispose();
            inputH?.Dispose();
            foreach (Tensor c in cos) c?.Dispose();
            foreach (Tensor s in sin) s?.Dispose();
        }
    }

    /// <summary>Batched decode: <paramref name="embeds"/> <c>[1, B, hidden]</c> holds one token per sequence, <paramref name="positions"/>[b] its absolute
    /// position and <paramref name="caches"/>[b] its cache; every sequence runs <paramref name="loops"/> passes. Returns the residual <c>[1, B, hidden]</c>
    /// and advances each cache by one.</summary>
    public Tensor ForwardBatchDecode(IBackend backend, Tensor embeds, ReadOnlySpan<int> positions, IKvCache[] caches, int loops)
    {
        ThrowIfDisposed();
        int b = (int)embeds.Shape[1];
        if (positions.Length != b || caches.Length != b)
            throw new ArgumentException($"positions ({positions.Length}) / caches ({caches.Length}) must match batch B={b}.");
        LoopedSchedule schedule = Schedule(loops);
        foreach (IKvCache cache in caches) RequireSlots(cache, schedule);
        int d = _cfg.HeadDim;
        Tensor[] cos = new Tensor[_thetas.Length];
        Tensor[] sin = new Tensor[_thetas.Length];
        Tensor? hidden = null;
        Tensor? inputH = null;
        try
        {
            for (int i = 0; i < _thetas.Length; i++)
            {
                Tensor c = new(new TensorShape(1, b, d), DType.F32);
                Tensor s = new(new TensorShape(1, b, d), DType.F32);
                RopeTables.BuildRopeBatched(c, s, positions, d, d, _thetas[i], RopeScaling.None);
                (cos[i], sin[i]) = ProjectionOps.MakeRopeTableResident(backend, c, s);
            }
            hidden = ProjectionOps.MakeResident(backend, embeds);
            for (int v = 0; v < schedule.VirtualLayers; v++)
            {
                hidden = EnterLayer(backend, schedule, v, hidden, ref inputH);
                int p = schedule.Physical(v);
                int theta = _thetaIndex[p];
                Tensor next = _layers[p].ForwardBatchDecode(backend, hidden, b, theta < 0 ? null : cos[theta], theta < 0 ? null : sin[theta],
                    caches, schedule.CacheSlot(v));
                hidden = LeaveLayer(backend, schedule, v, hidden, next, ref inputH);
            }
            foreach (IKvCache cache in caches) cache.AdvanceLength(1);
            Tensor result = hidden;
            hidden = null;
            return result;
        }
        finally
        {
            if (hidden is not null && !ReferenceEquals(hidden, inputH)) hidden.Dispose();
            inputH?.Dispose();
            foreach (Tensor c in cos) c?.Dispose();
            foreach (Tensor s in sin) s?.Dispose();
        }
    }

    /// <summary>Projects the residual <c>[1, T, hidden]</c> to logits <c>[1, T, vocab]</c>: the untied head times the per-vocabulary logit scale.</summary>
    public Tensor ProjectLogits(IBackend backend, Tensor hidden, int t)
    {
        ThrowIfDisposed();
        Tensor logits = new(new TensorShape(1, t, _cfg.VocabSize), DType.F32);
        ProjectionOps.ProjectHead(backend, logits, hidden, _lmHead!, _cfg.HiddenSize, t, _cfg.LowVramQuant);
        backend.AffineBroadcastLastDim(logits, logits, _logitScale!, null);
        return logits;
    }

    /// <summary>True when the single-token step captures on <paramref name="backend"/>: a backend with device-position ops, and one whose graph
    /// attention takes a sliding window when the model has sliding layers.</summary>
    public bool SupportsGraphDecode(IBackend backend) => backend.LlmGraphDecodeSupported && (!_usesWindow || backend.LlmGraphDecodeWindowSoftcap);

    /// <summary>Why an explicit graph-decode request must be refused on <paramref name="backend"/>, or null: only the sliding window can be in the way.</summary>
    public string? GraphDecodeRefusal(IBackend backend)
    {
        if (!backend.LlmGraphDecodeSupported || !_usesWindow || backend.LlmGraphDecodeWindowSoftcap) return null;
        return $"{_cfg.Architecture ?? "this model"}: sliding-window attention is not supported by this backend's decode graph; decode eagerly or use CUDA";
    }

    /// <summary>True when graph decode embeds from a host row staged before each replay (a quantized table the eager gather never uploads).</summary>
    public bool StagesGraphDecodeEmbedRows(IBackend backend) => _embedRowsStaged && backend.StagesDecodeEmbedRows;

    /// <summary>Makes the embedding table graph decode gathers from device-resident and returns it; the host table when rows are staged instead.</summary>
    public Tensor EnsureEmbedResidentForGraphDecode(IBackend backend)
    {
        ThrowIfDisposed();
        if (StagesGraphDecodeEmbedRows(backend)) return _embed!;
        if (_embedGatherQuant is { } quant && QuantEmbedGatherFits(backend, quant))
        {
            backend.PreloadWeights([quant]);
            return quant;
        }
        backend.PreloadWeights([_embed!]);
        return _embed!;
    }

    /// <summary>Device bytes graph decode adds for its embed table beyond the decode weights: nothing for a staged row, the table for a gathered one.</summary>
    public long GraphDecodeEmbedExtraBytes(IBackend backend)
    {
        if (StagesGraphDecodeEmbedRows(backend)) return 0;
        if (_embedGatherQuant is { } quant && QuantEmbedGatherFits(backend, quant)) return Tensor.ComputeByteSize(quant.Shape, quant.DType);
        return (long)_cfg.VocabSize * _cfg.HiddenSize * sizeof(float);
    }

    /// <inheritdoc/>
    public nint GraphDecodeEmbedRow(int tokenId)
    {
        ThrowIfDisposed();
        if ((uint)tokenId >= (uint)_cfg.VocabSize)
            throw new ArgumentException($"token id {tokenId} out of range [0, {_cfg.VocabSize}).", nameof(tokenId));
        return (nint)((float*)_embed!.DataPointer + (long)tokenId * _cfg.HiddenSize);
    }

    /// <summary>Builds (or grows) the device RoPE tables graph decode reads, one pair per distinct base frequency, for positions up to <paramref name="minCapacity"/>.</summary>
    /// <remarks>A rebuild only replaces these fields: a graph that captured the old tables keeps them alive by reference, and a new capture
    /// only happens while no other graph-sessioned sequence is active (the scheduler's solo gate), so no live graph loses its table.</remarks>
    public void EnsureRopeTablesForGraphDecode(IBackend backend, int minCapacity)
    {
        ThrowIfDisposed();
        if (_graphCos[0] is not null && _graphCapacity >= minCapacity) return;
        Tensor?[] oldCos = _graphCos;
        Tensor?[] oldSin = _graphSin;
        Tensor?[] newCos = new Tensor?[_thetas.Length];
        Tensor?[] newSin = new Tensor?[_thetas.Length];
        for (int i = 0; i < _thetas.Length; i++)
            (newCos[i], newSin[i]) = backend.BuildRopeTableDevice(minCapacity, _cfg.HeadDim, _cfg.HeadDim, _thetas[i], RopeScaling.None);
        _graphCos = newCos;
        _graphSin = newSin;
        _graphCapacity = minCapacity;
        foreach (Tensor? t in oldCos) t?.Dispose();
        foreach (Tensor? t in oldSin) t?.Dispose();
    }

    /// <summary>One full decode step (embed → every virtual layer → logits → [repetition penalty] → greedy argmax or device sampling) using the
    /// device-indexed ops throughout, so the whole step captures into one graph. <paramref name="deviceTokenId"/> is read at the start and written at
    /// the end, so replays chain on the device. Caller guarantees <see cref="SupportsGraphDecode"/>, <see cref="EnsureRopeTablesForGraphDecode"/> and a
    /// refreshed <paramref name="devicePos"/>.</summary>
    public void ForwardGraphDecodeStep(IBackend backend, Tensor embedTable, IKvCache cache, int loops, ulong devicePos, ulong deviceTokenId,
        ulong history = 0, ulong historyCount = 0, float repetitionPenalty = 1.0f, DeviceSamplerConfig? sampler = null, ulong rngState = 0,
        ulong embedRow = 0)
    {
        ThrowIfDisposed();
        LoopedSchedule schedule = Schedule(loops);
        RequireSlots(cache, schedule);
        if (_graphCos[0] is null) throw new InvalidOperationException("Call EnsureRopeTablesForGraphDecode before capturing a decode step.");
        bool applyRepPenalty = repetitionPenalty != 1.0f && history != 0 && historyCount != 0;
        if (applyRepPenalty) backend.AppendTokenHistoryStep(history, historyCount, deviceTokenId);

        Tensor embeds = new(new TensorShape(1, 1, _cfg.HiddenSize), DType.F32);
        if (StagesGraphDecodeEmbedRows(backend))
        {
            if (embedRow == 0) throw new InvalidOperationException("A quantized embed table needs its staged embedding row to graph-decode.");
            backend.EmbedStagedRowDecodeStep(embeds, embedRow);
        }
        else if (embedTable.DType.IsQuantized)
            backend.EmbedGatherQuantDecodeStep(embeds, embedTable, 1f, deviceTokenId);
        else
            backend.EmbedGatherDecodeStep(embeds, embedTable, deviceTokenId);

        Tensor hidden = embeds;
        Tensor? inputH = null;
        for (int v = 0; v < schedule.VirtualLayers; v++)
        {
            hidden = EnterLayer(backend, schedule, v, hidden, ref inputH);
            int p = schedule.Physical(v);
            int theta = _thetaIndex[p];
            Tensor next = _layers[p].ForwardGraphStep(backend, hidden, cache, schedule.CacheSlot(v),
                theta < 0 ? null : _graphCos[theta], theta < 0 ? null : _graphSin[theta], devicePos);
            hidden = LeaveLayer(backend, schedule, v, hidden, next, ref inputH);
        }
        inputH?.Dispose();
        Tensor logits = ProjectLogits(backend, hidden, 1);
        hidden.Dispose();
        if (applyRepPenalty) backend.ApplyRepetitionPenaltyStep(logits, history, historyCount, repetitionPenalty);
        if (sampler is { } s && rngState != 0) backend.SampleTopKInto(deviceTokenId, logits, s.TopK, s.Temperature, s.TopP, s.MinP, rngState);
        else backend.ArgMaxInto(deviceTokenId, logits);
        logits.Dispose();
    }

    /// <summary>The unrolled order for <paramref name="loops"/> passes, built once per loop count.</summary>
    public LoopedSchedule Schedule(int loops)
    {
        if (loops < 1 || loops > _cfg.EffectiveMaxLoops)
            throw new ArgumentOutOfRangeException(nameof(loops), loops, $"Loops must be in [1, {_cfg.EffectiveMaxLoops}] for this model.");
        return _schedules[loops] ??= new LoopedSchedule(_cfg, loops);
    }

    /// <summary>Before virtual layer <paramref name="v"/> runs: captures the input block's output when the recurrence begins, and at the start of
    /// every pass re-injects it (<c>norm(state + input)</c>). Returns the layer's input; <paramref name="inputH"/> stays alive across the passes.</summary>
    private Tensor EnterLayer(IBackend backend, LoopedSchedule schedule, int v, Tensor hidden, ref Tensor? inputH)
    {
        if (v == schedule.RecurrentStart) inputH = hidden;
        if (!schedule.IsLoopStart(v)) return hidden;
        using Tensor sum = new(hidden.Shape, DType.F32);
        backend.Add(sum, hidden, inputH!);
        Tensor normed = HypersphereOps.Normalized(backend, sum, _norms.Hidden, _cfg.NormEps);
        if (!ReferenceEquals(hidden, inputH)) hidden.Dispose();
        return normed;
    }

    /// <summary>After virtual layer <paramref name="v"/> produced <paramref name="next"/> from <paramref name="hidden"/>: releases the input, re-norms the
    /// state at the end of a pass, and releases the captured input once the recurrence is over. Returns the running residual.</summary>
    private Tensor LeaveLayer(IBackend backend, LoopedSchedule schedule, int v, Tensor hidden, Tensor next, ref Tensor? inputH)
    {
        if (!ReferenceEquals(hidden, inputH)) hidden.Dispose();
        Tensor result = next;
        if (schedule.IsLoopEnd(v))
        {
            result = HypersphereOps.Normalized(backend, next, _norms.Hidden, _cfg.NormEps);
            next.Dispose();
        }
        if (v == schedule.RecurrentEnd - 1 && inputH is not null)
        {
            inputH.Dispose();
            inputH = null;
        }
        return result;
    }

    private static void RequireSlots(IKvCache cache, LoopedSchedule schedule)
    {
        if (cache.NumLayers < schedule.SlotsUsed)
            throw new ArgumentException($"The cache has {cache.NumLayers} slots; {schedule.Loops} loops need {schedule.SlotsUsed}.", nameof(cache));
    }

    private bool QuantEmbedGatherFits(IBackend backend, Tensor quant)
    {
        int block = quant.DType.BlockElementCount;
        int blockBytes = quant.DType.BlockByteSize;
        if (_cfg.HiddenSize % block != 0) return false;
        long rows = quant.ElementCount / _cfg.HiddenSize;
        long rowBytes = (long)(_cfg.HiddenSize / block) * blockBytes;
        return rows * rowBytes <= uint.MaxValue && backend.CanGatherQuantEmbed(Tensor.ComputeByteSize(quant.Shape, quant.DType));
    }

    /// <summary>The per-vocabulary logit multiplier: the stored vector divided by <see cref="LoopedTransformerConfig.InitStd"/>.</summary>
    private Tensor BakeLogitScale(Tensor source)
    {
        Tensor f32 = ProjectionOps.EnsureF32(source);
        try
        {
            if (f32.ElementCount != _cfg.VocabSize)
                throw new ArgumentException($"lm_head.logit_scale has {f32.ElementCount} values; expected {_cfg.VocabSize}.");
            Tensor baked = new(new TensorShape(_cfg.VocabSize), DType.F32);
            float* src = (float*)f32.DataPointer;
            float* dst = (float*)baked.DataPointer;
            float inv = 1f / _cfg.InitStd;
            for (int i = 0; i < _cfg.VocabSize; i++) dst[i] = src[i] * inv;
            return baked;
        }
        finally
        {
            if (!ReferenceEquals(f32, source)) f32.Dispose();
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(LoopedTransformer));
    }

    /// <summary>Releases the tensors this transformer built (baked scales, fused projections, norm vectors, graph tables); checkpoint weights stay with their loader.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (LoopedLayer layer in _layers) layer.Dispose();
        _norms.Dispose();
        _logitScale?.Dispose();
        foreach (Tensor? t in _graphCos) t?.Dispose();
        foreach (Tensor? t in _graphSin) t?.Dispose();
        if (_embed is not null && !ReferenceEquals(_embed, _embedRaw)) _embed.Dispose();
    }
}
