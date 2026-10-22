using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>One attention layer composed of stages: projection, Q/K norm and rotation, the attention call
/// (docs/ATTENTION_CONTRACT.md), output gate and projection; each intermediate is disposed after its last use.</summary>
public sealed partial class AttentionLayer
{
    private readonly AttentionLayerConfig _config;
    private readonly AttentionLayerWeights? _weights;
    private readonly AttentionNormWeights? _norms;

    private static readonly ConcurrentDictionary<(int Heads, int HeadDim, float Scale, DType? ComputeDType), AttentionLayer> SharedLayers = new();

    /// <summary>The weightless layer of <paramref name="heads"/> equal query and key/value heads for a site that builds its own operands,
    /// one per geometry and shared, as it holds no state; <paramref name="scale"/> 0 means <c>1/√headDim</c>.</summary>
    public static AttentionLayer Shared(int heads, int headDim, float scale = 0f, DType? computeDType = null) =>
        SharedLayers.GetOrAdd((heads, headDim, scale, computeDType), static key => new AttentionLayer(new AttentionLayerConfig
        {
            Heads = key.Heads, KvHeads = key.Heads, HeadDim = key.HeadDim, Scale = key.Scale, ComputeDType = key.ComputeDType,
        }));

    /// <summary>Creates a layer. Without <paramref name="weights"/> only <see cref="Attend"/> and <see cref="AttendFlat"/> can
    /// run, and only without a QK norm. Weights without a query projection serve <see cref="AttendBranchHeadMajor"/> alone.</summary>
    public AttentionLayer(AttentionLayerConfig config, AttentionLayerWeights? weights = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        if (weights is not null)
        {
            if (config.Gate != AttentionGate.None && weights.Gate is null)
                throw new ArgumentException($"A {config.Gate} gate needs a gate projection.", nameof(weights));
            bool fused = config.Projection is AttentionProjection.FusedSlice or AttentionProjection.FusedSplitNorm;
            if (fused ? weights.Qkv is null : weights.Key is null || weights.Value is null)
                throw new ArgumentException($"A {config.Projection} projection needs its weights.", nameof(weights));
            bool hasQuery = fused || weights.Query is not null;
            if (config.QkNorm != AttentionQkNorm.None && (hasQuery && weights.QueryNorm is null || weights.KeyNorm is null))
                throw new ArgumentException($"A {config.QkNorm} QK norm needs the norm weights of the projections the weights have.", nameof(weights));
            if (config.Projection == AttentionProjection.FusedSplitNorm && weights.QueryNormEps != weights.KeyNormEps)
                throw new ArgumentException("The fused split-norm projection takes one epsilon for Q and K.", nameof(weights));
            if (config.QkNorm != AttentionQkNorm.None)
                _norms = new AttentionNormWeights(weights.QueryNorm, weights.KeyNorm, weights.QueryNormEps, weights.KeyNormEps);
        }
        _config = config;
        _weights = weights;
    }

    /// <summary>Creates a layer for a model that projects and outputs through its own code but norms Q and K here.</summary>
    public AttentionLayer(AttentionLayerConfig config, AttentionNormWeights norms)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        if (config.QkNorm == AttentionQkNorm.None)
            throw new ArgumentException("Norm weights need a QK norm in the config.", nameof(norms));
        ArgumentNullException.ThrowIfNull(norms.Query);
        ArgumentNullException.ThrowIfNull(norms.Key);
        _config = config;
        _norms = norms;
    }

    /// <summary>The layer's configuration.</summary>
    public AttentionLayerConfig Config => _config;

    /// <summary>Attention of <paramref name="qInput"/> <c>[B, S, C]</c> or <c>[S, C]</c> over <paramref name="kvInput"/> (the same
    /// tensor for self-attention). Returns the output projection, shaped like the query input with the output width; the caller
    /// owns it.</summary>
    public Tensor Forward(IBackend backend, Tensor qInput, Tensor kvInput, in AttentionCallContext context)
    {
        AttentionLayerWeights w = _weights ?? throw new InvalidOperationException("This attention layer has no weights; call Attend.");
        if (w.Output is null)
            throw new InvalidOperationException("This attention layer has no output projection; project it with ProjectSelf.");
        int rank = qInput.Shape.Rank;
        int batch = rank >= 3 ? (int)qInput.Shape[0] : 1;
        int sq = (int)qInput.Shape[rank - 2];
        int skv = (int)kvInput.Shape[kvInput.Shape.Rank - 2];
        DType act = _config.ComputeDType ?? qInput.DType;
        DType kvAct = _config.ComputeDType ?? kvInput.DType;
        ValidateContext(in context, batch);
        if (context.ValueRowScale is { } rowScale)
        {
            if (_config.Projection == AttentionProjection.GroupedByInput)
                throw new NotSupportedException("A value row-scale needs per-weight projections.");
            // A scale built for other geometry would make Mul read past the smaller buffer rather than fail.
            TensorShape vShape = ProjectedShape(kvInput.Shape, _config.KvHeads);
            if (rowScale.ElementCount != vShape.ElementCount || rowScale.DType != kvAct)
                throw new ArgumentException($"Value-row scale is {rowScale.Shape} {rowScale.DType} but v is {vShape} {kvAct}.",
                    nameof(context));
        }

        if (context.ReleaseQueryAfterProjection && (context.ReleaseInputs || ReferenceEquals(qInput, kvInput)
            || _config.Projection != AttentionProjection.PerWeight || _config.Gate != AttentionGate.None || context.ValueRowScale is not null))
            throw new ArgumentException("Releasing the query after its projection needs a distinct K/V input, per-weight projections, "
                + "no gate, no value row-scale and no ReleaseInputs.", nameof(context));
        TensorShape flatShape = WithLastDim(qInput.Shape, _config.InnerDim);
        (Tensor q, Tensor k, Tensor v, Tensor? gate) = context.ReleaseQueryAfterProjection
            ? ProjectReleasingQuery(backend, w, qInput, kvInput, act, kvAct)
            : Project(backend, w, qInput, kvInput, act, kvAct, context.ReleaseInputs, in context);
        Tensor flat;
        try
        {
            flat = Transform(backend, q, k, v, batch, sq, skv, act, flatShape, AttentionShape(flatShape), FlatOperands, in context);
        }
        catch
        {
            gate?.Dispose();
            throw;
        }
        return Output(backend, w, flat, gate, act, flatShape);
    }

    /// <summary>The transform and attention stages for a model that projects and outputs through its own code: token-major
    /// <paramref name="q"/> <c>[B, S, heads, D]</c>, <paramref name="k"/> and <paramref name="v"/> <c>[B, Skv, kvHeads, D]</c>,
    /// all three taken over and disposed. Returns the attention output in <paramref name="flatShape"/> on either route; the caller
    /// owns it.</summary>
    public Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, TensorShape flatShape, in AttentionCallContext context)
    {
        if (_config.QkNorm != AttentionQkNorm.None && _norms is null)
            throw new InvalidOperationException("A QK norm needs the layer's norm weights.");
        if (q.Shape.Rank != 4 || k.Shape.Rank != 4 || v.Shape.Rank != 4)
            throw new ArgumentException($"Attend takes [B, S, heads, D] operands; got {q.Shape}, {k.Shape}, {v.Shape}.");
        ValidateContext(in context, (int)q.Shape[0]);
        return Transform(backend, q, k, v, (int)q.Shape[0], (int)q.Shape[1], (int)k.Shape[1], q.DType, flatShape, flatShape, flat: false,
            in context);
    }

    /// <summary><see cref="Attend"/> over flat operands: <paramref name="q"/> <c>[B, S, heads·D]</c>, <paramref name="k"/> and
    /// <paramref name="v"/> <c>[B, Skv, kvHeads·D]</c> (a joint sequence, or a full-width norm's layout); rank-2
    /// <c>[S, heads·D]</c> operands are one batch item.</summary>
    public Tensor AttendFlat(IBackend backend, Tensor q, Tensor k, Tensor v, TensorShape flatShape, in AttentionCallContext context)
    {
        if (_config.QkNorm is not (AttentionQkNorm.None or AttentionQkNorm.FullWidthRms))
            throw new NotSupportedException($"Flat operands cannot take a {_config.QkNorm} QK norm.");
        if (_config.QkNorm != AttentionQkNorm.None && _norms is null)
            throw new InvalidOperationException("A QK norm needs the layer's norm weights.");
        if (context.QueryRotation.Kind is not (QkRotationKind.None or QkRotationKind.InterleavedPairs)
            || context.KeyRotation.Kind is not (QkRotationKind.None or QkRotationKind.InterleavedPairs))
            throw new NotSupportedException("Flat operands take interleaved-pair tables or a rotation adapter; the other table kinds read [.., heads, D].");
        int rank = q.Shape.Rank;
        if (rank is not (2 or 3) || k.Shape.Rank != rank || v.Shape.Rank != rank || q.Shape[rank - 1] != _config.InnerDim
            || k.Shape[rank - 1] != (long)_config.KvHeads * _config.HeadDim || v.Shape[rank - 1] != k.Shape[rank - 1])
            throw new ArgumentException($"AttendFlat takes [B, S, heads·D] or [S, heads·D] operands of this layer's widths; got {q.Shape}, {k.Shape}, {v.Shape}.");
        int batch = rank == 3 ? (int)q.Shape[0] : 1;
        ValidateContext(in context, batch);
        return Transform(backend, q, k, v, batch, (int)q.Shape[rank - 2], (int)k.Shape[rank - 2], q.DType, flatShape, flatShape, flat: true,
            in context);
    }

    /// <summary>The attention call alone over operands the caller laid out head-major: <paramref name="qMh"/> <c>[B, heads, Sq, D]</c>,
    /// <paramref name="kMh"/> and <paramref name="vMh"/> <c>[B, heads, Skv, D]</c>. Applies no transform, so it takes only the bias and
    /// probe of the context. Disposes the operands named in <paramref name="consume"/> once the call has run, also when the call
    /// throws (a refused argument takes nothing over); returns the head-major <c>[B, heads, Sq, D]</c> output, which the caller owns. A caller-provided
    /// <paramref name="output"/> of that shape and dtype is written and returned instead, and is never disposed.</summary>
    public Tensor AttendHeadMajorOperands(IBackend backend, Tensor qMh, Tensor kMh, Tensor vMh, in AttentionCallContext context,
        OperandOwnership consume = OperandOwnership.All, Tensor? output = null)
    {
        if (_config.KvHeads != _config.Heads)
            throw new NotSupportedException("Head-major operands carry one head per query head; grouped key/value heads attend through Attend.");
        ValidateOperandContext(in context);
        if (qMh.Shape.Rank != 4 || kMh.Shape.Rank != 4 || vMh.Shape.Rank != 4)
            throw new ArgumentException($"AttendHeadMajorOperands takes [B, heads, S, D] operands; got {qMh.Shape}, {kMh.Shape}, {vMh.Shape}.");
        if (qMh.Shape[1] != _config.Heads || qMh.Shape[3] != _config.HeadDim || kMh.Shape[0] != qMh.Shape[0]
            || kMh.Shape[1] != _config.Heads || kMh.Shape[3] != _config.HeadDim || !vMh.Shape.Equals(kMh.Shape))
            throw new ArgumentException($"Operands do not match {_config.Heads} heads of {_config.HeadDim}: {qMh.Shape}, {kMh.Shape}, {vMh.Shape}.");
        DType act = _config.ComputeDType ?? qMh.DType;
        ValidateOutput(output, qMh, kMh, vMh, act);
        return RunAttention(backend, qMh, kMh, vMh, act, in context, consume, output);
    }

    /// <summary>The attention call over a head-major key/value cache through the Flash entry: <paramref name="qMh"/>
    /// <c>[B, heads, Sq, D]</c>, the caches <c>[B, kvHeads, capacity, D]</c> of which <paramref name="view"/> names the valid rows
    /// and the query position, grouped heads native. Applies no transform, so the context carries at most a probe. Disposes the
    /// operands named in <paramref name="consume"/> (the query by default) once the call has run, also when the call throws (a refused argument takes nothing over); returns
    /// the head-major <c>[B, heads, Sq, D]</c> output, which the caller owns. A caller-provided <paramref name="output"/> is
    /// written and returned instead, and is never disposed.</summary>
    public Tensor AttendCachedHeadMajor(IBackend backend, Tensor qMh, Tensor keyCache, Tensor valueCache, in AttentionCacheView view,
        in AttentionCallContext context = default, OperandOwnership consume = OperandOwnership.Query, Tensor? output = null)
    {
        ValidateOperandContext(in context);
        if (context.Bias is not null)
            throw new ArgumentException("The Flash entry takes no additive bias.", nameof(context));
        if (view.DevicePositions != 0 && view.Sinks is not null)
            throw new ArgumentException("Device positions cannot carry sinks.", nameof(view));
        if (qMh.Shape.Rank != 4 || keyCache.Shape.Rank != 4 || !valueCache.Shape.Equals(keyCache.Shape))
            throw new ArgumentException($"AttendCachedHeadMajor takes [B, heads, S, D] operands; got {qMh.Shape}, {keyCache.Shape}, {valueCache.Shape}.");
        if (qMh.Shape[1] != _config.Heads || qMh.Shape[3] != _config.HeadDim || keyCache.Shape[0] != qMh.Shape[0]
            || keyCache.Shape[1] != _config.KvHeads || keyCache.Shape[3] != _config.HeadDim)
            throw new ArgumentException($"Operands do not match {_config.Heads}/{_config.KvHeads} heads of {_config.HeadDim}: {qMh.Shape}, {keyCache.Shape}.");
        if (view.KvLength < 0 || view.KvLength > keyCache.Shape[2])
            throw new ArgumentException($"The cache view names {view.KvLength} keys of a {keyCache.Shape[2]}-row cache.", nameof(view));
        DType act = _config.ComputeDType ?? qMh.DType;
        ValidateOutput(output, qMh, keyCache, valueCache, act);
        bool owned = output is null;
        Tensor attn = output ?? new(new TensorShape(qMh.Shape[0], _config.Heads, qMh.Shape[2], _config.HeadDim), act);
        try
        {
            AttentionCall call = view.DevicePositions != 0
                ? AttentionCall.FromFlashAttentionDev(attn, qMh, keyCache, valueCache, view.KvLength, view.Causal, view.QueryOffset,
                    _config.EffectiveScale, view.DevicePositions, view.Softcap, view.SlidingWindow)
                : AttentionCall.FromFlashAttention(attn, qMh, keyCache, valueCache, view.KvLength, view.Causal, view.QueryOffset,
                    _config.EffectiveScale, view.Softcap, view.Sinks, view.SlidingWindow, null);
            backend.Attention(in call);
        }
        catch
        {
            if (owned) attn.Dispose();
            throw;
        }
        finally
        {
            Consume(consume, qMh, keyCache, valueCache);
        }
        context.Probe?.Observe("attention", attn);
        return attn;
    }

    /// <summary>The projection stage alone, for self-attention: Q, K and V of <paramref name="input"/>, which is disposed once
    /// they are read when <paramref name="releaseInput"/> is set. The caller owns the three tensors.</summary>
    public (Tensor Q, Tensor K, Tensor V) ProjectSelf(IBackend backend, Tensor input, bool releaseInput)
    {
        AttentionLayerWeights w = _weights ?? throw new InvalidOperationException("This attention layer has no projection weights.");
        if (_config.Gate != AttentionGate.None)
            throw new NotSupportedException("A gated layer projects through Forward.");
        DType act = _config.ComputeDType ?? input.DType;
        (Tensor q, Tensor k, Tensor v, _) = Project(backend, w, input, input, act, act, releaseInput, default);
        return (q, k, v);
    }

    /// <summary>The QK norm stage alone: <paramref name="q"/> and <paramref name="k"/> become their normed copies and the
    /// originals are disposed; nothing happens without a norm.</summary>
    public void NormalizeQk(IBackend backend, ref Tensor q, ref Tensor k)
    {
        if (_config.QkNorm == AttentionQkNorm.None || _config.Projection == AttentionProjection.FusedSplitNorm)
            return;
        if (_config.QkNorm == AttentionQkNorm.FullWidth)
            throw new NotSupportedException("The fused full-width norm runs inside the attention call.");
        AttentionNormWeights n = _norms ?? throw new InvalidOperationException("A QK norm needs the layer's norm weights.");
        q = NormPerHead(backend, q, Have(n.Query), n.QueryEps);
        k = NormPerHead(backend, k, Have(n.Key), n.KeyEps);
    }

    /// <summary>The output stage alone: the output projection of <paramref name="flat"/> (<c>[.., heads·D]</c> or
    /// <c>[B, S, heads, D]</c>), which is disposed.</summary>
    public Tensor ProjectOutput(IBackend backend, Tensor flat)
    {
        AttentionLayerWeights w = _weights ?? throw new InvalidOperationException("This attention layer has no projection weights.");
        if (w.Output is null)
            throw new InvalidOperationException("This attention layer has no output projection.");
        if (_config.Gate != AttentionGate.None)
            throw new NotSupportedException("A gated layer projects its output through Forward.");
        TensorShape logical = flat.Shape.Rank == 4 ? new TensorShape(flat.Shape[0], flat.Shape[1], flat.Shape[2] * flat.Shape[3]) : flat.Shape;
        return Output(backend, w, flat, null, flat.DType, logical);
    }

    /// <summary>The query stage of a shared-query branch: Q of <paramref name="qInput"/> (<c>[B, S, C]</c> or <c>[S, C]</c>), normed
    /// and rotated per the config and context, permuted to head-major <c>[B, heads, S, D]</c>. Disposes <paramref name="qInput"/>
    /// when <paramref name="releaseInput"/> is set, right after the projection. The caller owns the result.</summary>
    public Tensor ProjectQueryHeadMajor(IBackend backend, Tensor qInput, bool releaseInput, in AttentionCallContext context)
    {
        AttentionLayerWeights w = BranchWeights(in context);
        if (w.Query is null)
            throw new InvalidOperationException("This attention layer's weights have no query projection.");
        int rank = qInput.Shape.Rank;
        int batch = rank >= 3 ? (int)qInput.Shape[0] : 1;
        int sq = (int)qInput.Shape[rank - 2];
        DType act = _config.ComputeDType ?? qInput.DType;
        Tensor q = new(BranchShape(qInput.Shape, _config.Heads), act);
        Tensor? qMh = null;
        try
        {
            backend.Linear(q, qInput, w.Query!, w.QueryBias);
            if (releaseInput) qInput.Dispose();
            if (_config.QkNorm != AttentionQkNorm.None)
            {
                AttentionNormWeights n = _norms!.Value;
                q = NormPerHead(backend, q, Have(n.Query), n.QueryEps);
            }
            RotateWithTables(backend, q, context.QueryRotation, _config.Heads);
            qMh = new(new TensorShape(batch, _config.Heads, sq, _config.HeadDim), q.DType);
            backend.Permute0213(qMh, q, sq, _config.Heads, _config.HeadDim);
        }
        catch
        {
            qMh?.Dispose();
            q.Dispose();
            throw;
        }
        q.Dispose();
        return qMh;
    }

    /// <summary>One attention branch over <paramref name="qMh"/> from <see cref="ProjectQueryHeadMajor"/>: K and V of
    /// <paramref name="kvInput"/> (<c>[B, Skv, C]</c>) come from <see cref="AttentionCallContext.CachedKv"/> or are projected, normed,
    /// rotated and permuted head-major, then offered to it. <paramref name="qMh"/>, <paramref name="kvInput"/> and cached K/V stay the
    /// caller's; K/V the cache did not take are disposed. Returns the head-major <c>[B, heads, Sq, D]</c> output; the caller owns it.</summary>
    public Tensor AttendBranchHeadMajor(IBackend backend, Tensor qMh, Tensor kvInput, in AttentionCallContext context)
    {
        AttentionLayerWeights w = BranchWeights(in context);
        if (_config.KvHeads != _config.Heads)
            throw new NotSupportedException("Head-major operands carry one head per query head; grouped key/value heads attend through Attend.");
        int rank = kvInput.Shape.Rank;
        int batch = rank >= 3 ? (int)kvInput.Shape[0] : 1;
        int skv = (int)kvInput.Shape[rank - 2];
        if (qMh.Shape.Rank != 4 || qMh.Shape[0] != batch || qMh.Shape[1] != _config.Heads || qMh.Shape[3] != _config.HeadDim)
            throw new ArgumentException($"AttendBranchHeadMajor takes [{batch}, {_config.Heads}, S, {_config.HeadDim}] queries; got {qMh.Shape}.");
        Tensor keyMh = null!, valueMh = null!;
        bool borrowed = context.CachedKv?.TryGet(backend, kvInput, out keyMh, out valueMh) == true;
        bool stored = false;
        if (!borrowed)
        {
            DType kvAct = _config.ComputeDType ?? kvInput.DType;
            TensorShape projected = BranchShape(kvInput.Shape, _config.KvHeads);
            Tensor? k = null, v = null;
            try
            {
                k = new Tensor(projected, kvAct);
                backend.Linear(k, kvInput, w.Key!, w.KeyBias);
                v = new Tensor(projected, kvAct);
                backend.Linear(v, kvInput, w.Value!, w.ValueBias);
                if (_config.QkNorm != AttentionQkNorm.None)
                {
                    AttentionNormWeights n = _norms!.Value;
                    k = NormPerHead(backend, k, Have(n.Key), n.KeyEps);
                }
                RotateWithTables(backend, k, context.KeyRotation, _config.KvHeads);
                TensorShape headMajor = new(batch, _config.KvHeads, skv, _config.HeadDim);
                keyMh = new Tensor(headMajor, k.DType);
                backend.Permute0213(keyMh, k, skv, _config.KvHeads, _config.HeadDim);
                k.Dispose();
                k = null;
                valueMh = new Tensor(headMajor, v.DType);
                backend.Permute0213(valueMh, v, skv, _config.KvHeads, _config.HeadDim);
                v.Dispose();
                v = null;
            }
            catch
            {
                k?.Dispose();
                v?.Dispose();
                keyMh?.Dispose();
                valueMh?.Dispose();
                throw;
            }
            stored = context.CachedKv?.TryStore(backend, kvInput, keyMh, valueMh) == true;
        }
        return RunAttention(backend, qMh, keyMh, valueMh, _config.ComputeDType ?? qMh.DType, in context,
            borrowed || stored ? OperandOwnership.None : OperandOwnership.Key | OperandOwnership.Value);
    }

    /// <summary>The permute back to <paramref name="flatShape"/> (<c>[B, S, heads·D]</c>) of a branch's head-major output, which is
    /// disposed; the caller owns the result.</summary>
    public Tensor MergeHeads(IBackend backend, Tensor attnMh, TensorShape flatShape)
    {
        Tensor flat = new(flatShape, attnMh.DType);
        backend.Permute0213(flat, attnMh, _config.Heads, (int)attnMh.Shape[2], _config.HeadDim);
        attnMh.Dispose();
        return flat;
    }

    private AttentionLayerWeights BranchWeights(in AttentionCallContext context)
    {
        AttentionLayerWeights w = _weights ?? throw new InvalidOperationException("This attention layer has no projection weights.");
        if (_config.Projection != AttentionProjection.PerWeight || _config.Gate != AttentionGate.None || _config.QkNorm == AttentionQkNorm.FullWidth)
            throw new NotSupportedException("The branch stages run per-weight projections with no gate and no fused full-width norm.");
        if (context.Rotation is not null || context.HeadMajorRotation is not null
            || context.QueryRotation.Kind == QkRotationKind.RotateHalfPaired || context.KeyRotation.Kind == QkRotationKind.RotateHalfPaired)
            throw new NotSupportedException("A branch stage rotates through per-tensor tables, not a rotation adapter or a paired rotation.");
        if (context.ValueRowScale is not null || context.KvRewrite is not null || context.KvCapture is not null || context.Tap is not null
            || context.ReleaseInputs || context.ReleaseQueryAfterProjection)
            throw new ArgumentException("A branch stage runs no value row-scale, K/V rewrite, capture or tap and releases through its arguments.", nameof(context));
        return w;
    }

    /// <summary>Projections keep the model's flat <c>[.., heads·D]</c> shape unless a per-head norm needs <c>[.., heads, D]</c>.</summary>
    private TensorShape BranchShape(TensorShape input, int heads) =>
        _config.QkNorm == AttentionQkNorm.None ? WithLastDim(input, heads * _config.HeadDim) : ProjectedShape(input, heads);

    /// <summary>The projections; with <paramref name="release"/> the inputs are disposed as soon as nothing reads them.</summary>
    private (Tensor Q, Tensor K, Tensor V, Tensor? Gate) Project(IBackend backend, AttentionLayerWeights w, Tensor qInput,
        Tensor kvInput, DType act, DType kvAct, bool release, in AttentionCallContext context)
    {
        if (_config.Projection is AttentionProjection.FusedSlice or AttentionProjection.FusedSplitNorm)
            return ProjectFused(backend, w, qInput, kvInput, act, release);
        if (w.Query is null)
            throw new InvalidOperationException("This attention layer's weights have no query projection.");
        Tensor? gate = _config.Gate switch
        {
            AttentionGate.Sigmoid => new Tensor(WithLastDim(qInput.Shape, _config.InnerDim), act),
            AttentionGate.PerHeadDoubledSigmoid => new Tensor(WithLastDim(qInput.Shape, _config.Heads), act),
            _ => null,
        };
        Tensor q = new(ProjectedShape(qInput.Shape, _config.Heads), act);
        Tensor k = new(ProjectedShape(kvInput.Shape, _config.KvHeads), kvAct);
        Tensor v = new(ProjectedShape(kvInput.Shape, _config.KvHeads), kvAct);
        if (_config.Projection == AttentionProjection.GroupedByInput)
        {
            if (ReferenceEquals(qInput, kvInput))
            {
                backend.LinearMulti(qInput, gate is null
                    ? [new(q, w.Query!, w.QueryBias), new(k, w.Key!, w.KeyBias), new(v, w.Value!, w.ValueBias)]
                    : [new(gate, w.Gate!, w.GateBias), new(q, w.Query!, w.QueryBias), new(k, w.Key!, w.KeyBias), new(v, w.Value!, w.ValueBias)]);
            }
            else
            {
                backend.LinearMulti(qInput, gate is null ? [new LinearOp(q, w.Query!, w.QueryBias)]
                    : [new(gate, w.Gate!, w.GateBias), new(q, w.Query!, w.QueryBias)]);
                backend.LinearMulti(kvInput, [new LinearOp(k, w.Key!, w.KeyBias), new(v, w.Value!, w.ValueBias)]);
            }
            if (release) Release(qInput, kvInput);
            return (q, k, v, gate);
        }

        backend.Linear(q, qInput, w.Query!, w.QueryBias);
        backend.Linear(k, kvInput, w.Key!, w.KeyBias);
        backend.Linear(v, kvInput, w.Value!, w.ValueBias);
        if (context.ValueRowScale is { } rowScale)
        {
            Tensor scaled = new(v.Shape, v.DType);
            backend.Mul(scaled, v, rowScale);
            v.Dispose();
            v = scaled;
        }
        if (gate is not null)
            backend.Linear(gate, qInput, w.Gate!, w.GateBias);
        if (release) Release(qInput, kvInput);
        return (q, k, v, gate);
    }

    /// <summary>One Linear of the fused <c>[q|k|v]</c> weight, the input released right after it, then the split (with the
    /// per-head norm for <see cref="AttentionProjection.FusedSplitNorm"/>).</summary>
    private (Tensor Q, Tensor K, Tensor V, Tensor? Gate) ProjectFused(IBackend backend, AttentionLayerWeights w, Tensor input,
        Tensor kvInput, DType act, bool release)
    {
        if (!ReferenceEquals(input, kvInput))
            throw new NotSupportedException("A fused [q|k|v] projection is self-attention only.");
        TensorShape projected = ProjectedShape(input.Shape, _config.Heads);
        Tensor qkv = new(WithLastDim(input.Shape, 3 * _config.InnerDim), act);
        Tensor? q = null, k = null, v = null;
        try
        {
            backend.Linear(qkv, input, w.Qkv!, w.QkvBias);
            if (release) input.Dispose();
            q = new(projected, act);
            k = new(projected, act);
            v = new(projected, act);
            if (_config.Projection == AttentionProjection.FusedSplitNorm)
            {
                AttentionNormWeights n = _norms!.Value;
                backend.QkvSplitNorm(q, k, v, qkv, n.Query!, n.Key!, n.QueryEps);
            }
            else
            {
                backend.SliceLastDim(q, qkv, 0);
                backend.SliceLastDim(k, qkv, _config.InnerDim);
                backend.SliceLastDim(v, qkv, 2 * _config.InnerDim);
            }
            return (q, k, v, null);
        }
        catch
        {
            q?.Dispose();
            k?.Dispose();
            v?.Dispose();
            throw;
        }
        finally
        {
            qkv.Dispose();
        }
    }

    /// <summary>Q first, its input disposed, then K and V: a cross-attention's query input never coexists with K and V.</summary>
    private (Tensor Q, Tensor K, Tensor V, Tensor? Gate) ProjectReleasingQuery(IBackend backend, AttentionLayerWeights w, Tensor qInput,
        Tensor kvInput, DType act, DType kvAct)
    {
        if (w.Query is null)
            throw new InvalidOperationException("This attention layer's weights have no query projection.");
        Tensor q = new(ProjectedShape(qInput.Shape, _config.Heads), act);
        try
        {
            backend.Linear(q, qInput, w.Query!, w.QueryBias);
        }
        catch
        {
            q.Dispose();
            throw;
        }
        qInput.Dispose();
        Tensor? k = null, v = null;
        try
        {
            k = new(ProjectedShape(kvInput.Shape, _config.KvHeads), kvAct);
            backend.Linear(k, kvInput, w.Key!, w.KeyBias);
            v = new(ProjectedShape(kvInput.Shape, _config.KvHeads), kvAct);
            backend.Linear(v, kvInput, w.Value!, w.ValueBias);
            return (q, k, v, null);
        }
        catch
        {
            q.Dispose();
            k?.Dispose();
            v?.Dispose();
            throw;
        }
    }

    private static void Release(Tensor qInput, Tensor kvInput)
    {
        qInput.Dispose();
        if (!ReferenceEquals(kvInput, qInput)) kvInput.Dispose();
    }

    /// <summary>Refuses what the configured norm cannot apply, before anything is allocated.</summary>
    private void ValidateContext(in AttentionCallContext context, int batch)
    {
        if (context.CachedKv is not null)
            throw new ArgumentException("A cached key/value attends through AttendBranchHeadMajor.", nameof(context));
        bool fullWidth = _config.QkNorm == AttentionQkNorm.FullWidth;
        if (fullWidth && batch != 1)
            throw new NotSupportedException($"The full-width QK norm runs one batch item; got {batch}.");
        foreach (QkRotationKind kind in (ReadOnlySpan<QkRotationKind>)[context.QueryRotation.Kind, context.KeyRotation.Kind])
        {
            bool fullWidthKind = kind is QkRotationKind.FullWidthSplit or QkRotationKind.FullWidthInterleaved;
            if (kind != QkRotationKind.None && fullWidthKind != fullWidth)
                throw new NotSupportedException(fullWidth
                    ? $"The full-width QK norm runs only a full-width rotation, not {kind}."
                    : $"A {kind} rotation runs only with the full-width QK norm.");
        }
        if (fullWidth && (context.Rotation is not null || context.HeadMajorRotation is not null))
            throw new NotSupportedException("The full-width QK norm rotates through its tables; a rotation adapter cannot run with it.");
        if (fullWidth && (context.KvRewrite is not null || context.KvCapture is not null || context.Tap is not null))
            throw new NotSupportedException("The full-width QK norm attends its own K and V; a rewrite, capture or tap cannot run with it.");
        if ((context.QueryRotation.Kind == QkRotationKind.RotateHalfPaired || context.KeyRotation.Kind == QkRotationKind.RotateHalfPaired)
            && (context.QueryRotation.Kind != context.KeyRotation.Kind || context.QueryRotation.Cos is null || context.QueryRotation.Sin is null
                || !ReferenceEquals(context.QueryRotation.Cos, context.KeyRotation.Cos)
                || !ReferenceEquals(context.QueryRotation.Sin, context.KeyRotation.Sin)))
            throw new ArgumentException("A paired rotate-half rotation is set on both Q and K with the same tables.", nameof(context));
    }

    private static void ValidateOperandContext(in AttentionCallContext context)
    {
        if (context.QueryRotation.Kind != QkRotationKind.None || context.KeyRotation.Kind != QkRotationKind.None
            || context.Rotation is not null || context.HeadMajorRotation is not null)
            throw new ArgumentException("The operator stage runs no rotation.", nameof(context));
        if (context.ValueRowScale is not null || context.KvRewrite is not null || context.KvCapture is not null || context.Tap is not null)
            throw new ArgumentException("The operator stage runs no value row-scale, K/V rewrite, K/V capture or tap.", nameof(context));
        if (context.CachedKv is not null)
            throw new ArgumentException("The operator stage has no key/value projection to cache.", nameof(context));
        if (context.ReleaseInputs || context.ReleaseQueryAfterProjection)
            throw new ArgumentException("The operator stage has no projection inputs to release; name the operands in consume.", nameof(context));
    }

    /// <summary>Whether Q, K and V are declared flat <c>[.., S, heads·D]</c> rather than <c>[.., S, heads, D]</c>.</summary>
    private bool FlatOperands => _config.QkNorm is AttentionQkNorm.FullWidth or AttentionQkNorm.FullWidthRms;

    /// <summary>The norm, rotation and attention stages: a head-major output is permuted into <paramref name="flatShape"/>, a
    /// token-major one written in <paramref name="tokenMajorShape"/>, which holds the same bytes. Only <see cref="Forward"/> passes
    /// a different shape there (<see cref="AttentionShape"/>); <see cref="Attend"/> and <see cref="AttendFlat"/> pass the same.</summary>
    private Tensor Transform(IBackend backend, Tensor q, Tensor k, Tensor v, int batch, int sq, int skv, DType act,
        TensorShape flatShape, TensorShape tokenMajorShape, bool flat, in AttentionCallContext context)
    {
        if (context.Probe is { } probe)
        {
            probe.Observe("q", q);
            probe.Observe("k", k);
            probe.Observe("v", v);
        }
        if (_config.QkNorm == AttentionQkNorm.FullWidth)
            return AttendFullWidth(backend, q, k, v, sq, skv, act, flatShape, tokenMajorShape,
                TokenMajorServed(backend, batch, sq, skv, act, in context), in context);

        if (_config.QkNorm != AttentionQkNorm.None)
        {
            NormalizeQk(backend, ref q, ref k);
            if (context.Probe is { } normProbe)
            {
                normProbe.Observe("qNormed", q);
                normProbe.Observe("kNormed", k);
            }
        }
        context.Tap?.Observe(backend, q, k, v);
        RotateQk(backend, q, k, in context);
        if (context.KvRewrite is { } rewrite)
        {
            try
            {
                (k, v) = rewrite.Rewrite(backend, k, v);
            }
            catch
            {
                q.Dispose();
                throw;
            }
            skv = (int)k.Shape[k.Shape.Rank - (flat ? 2 : 3)];
        }
        bool ownsKv = context.KvCapture is null;
        context.KvCapture?.Capture(k, v);
        bool tokenMajor = TokenMajorServed(backend, batch, sq, skv, act, in context);
        try
        {
            return tokenMajor ? AttendTokenMajor(backend, q, k, v, act, tokenMajorShape, ownsKv, in context)
                : AttendHeadMajor(backend, q, k, v, batch, sq, skv, act, flatShape, ownsKv, in context);
        }
        catch when (!ownsKv)
        {
            k.Dispose();
            v.Dispose();
            throw;
        }
    }

    /// <summary>The rotation stage alone: the context's model rotation, paired table or per-tensor tables, in place on
    /// token-major Q and K.</summary>
    public void RotateQk(IBackend backend, Tensor q, Tensor k, in AttentionCallContext context)
    {
        if (context.Rotation is { } rotation)
        {
            rotation.RotateTokenMajor(backend, q, k, _config.Heads, _config.KvHeads);
        }
        else if (context.QueryRotation.Kind == QkRotationKind.RotateHalfPaired)
        {
            backend.ApplyRope(q, k, context.QueryRotation.Cos!, context.QueryRotation.Sin!);
        }
        else
        {
            RotateWithTables(backend, q, context.QueryRotation, _config.Heads);
            RotateWithTables(backend, k, context.KeyRotation, _config.KvHeads);
        }
    }

    /// <summary>The token-major entry runs for one batch item when the config names one, its switch is on, no host rotation
    /// needs the head-major layout, <see cref="AttentionLayerConfig.HeadMajorForSage"/> does not keep the call head-major, and the
    /// backend serves it.</summary>
    private bool TokenMajorServed(IBackend backend, int batch, int sq, int skv, DType act, in AttentionCallContext context) =>
        _config.TokenMajorRoute != AttentionRoute.Dense && batch == 1 && context.HeadMajorRotation is null
        && (_config.TokenMajorKnob?.Value ?? true)
        && !(_config.HeadMajorForSage && SageServesHeadMajor(sq, skv, act, context.Bias))
        && (_config.TokenMajorRoute == AttentionRoute.TokenMajorGqa
            ? backend.SupportsTokenMajorGqaAttention : backend.SupportsTokenMajorAttention);

    /// <summary>The CUDA head-major Sage row's admission (S.1, <c>CudaBackend.SageF16Preferred</c>), less the kernel check only the backend
    /// can make; a change to either side must change the other.</summary>
    private bool SageServesHeadMajor(int sq, int skv, DType act, Tensor? bias) =>
        act == DType.F16 && bias is null && _config.HeadDim is 64 or 128 && EngineKnobs.SageAttn.Value
        && Math.Min(sq, skv) >= EngineKnobs.SageF16MinSkv.Value;

    private static Tensor Have(Tensor? norm) =>
        norm ?? throw new InvalidOperationException("This attention layer's weights have no norm for that projection.");

    private static Tensor NormPerHead(IBackend backend, Tensor x, Tensor weight, float eps)
    {
        Tensor normed = new(x.Shape, x.DType);
        backend.RmsNorm(normed, x, weight, eps);
        x.Dispose();
        return normed;
    }

    private void RotateWithTables(IBackend backend, Tensor x, in QkRotation rotation, int heads)
    {
        switch (rotation.Kind)
        {
            case QkRotationKind.None:
                break;
            case QkRotationKind.InterleavedPairs:
                backend.WanRopeInterleaved(x, rotation.Cos!, rotation.Sin!, rotation.Tokens, heads, _config.HeadDim);
                break;
            case QkRotationKind.InterleavedPairsPerHead:
                backend.WanRopeInterleavedPerHead(x, rotation.Cos!, rotation.Sin!, rotation.Tokens, heads, _config.HeadDim);
                break;
            case QkRotationKind.RotateHalf:
                backend.ApplyRopeSingle(x, rotation.Cos!, rotation.Sin!);
                break;
            default:
                throw new NotSupportedException($"A {rotation.Kind} rotation runs only with the full-width QK norm.");
        }
    }

    private Tensor AttendTokenMajor(IBackend backend, Tensor q, Tensor k, Tensor v, DType act, TensorShape outShape, bool ownsKv,
        in AttentionCallContext context)
    {
        Tensor flat = new(outShape, act);
        try
        {
            AttentionCall call = AttentionCall.FromTokenMajor(flat, q, k, v, context.Bias, _config.Heads, _config.KvHeads,
                _config.HeadDim, _config.EffectiveScale, _config.AllowF16) with { Route = _config.TokenMajorRoute };
            backend.Attention(in call);
        }
        catch
        {
            flat.Dispose();
            throw;
        }
        finally
        {
            q.Dispose();
            if (ownsKv)
            {
                k.Dispose();
                v.Dispose();
            }
        }
        context.Probe?.Observe("attention", flat);
        return flat;
    }

    private Tensor AttendHeadMajor(IBackend backend, Tensor q, Tensor k, Tensor v, int batch, int sq, int skv, DType act,
        TensorShape flatShape, bool ownsKv, in AttentionCallContext context)
    {
        int heads = _config.Heads, kvHeads = _config.KvHeads, d = _config.HeadDim;
        Tensor qMh = new(new TensorShape(batch, heads, sq, d), q.DType);
        backend.Permute0213(qMh, q, sq, heads, d);
        q.Dispose();
        Tensor kMh = new(new TensorShape(batch, kvHeads, skv, d), k.DType);
        backend.Permute0213(kMh, k, skv, kvHeads, d);
        if (ownsKv) k.Dispose();
        Tensor vMh = new(new TensorShape(batch, kvHeads, skv, d), v.DType);
        backend.Permute0213(vMh, v, skv, kvHeads, d);
        if (ownsKv) v.Dispose();
        context.HeadMajorRotation?.RotateHeadMajor(backend, qMh, kMh, batch, heads, kvHeads, sq);
        if (kvHeads != heads)
        {
            int group = heads / kvHeads;
            Tensor kRep = new(new TensorShape(batch, heads, skv, d), kMh.DType);
            backend.RepeatKvHeads(kRep, kMh, kvHeads, group);
            kMh.Dispose();
            kMh = kRep;
            Tensor vRep = new(new TensorShape(batch, heads, skv, d), vMh.DType);
            backend.RepeatKvHeads(vRep, vMh, kvHeads, group);
            vMh.Dispose();
            vMh = vRep;
        }
        return RunHeadMajor(backend, qMh, kMh, vMh, sq, act, flatShape, in context);
    }

    /// <summary>The head-major call over permuted, head-repeated operands, then the permute back to <paramref name="flatShape"/>.</summary>
    private Tensor RunHeadMajor(IBackend backend, Tensor qMh, Tensor kMh, Tensor vMh, int sq, DType act, TensorShape flatShape,
        in AttentionCallContext context)
    {
        Tensor attn = RunAttention(backend, qMh, kMh, vMh, act, in context, OperandOwnership.All);
        Tensor flat = new(flatShape, act);
        backend.Permute0213(flat, attn, _config.Heads, sq, _config.HeadDim);
        attn.Dispose();
        return flat;
    }

    private Tensor RunAttention(IBackend backend, Tensor qMh, Tensor kMh, Tensor vMh, DType act, in AttentionCallContext context,
        OperandOwnership consume, Tensor? output = null)
    {
        bool owned = output is null;
        Tensor attn = output ?? new(new TensorShape(qMh.Shape[0], _config.Heads, qMh.Shape[2], _config.HeadDim), act);
        try
        {
            AttentionCall call = AttentionCall.FromScaledDotProductAttention(attn, qMh, kMh, vMh, context.Bias, _config.EffectiveScale,
                _config.AllowF16);
            backend.Attention(in call);
        }
        catch
        {
            if (owned) attn.Dispose();
            throw;
        }
        finally
        {
            Consume(consume, qMh, kMh, vMh);
        }
        context.Probe?.Observe("attention", attn);
        return attn;
    }

    private static void Consume(OperandOwnership consume, Tensor qMh, Tensor kMh, Tensor vMh)
    {
        if (consume.HasFlag(OperandOwnership.Query)) qMh.Dispose();
        if (consume.HasFlag(OperandOwnership.Key)) kMh.Dispose();
        if (consume.HasFlag(OperandOwnership.Value)) vMh.Dispose();
    }

    private void ValidateOutput(Tensor? output, Tensor qMh, Tensor keys, Tensor values, DType act)
    {
        if (output is null) return;
        if (ReferenceEquals(output, qMh) || ReferenceEquals(output, keys) || ReferenceEquals(output, values))
            throw new ArgumentException("The output cannot be one of the attention's operands.", nameof(output));
        TensorShape expected = new(qMh.Shape[0], _config.Heads, qMh.Shape[2], _config.HeadDim);
        if (!output.Shape.Equals(expected) || output.DType != act)
            throw new ArgumentException($"The output is {output.Shape} {output.DType} but the attention writes {expected} {act}.", nameof(output));
    }

    /// <summary>The full-width norm and rotation as one op that emits the attention layout directly (LTX-2), with K and V
    /// widened onto the query's dtype first.</summary>
    private Tensor AttendFullWidth(IBackend backend, Tensor q, Tensor k, Tensor v, int sq, int skv, DType act, TensorShape flatShape,
        TensorShape tokenMajorShape, bool tokenMajor, in AttentionCallContext context)
    {
        AttentionNormWeights norms = _norms!.Value;
        Tensor qNorm = Have(norms.Query), kNorm = Have(norms.Key);
        if (tokenMajor)
        {
            Tensor qTm = NormRotateTokenMajor(backend, q, qNorm, context.QueryRotation, sq, act);
            Tensor kTm = NormRotateTokenMajor(backend, k, kNorm, context.KeyRotation, skv, act);
            Tensor vTm = v.DType == act ? v : Widen(backend, v, act);
            if (!ReferenceEquals(vTm, v)) v.Dispose();
            return AttendTokenMajor(backend, qTm, kTm, vTm, act, tokenMajorShape, ownsKv: true, in context);
        }
        Tensor qMh = NormRotateHeadMajor(backend, q, qNorm, context.QueryRotation, sq, act);
        Tensor kMh = NormRotateHeadMajor(backend, k, kNorm, context.KeyRotation, skv, act);
        Tensor vMh = ToHeadMajor(backend, v, skv, act);
        v.Dispose();
        return RunHeadMajor(backend, qMh, kMh, vMh, sq, act, flatShape, in context);
    }

    private Tensor NormRotateHeadMajor(IBackend backend, Tensor x, Tensor normWeight, in QkRotation rotation, int s, DType act)
    {
        Tensor src = x.DType == act ? x : Widen(backend, x, act);
        if (!ReferenceEquals(src, x)) x.Dispose();
        if (rotation.Kind == QkRotationKind.FullWidthInterleaved)
        {
            Tensor normed = new(src.Shape, src.DType);
            backend.RmsNorm(normed, src, normWeight, _config.FullWidthNormEps);
            src.Dispose();
            RotateFullWidth(backend, normed, rotation);
            Tensor headMajor = ToHeadMajor(backend, normed, s, act);
            normed.Dispose();
            return headMajor;
        }
        Tensor o = new(new TensorShape(1, _config.Heads, s, _config.HeadDim), act);
        backend.Ltx2QkNormRopeHeadMajor(o, src, normWeight, TableCos(rotation), TableSin(rotation), s, _config.Heads, _config.HeadDim,
            _config.FullWidthNormEps);
        src.Dispose();
        return o;
    }

    private Tensor NormRotateTokenMajor(IBackend backend, Tensor x, Tensor normWeight, in QkRotation rotation, int s, DType act)
    {
        Tensor src = x.DType == act ? x : Widen(backend, x, act);
        if (!ReferenceEquals(src, x)) x.Dispose();
        Tensor o = new(new TensorShape(s, _config.InnerDim), act);
        if (rotation.Kind == QkRotationKind.FullWidthInterleaved)
        {
            backend.RmsNorm(o, src, normWeight, _config.FullWidthNormEps);
            src.Dispose();
            RotateFullWidth(backend, o, rotation);
            return o;
        }
        backend.Ltx2QkNormRopeTokenMajor(o, src, normWeight, TableCos(rotation), TableSin(rotation), s, _config.Heads, _config.HeadDim,
            _config.FullWidthNormEps);
        src.Dispose();
        return o;
    }

    /// <summary>Interleaved pairs over the full inner width of <c>[S, inner]</c> <paramref name="x"/>.</summary>
    private void RotateFullWidth(IBackend backend, Tensor x, in QkRotation rotation)
    {
        backend.WanRopeInterleaved(x, rotation.Cos!, rotation.Sin!, (int)x.Shape[0], 1, _config.InnerDim);
    }

    private static Tensor? TableCos(in QkRotation rotation) => rotation.Kind == QkRotationKind.None ? null : rotation.Cos;

    private static Tensor? TableSin(in QkRotation rotation) => rotation.Kind == QkRotationKind.None ? null : rotation.Sin;

    /// <summary><c>[S, heads·D]</c> → <c>[1, heads, S, D]</c>, widening first: the permute cannot convert.</summary>
    private Tensor ToHeadMajor(IBackend backend, Tensor x, int s, DType act)
    {
        Tensor src = x.DType == act ? x : Widen(backend, x, act);
        Tensor o = new(new TensorShape(1, _config.Heads, s, _config.HeadDim), act);
        backend.Permute0213(o, src, s, _config.Heads, _config.HeadDim);
        if (!ReferenceEquals(src, x)) src.Dispose();
        return o;
    }

    /// <summary>Widens an F32 key/value stream to an F16 query's dtype, the only mismatch the configs allow.</summary>
    private Tensor Widen(IBackend backend, Tensor x, DType target)
    {
        if (!_config.WidenKvToQuery || target != DType.F16 || x.DType != DType.F32)
            throw new NotSupportedException($"Attention only widens an F32 K/V stream to F16, got {x.DType} to {target}.");
        Tensor o = new(x.Shape, target);
        backend.CastToF16(o, x);
        return o;
    }

    /// <summary>The output stage; <paramref name="logical"/> is the attention output's <c>[.., heads·D]</c> shape, which
    /// <paramref name="flat"/> may hold as <c>[1, S, heads, D]</c>.</summary>
    private Tensor Output(IBackend backend, AttentionLayerWeights w, Tensor flat, Tensor? gate, DType act, TensorShape logical)
    {
        TensorShape outShape = WithLastDim(logical, _config.OutDim > 0 ? _config.OutDim : _config.InnerDim);
        switch (_config.Gate)
        {
            case AttentionGate.PerHeadDoubledSigmoid:
            {
                Tensor gatedOut = new(outShape, act);
                backend.LinearHeadGated(gatedOut, flat, w.Output!, w.OutputBias, gate!, _config.Heads, _config.HeadDim);
                gate!.Dispose();
                flat.Dispose();
                return gatedOut;
            }
            case AttentionGate.Sigmoid:
            {
                Tensor sig = new(flat.Shape, act);
                backend.Sigmoid(sig, gate!);
                gate!.Dispose();
                Tensor gated = new(flat.Shape, act);
                backend.Mul(gated, flat, sig);
                flat.Dispose();
                sig.Dispose();
                Tensor gatedOut = new(outShape, act);
                backend.Linear(gatedOut, gated, w.Output!, w.OutputBias);
                gated.Dispose();
                return gatedOut;
            }
            default:
            {
                Tensor projected = new(outShape, act);
                backend.Linear(projected, flat, w.Output!, w.OutputBias);
                flat.Dispose();
                return projected;
            }
        }
    }

    /// <summary>The shape <see cref="Forward"/> holds an equal-head token-major output in: <c>[1, S, heads·D]</c> becomes the
    /// <c>[S, heads·D]</c> rows its output projection has always read (backends pick Linear fast paths by the input's shape).</summary>
    private TensorShape AttentionShape(TensorShape flatShape) =>
        _config.TokenMajorRoute == AttentionRoute.TokenMajor && flatShape.Rank == 3 && flatShape[0] == 1
            ? new TensorShape(flatShape[1], flatShape[2])
            : flatShape;

    /// <summary>The projection's shape: <c>[.., heads, D]</c> for a per-head norm, else <c>[.., heads·D]</c>.</summary>
    private TensorShape ProjectedShape(TensorShape input, int heads)
    {
        if (_config.QkNorm is AttentionQkNorm.FullWidth or AttentionQkNorm.FullWidthRms)
            return WithLastDim(input, heads * _config.HeadDim);
        long[] dims = new long[input.Rank + 1];
        for (int i = 0; i < input.Rank - 1; i++) dims[i] = input[i];
        dims[input.Rank - 1] = heads;
        dims[input.Rank] = _config.HeadDim;
        return new TensorShape(dims);
    }

    private static TensorShape WithLastDim(TensorShape shape, int last)
    {
        long[] dims = new long[shape.Rank];
        for (int i = 0; i < shape.Rank - 1; i++) dims[i] = shape[i];
        dims[shape.Rank - 1] = last;
        return new TensorShape(dims);
    }
}

/// <summary>The operands of <see cref="AttentionLayer.AttendHeadMajorOperands"/> the layer disposes after the call.</summary>
[Flags]
public enum OperandOwnership
{
    /// <summary>The caller disposes all three.</summary>
    None = 0,

    /// <summary>The query.</summary>
    Query = 1,

    /// <summary>The key.</summary>
    Key = 2,

    /// <summary>The value.</summary>
    Value = 4,

    /// <summary>All three.</summary>
    All = Query | Key | Value,
}

/// <summary>The valid rows and positions of a head-major key/value cache for <see cref="AttentionLayer.AttendCachedHeadMajor"/>:
/// <paramref name="KvLength"/> rows are attended, query row 0 sits at <paramref name="QueryOffset"/>, and a nonzero
/// <paramref name="DevicePositions"/> buffer supplies both on the device instead (the host values are the eager fallback's).</summary>
public readonly record struct AttentionCacheView(int KvLength, int QueryOffset, bool Causal,
    int SlidingWindow = 0, float Softcap = 0f, Tensor? Sinks = null, ulong DevicePositions = 0);
