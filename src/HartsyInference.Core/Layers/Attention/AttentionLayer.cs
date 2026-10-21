using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>One attention layer composed of stages: projection, Q/K norm and rotation, the attention call
/// (docs/ATTENTION_CONTRACT.md), output gate and projection; each intermediate is disposed after its last use.</summary>
public sealed class AttentionLayer
{
    private readonly AttentionLayerConfig _config;
    private readonly AttentionLayerWeights? _weights;
    private readonly AttentionNormWeights? _norms;

    /// <summary>Creates a layer. Without <paramref name="weights"/> only <see cref="Attend"/> and <see cref="AttendFlat"/> can
    /// run, and only without a QK norm.</summary>
    public AttentionLayer(AttentionLayerConfig config, AttentionLayerWeights? weights = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        if (weights is not null)
        {
            if (config.Gate != AttentionGate.None && weights.Gate is null)
                throw new ArgumentException($"A {config.Gate} gate needs a gate projection.", nameof(weights));
            if (config.QkNorm != AttentionQkNorm.None && (weights.QueryNorm is null || weights.KeyNorm is null))
                throw new ArgumentException($"A {config.QkNorm} QK norm needs both norm weights.", nameof(weights));
            bool fused = config.Projection is AttentionProjection.FusedSlice or AttentionProjection.FusedSplitNorm;
            if (fused ? weights.Qkv is null : weights.Query is null || weights.Key is null || weights.Value is null)
                throw new ArgumentException($"A {config.Projection} projection needs its weights.", nameof(weights));
            if (config.Projection == AttentionProjection.FusedSplitNorm && weights.QueryNormEps != weights.KeyNormEps)
                throw new ArgumentException("The fused split-norm projection takes one epsilon for Q and K.", nameof(weights));
            if (config.QkNorm != AttentionQkNorm.None)
                _norms = new AttentionNormWeights(weights.QueryNorm!, weights.KeyNorm!, weights.QueryNormEps, weights.KeyNormEps);
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

        TensorShape flatShape = WithLastDim(qInput.Shape, _config.InnerDim);
        (Tensor q, Tensor k, Tensor v, Tensor? gate) = Project(backend, w, qInput, kvInput, act, kvAct, context.ReleaseInputs, in context);
        Tensor flat = Transform(backend, q, k, v, batch, sq, skv, act, flatShape, in context);
        return Output(backend, w, flat, gate, act);
    }

    /// <summary>The transform and attention stages for a model that projects and outputs through its own code: token-major
    /// <paramref name="q"/> <c>[B, S, heads, D]</c>, <paramref name="k"/> and <paramref name="v"/> <c>[B, Skv, kvHeads, D]</c>,
    /// all three taken over and disposed. Returns the attention output in <paramref name="flatShape"/>; the caller owns it.</summary>
    public Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, TensorShape flatShape, in AttentionCallContext context)
    {
        if (_config.QkNorm != AttentionQkNorm.None && _norms is null)
            throw new InvalidOperationException("A QK norm needs the layer's norm weights.");
        if (q.Shape.Rank != 4 || k.Shape.Rank != 4 || v.Shape.Rank != 4)
            throw new ArgumentException($"Attend takes [B, S, heads, D] operands; got {q.Shape}, {k.Shape}, {v.Shape}.");
        ValidateContext(in context, (int)q.Shape[0]);
        return Transform(backend, q, k, v, (int)q.Shape[0], (int)q.Shape[1], (int)k.Shape[1], q.DType, flatShape, in context);
    }

    /// <summary><see cref="Attend"/> over flat operands: <paramref name="q"/> <c>[B, S, heads·D]</c>, <paramref name="k"/> and
    /// <paramref name="v"/> <c>[B, Skv, kvHeads·D]</c> (a joint sequence, or a full-width norm's layout), head-major only.</summary>
    public Tensor AttendFlat(IBackend backend, Tensor q, Tensor k, Tensor v, TensorShape flatShape, in AttentionCallContext context)
    {
        if (_config.QkNorm is not (AttentionQkNorm.None or AttentionQkNorm.FullWidthRms))
            throw new NotSupportedException($"Flat operands cannot take a {_config.QkNorm} QK norm.");
        if (_config.QkNorm != AttentionQkNorm.None && _norms is null)
            throw new InvalidOperationException("A QK norm needs the layer's norm weights.");
        if (_config.TokenMajorRoute != AttentionRoute.Dense)
            throw new NotSupportedException("Flat operands attend head-major.");
        if (context.QueryRotation.Kind is not (QkRotationKind.None or QkRotationKind.InterleavedPairs)
            || context.KeyRotation.Kind is not (QkRotationKind.None or QkRotationKind.InterleavedPairs))
            throw new NotSupportedException("Flat operands take interleaved-pair tables or a rotation adapter; the other table kinds read [.., heads, D].");
        if (q.Shape.Rank != 3 || k.Shape.Rank != 3 || v.Shape.Rank != 3 || q.Shape[2] != _config.InnerDim
            || k.Shape[2] != (long)_config.KvHeads * _config.HeadDim || v.Shape[2] != k.Shape[2])
            throw new ArgumentException($"AttendFlat takes [B, S, heads·D] operands of this layer's widths; got {q.Shape}, {k.Shape}, {v.Shape}.");
        ValidateContext(in context, (int)q.Shape[0]);
        return Transform(backend, q, k, v, (int)q.Shape[0], (int)q.Shape[1], (int)k.Shape[1], q.DType, flatShape, in context);
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
        q = NormPerHead(backend, q, n.Query, n.QueryEps);
        k = NormPerHead(backend, k, n.Key, n.KeyEps);
    }

    /// <summary>The output stage alone: the output projection of <paramref name="flat"/>, which is disposed.</summary>
    public Tensor ProjectOutput(IBackend backend, Tensor flat)
    {
        AttentionLayerWeights w = _weights ?? throw new InvalidOperationException("This attention layer has no projection weights.");
        if (w.Output is null)
            throw new InvalidOperationException("This attention layer has no output projection.");
        if (_config.Gate != AttentionGate.None)
            throw new NotSupportedException("A gated layer projects its output through Forward.");
        return Output(backend, w, flat, null, flat.DType);
    }

    /// <summary>The projections; with <paramref name="release"/> the inputs are disposed as soon as nothing reads them.</summary>
    private (Tensor Q, Tensor K, Tensor V, Tensor? Gate) Project(IBackend backend, AttentionLayerWeights w, Tensor qInput,
        Tensor kvInput, DType act, DType kvAct, bool release, in AttentionCallContext context)
    {
        if (_config.Projection is AttentionProjection.FusedSlice or AttentionProjection.FusedSplitNorm)
            return ProjectFused(backend, w, qInput, kvInput, act, release);
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
        backend.Linear(qkv, input, w.Qkv!, w.QkvBias);
        if (release) input.Dispose();
        Tensor q = new(projected, act), k = new(projected, act), v = new(projected, act);
        if (_config.Projection == AttentionProjection.FusedSplitNorm)
        {
            AttentionNormWeights n = _norms!.Value;
            backend.QkvSplitNorm(q, k, v, qkv, n.Query, n.Key, n.QueryEps);
        }
        else
        {
            backend.SliceLastDim(q, qkv, 0);
            backend.SliceLastDim(k, qkv, _config.InnerDim);
            backend.SliceLastDim(v, qkv, 2 * _config.InnerDim);
        }
        qkv.Dispose();
        return (q, k, v, null);
    }

    private static void Release(Tensor qInput, Tensor kvInput)
    {
        qInput.Dispose();
        if (!ReferenceEquals(kvInput, qInput)) kvInput.Dispose();
    }

    /// <summary>Refuses what the configured norm cannot apply, before anything is allocated.</summary>
    private void ValidateContext(in AttentionCallContext context, int batch)
    {
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
        if (fullWidth && (context.KvRewrite is not null || context.KvCapture is not null))
            throw new NotSupportedException("The full-width QK norm attends its own K and V; a rewrite or capture cannot run with it.");
        if ((context.QueryRotation.Kind == QkRotationKind.RotateHalfPaired || context.KeyRotation.Kind == QkRotationKind.RotateHalfPaired)
            && (context.QueryRotation.Kind != context.KeyRotation.Kind || context.QueryRotation.Cos is null || context.QueryRotation.Sin is null
                || !ReferenceEquals(context.QueryRotation.Cos, context.KeyRotation.Cos)
                || !ReferenceEquals(context.QueryRotation.Sin, context.KeyRotation.Sin)))
            throw new ArgumentException("A paired rotate-half rotation is set on both Q and K with the same tables.", nameof(context));
    }

    private Tensor Transform(IBackend backend, Tensor q, Tensor k, Tensor v, int batch, int sq, int skv, DType act,
        TensorShape flatShape, in AttentionCallContext context)
    {
        if (context.Probe is { } probe)
        {
            probe.Observe("q", q);
            probe.Observe("k", k);
            probe.Observe("v", v);
        }
        bool tokenMajor = TokenMajorServed(backend, batch, in context);
        if (_config.QkNorm == AttentionQkNorm.FullWidth)
            return AttendFullWidth(backend, q, k, v, sq, skv, act, flatShape, tokenMajor, in context);

        if (_config.QkNorm != AttentionQkNorm.None)
        {
            NormalizeQk(backend, ref q, ref k);
            if (context.Probe is { } normProbe)
            {
                normProbe.Observe("qNormed", q);
                normProbe.Observe("kNormed", k);
            }
        }
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
                k.Dispose();
                v.Dispose();
                throw;
            }
            skv = (int)k.Shape[1];
        }
        bool ownsKv = context.KvCapture is null;
        context.KvCapture?.Capture(k, v);
        return tokenMajor ? AttendTokenMajor(backend, q, k, v, act, flatShape, ownsKv, in context)
            : AttendHeadMajor(backend, q, k, v, batch, sq, skv, act, flatShape, ownsKv, in context);
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
    /// needs the head-major layout, and the backend serves it.</summary>
    private bool TokenMajorServed(IBackend backend, int batch, in AttentionCallContext context) =>
        _config.TokenMajorRoute != AttentionRoute.Dense && batch == 1 && context.HeadMajorRotation is null
        && (_config.TokenMajorKnob?.Value ?? true)
        && (_config.TokenMajorRoute == AttentionRoute.TokenMajorGqa
            ? backend.SupportsTokenMajorGqaAttention : backend.SupportsTokenMajorAttention);

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
            case QkRotationKind.RotateHalf:
                backend.ApplyRopeSingle(x, rotation.Cos!, rotation.Sin!);
                break;
            default:
                throw new NotSupportedException($"A {rotation.Kind} rotation runs only with the full-width QK norm.");
        }
    }

    private Tensor AttendTokenMajor(IBackend backend, Tensor q, Tensor k, Tensor v, DType act, TensorShape flatShape, bool ownsKv,
        in AttentionCallContext context)
    {
        Tensor flat = new(flatShape, act);
        AttentionCall call = AttentionCall.FromTokenMajor(flat, q, k, v, context.Bias, _config.Heads, _config.KvHeads,
            _config.HeadDim, _config.EffectiveScale, _config.AllowF16) with { Route = _config.TokenMajorRoute };
        backend.Attention(in call);
        q.Dispose();
        if (ownsKv)
        {
            k.Dispose();
            v.Dispose();
        }
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
        return RunHeadMajor(backend, qMh, kMh, vMh, batch, sq, act, flatShape, in context);
    }

    /// <summary>The head-major call over permuted, head-repeated operands, then the permute back to <paramref name="flatShape"/>.</summary>
    private Tensor RunHeadMajor(IBackend backend, Tensor qMh, Tensor kMh, Tensor vMh, int batch, int sq, DType act, TensorShape flatShape,
        in AttentionCallContext context)
    {
        Tensor attn = new(new TensorShape(batch, _config.Heads, sq, _config.HeadDim), act);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(attn, qMh, kMh, vMh, context.Bias, _config.EffectiveScale,
            _config.AllowF16);
        backend.Attention(in call);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();
        context.Probe?.Observe("attention", attn);
        Tensor flat = new(flatShape, act);
        backend.Permute0213(flat, attn, _config.Heads, sq, _config.HeadDim);
        attn.Dispose();
        return flat;
    }

    /// <summary>The full-width norm and rotation as one op that emits the attention layout directly (LTX-2), with K and V
    /// widened onto the query's dtype first.</summary>
    private Tensor AttendFullWidth(IBackend backend, Tensor q, Tensor k, Tensor v, int sq, int skv, DType act, TensorShape flatShape,
        bool tokenMajor, in AttentionCallContext context)
    {
        AttentionNormWeights norms = _norms!.Value;
        Tensor qNorm = norms.Query, kNorm = norms.Key;
        if (tokenMajor)
        {
            Tensor qTm = NormRotateTokenMajor(backend, q, qNorm, context.QueryRotation, sq, act);
            Tensor kTm = NormRotateTokenMajor(backend, k, kNorm, context.KeyRotation, skv, act);
            Tensor vTm = v.DType == act ? v : Widen(backend, v, act);
            if (!ReferenceEquals(vTm, v)) v.Dispose();
            return AttendTokenMajor(backend, qTm, kTm, vTm, act, flatShape, ownsKv: true, in context);
        }
        Tensor qMh = NormRotateHeadMajor(backend, q, qNorm, context.QueryRotation, sq, act);
        Tensor kMh = NormRotateHeadMajor(backend, k, kNorm, context.KeyRotation, skv, act);
        Tensor vMh = ToHeadMajor(backend, v, skv, act);
        v.Dispose();
        return RunHeadMajor(backend, qMh, kMh, vMh, 1, sq, act, flatShape, in context);
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

    private Tensor Output(IBackend backend, AttentionLayerWeights w, Tensor flat, Tensor? gate, DType act)
    {
        TensorShape outShape = WithLastDim(flat.Shape, _config.OutDim > 0 ? _config.OutDim : _config.InnerDim);
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
