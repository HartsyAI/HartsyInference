using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Looped;

/// <summary>One physical layer of a looped normalized-GPT decoder: a gated attention mixer or a gated MLP, joined to the residual by the
/// hypersphere interpolation. Holds the checkpoint's projection weights in their loaded dtype and the learned scale vectors baked to the
/// values the forward multiplies with (nGPT stores them as <c>value · init / scaling</c>; see <see cref="LoopedTransformerConfig"/>).</summary>
/// <remarks>Dory's attention is <c>q = (sqk)² ⊙ norm(rope(Wq x))</c>, <c>k = norm(rope(Wk x))</c>, <c>v = Wv x</c>, output
/// <c>Wo · norm(attn ⊙ sigmoid((s_gate)² ⊙ Wg x))</c>; its MLP is <c>Wdown · norm(silu(suv_g ⊙ Wgate x) ⊙ (suv_u ⊙ Wup x))</c>.
/// Every norm is the unit-norm over the vector it names (one head, the merged heads, the MLP width).</remarks>
internal sealed class LoopedLayer : IDisposable
{
    private readonly LoopedTransformerConfig _cfg;
    private readonly LoopedNormVectors _norms;
    private readonly int _physical;
    // Tensors built here (baked scales, fused projections): owned and disposed by this layer. Checkpoint weights are the loader's.
    private readonly List<Tensor> _owned = [];
    private Tensor? _qW, _kW, _vW, _oW, _gateW, _qkvW;
    private Tensor? _qkScale, _gateScale;
    private Tensor? _ffnGateW, _ffnUpW, _ffnDownW, _gateUpW;
    private Tensor? _suvScale, _ffnGateScale, _ffnUpScale;
    private Tensor? _alpha, _oneMinusAlpha;
    private int _disposed;

    /// <summary>Builds the layer for physical index <paramref name="physical"/> of <paramref name="cfg"/>, normalizing with <paramref name="norms"/>.</summary>
    public LoopedLayer(LoopedTransformerConfig cfg, LoopedNormVectors norms, int physical)
    {
        _cfg = cfg;
        _norms = norms;
        _physical = physical;
    }

    /// <summary>What this layer computes.</summary>
    public LoopedLayerKind Kind => _cfg.LayerKinds[_physical];

    /// <summary>Whether this layer attends (and so reads and writes a cache slot).</summary>
    public bool IsAttention => Kind != LoopedLayerKind.Mlp;

    private int Window => Kind == LoopedLayerKind.SlidingAttention ? _cfg.SlidingWindow : 0;

    private float AttentionScale => 1f / MathF.Sqrt(_cfg.HeadDim);

    /// <summary>Loads this layer's tensors under <paramref name="prefix"/> (<c>model.layers.N</c>) and bakes the scale vectors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        int h = _cfg.HiddenSize;
        _alpha = Bake(w, $"{prefix}.residual_alpha.weight", h, _cfg.AlphaInit / _cfg.InitStd, square: false, abs: true);
        _oneMinusAlpha = OneMinus(_alpha);
        if (IsAttention)
        {
            int nq = _cfg.QDim, nk = _cfg.KvDim;
            _qW = Require(w, $"{prefix}.self_attn.q_proj.weight", nq, h);
            _kW = Require(w, $"{prefix}.self_attn.k_proj.weight", nk, h);
            _vW = Require(w, $"{prefix}.self_attn.v_proj.weight", nk, h);
            _oW = Require(w, $"{prefix}.self_attn.o_proj.weight", h, nq);
            _gateW = Require(w, $"{prefix}.self_attn.gate_proj.weight", nq, h);
            if (_qW.DType == _kW.DType && _qW.DType == _vW.DType && ProjectionOps.Fusable(_qW, _kW, _vW))
            {
                _qkvW = ProjectionOps.ConcatRows(_qW, _kW, _vW);
                _owned.Add(_qkvW);
            }
            _qkScale = Bake(w, $"{prefix}.self_attn.qk_scale.weight", nq, MathF.Sqrt(_cfg.HeadDim) / _cfg.InitStd, square: true, abs: false);
            _gateScale = Bake(w, $"{prefix}.self_attn.gate_scale.weight", nq, MathF.Sqrt(_cfg.EffectiveGateInit) / _cfg.InitStd, square: true, abs: false);
            return;
        }
        int ff = _cfg.IntermediateSize;
        _ffnGateW = Require(w, $"{prefix}.mlp.gate_proj.weight", ff, h);
        _ffnUpW = Require(w, $"{prefix}.mlp.up_proj.weight", ff, h);
        _ffnDownW = Require(w, $"{prefix}.mlp.down_proj.weight", h, ff);
        _ffnGateScale = Bake(w, $"{prefix}.mlp.gate_scale.weight", ff, MathF.Sqrt(h), square: false, abs: false);
        _ffnUpScale = Bake(w, $"{prefix}.mlp.up_scale.weight", ff, MathF.Sqrt(h), square: false, abs: false);
        if (_ffnGateW.DType == _ffnUpW.DType && ProjectionOps.Fusable(_ffnGateW, _ffnUpW))
        {
            _gateUpW = ProjectionOps.ConcatRows(_ffnGateW, _ffnUpW);
            _owned.Add(_gateUpW);
            _suvScale = ProjectionOps.ConcatRows(_ffnGateScale, _ffnUpScale);
            _owned.Add(_suvScale);
        }
    }

    /// <summary>Every tensor a forward reads, for residency: fused projections stand in for the parts they replaced.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _alpha!;
        yield return _oneMinusAlpha!;
        if (IsAttention)
        {
            if (_qkvW is not null) yield return _qkvW;
            else { yield return _qW!; yield return _kW!; yield return _vW!; }
            yield return _oW!;
            yield return _gateW!;
            yield return _qkScale!;
            yield return _gateScale!;
            yield break;
        }
        if (_gateUpW is not null) { yield return _gateUpW; yield return _suvScale!; }
        else { yield return _ffnGateW!; yield return _ffnUpW!; yield return _ffnGateScale!; yield return _ffnUpScale!; }
        yield return _ffnDownW!;
    }

    /// <summary>Runs the layer over <paramref name="hidden"/> <c>[1, T, hidden]</c> at positions <paramref name="posStart"/>.., appending this
    /// layer's keys/values to cache slot <paramref name="slot"/>; returns the new residual (owned by the caller), consuming nothing.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, int t, int posStart, IKvCache cache, int slot, Tensor? cos, Tensor? sin)
    {
        Tensor sublayer = IsAttention
            ? Attention(backend, hidden, t, posStart, cache, slot, cos!, sin!)
            : Mlp(backend, hidden, t);
        Tensor joined = HypersphereOps.Interpolate(backend, hidden, sublayer, _alpha!, _oneMinusAlpha!, _norms.Hidden, _cfg.NormEps);
        sublayer.Dispose();
        return joined;
    }

    /// <summary>Single-token step for graph capture: the same math as <see cref="Forward"/> at <c>T = 1</c>, with RoPE, the cache write and
    /// attention reading their position from the device buffer <paramref name="devicePos"/>.</summary>
    public Tensor ForwardGraphStep(IBackend backend, Tensor hidden, IKvCache cache, int slot, Tensor? cosTable, Tensor? sinTable, ulong devicePos)
    {
        Tensor sublayer = IsAttention
            ? AttentionGraphStep(backend, hidden, cache, slot, cosTable!, sinTable!, devicePos)
            : Mlp(backend, hidden, 1);
        Tensor joined = HypersphereOps.Interpolate(backend, hidden, sublayer, _alpha!, _oneMinusAlpha!, _norms.Hidden, _cfg.NormEps);
        sublayer.Dispose();
        return joined;
    }

    /// <summary>Batched decode: one token per sequence in <paramref name="hidden"/> <c>[1, B, hidden]</c>, each attending its own cache; the projections run
    /// once over the batch.</summary>
    public Tensor ForwardBatchDecode(IBackend backend, Tensor hidden, int b, Tensor? cos, Tensor? sin, IKvCache[] caches, int slot)
    {
        Tensor sublayer = IsAttention
            ? AttentionBatch(backend, hidden, b, cos!, sin!, caches, slot)
            : Mlp(backend, hidden, b);
        Tensor joined = HypersphereOps.Interpolate(backend, hidden, sublayer, _alpha!, _oneMinusAlpha!, _norms.Hidden, _cfg.NormEps);
        sublayer.Dispose();
        return joined;
    }

    private Tensor Attention(IBackend backend, Tensor hidden, int t, int posStart, IKvCache cache, int slot, Tensor cos, Tensor sin)
    {
        int hq = _cfg.NumHeads, hkv = _cfg.NumKvHeads, d = _cfg.HeadDim;
        // RoPE reads positions from the token-major shape, so q and k stay [1,t,heads,d] until after it; a single token's
        // token-major and head-major layouts are byte-identical, so v is projected head-major and the attention output is
        // consumed as is, skipping two of prefill's four permute copies on every decode step.
        bool single = t == 1;
        Tensor q = new(new TensorShape(1, t, hq, d), DType.F32);
        Tensor k = new(new TensorShape(1, t, hkv, d), DType.F32);
        Tensor v = new(single ? new TensorShape(1, hkv, 1, d) : new TensorShape(1, t, hkv, d), DType.F32);
        ProjectQkv(backend, q, k, v, hidden, t);
        Tensor gate = ProjectGate(backend, hidden, t);
        backend.ApplyRopeSingle(q, cos, sin, d);
        backend.ApplyRopeSingle(k, cos, sin, d);
        Tensor qNormed = HypersphereOps.Normalized(backend, q, _norms.Head, _cfg.NormEps);
        q.Dispose();
        Tensor kNormed = HypersphereOps.Normalized(backend, k, _norms.Head, _cfg.NormEps);
        k.Dispose();
        Tensor qHeads = new(new TensorShape(1, hq, t, d), DType.F32);
        backend.Permute0213(qHeads, qNormed, t, hq, d);
        qNormed.Dispose();
        // Per-head, per-dimension query scale: a [heads, d] vector broadcast over the head-major layout's sequence axis.
        Tensor qScaled = new(qHeads.Shape, DType.F32);
        backend.AffineBroadcastLastDim(qScaled, qHeads, _qkScale!, null);
        qHeads.Dispose();
        Tensor kHeads = new(new TensorShape(1, hkv, t, d), DType.F32);
        backend.Permute0213(kHeads, kNormed, t, hkv, d);
        kNormed.Dispose();
        Tensor vHeads = ToHeadMajor(backend, v, t, hkv, d);
        cache.AppendStep(backend, slot, kHeads, vHeads);
        kHeads.Dispose();
        vHeads.Dispose();
        Tensor attn = new(new TensorShape(1, hq, t, d), DType.F32);
        backend.FlashAttention(attn, qScaled, cache.KeyPrefix(slot), cache.ValuePrefix(slot), posStart + t, _cfg.KvGroup, causal: true,
            qOffset: posStart, AttentionScale, 0f, null, Window, null);
        qScaled.Dispose();
        if (single) return GateAndProject(backend, attn, gate, 1);
        Tensor attnFlat = new(new TensorShape(1, t, _cfg.QDim), DType.F32);
        backend.Permute0213(attnFlat, attn, hq, t, d);
        attn.Dispose();
        return GateAndProject(backend, attnFlat, gate, t);
    }

    /// <summary>The head-major <c>[1, heads, t, d]</c> copy of a token-major <c>[1, t, heads, d]</c> tensor, consuming it; a single token is already
    /// in that layout, so it is returned as is.</summary>
    private static Tensor ToHeadMajor(IBackend backend, Tensor tokenMajor, int t, int heads, int d)
    {
        if (t == 1) return tokenMajor;
        Tensor headMajor = new(new TensorShape(1, heads, t, d), DType.F32);
        backend.Permute0213(headMajor, tokenMajor, t, heads, d);
        tokenMajor.Dispose();
        return headMajor;
    }

    private Tensor AttentionGraphStep(IBackend backend, Tensor hidden, IKvCache cache, int slot, Tensor cosTable, Tensor sinTable, ulong devicePos)
    {
        int hq = _cfg.NumHeads, hkv = _cfg.NumKvHeads, d = _cfg.HeadDim;
        // At T = 1 the token-major [1,1,heads,d] and head-major [1,heads,1,d] layouts are byte-identical, so q/k/v are allocated
        // head-major straight from the projection: no permutes, and the per-head norm and scale index heads as rows.
        Tensor q = new(new TensorShape(1, hq, 1, d), DType.F32);
        Tensor k = new(new TensorShape(1, hkv, 1, d), DType.F32);
        Tensor v = new(new TensorShape(1, hkv, 1, d), DType.F32);
        ProjectQkv(backend, q, k, v, hidden, 1);
        Tensor gate = ProjectGate(backend, hidden, 1);
        backend.RopeApplyDecodeStep(q, cosTable, sinTable, d, interleaved: false, devicePos);
        backend.RopeApplyDecodeStep(k, cosTable, sinTable, d, interleaved: false, devicePos);
        Tensor qNormed = HypersphereOps.Normalized(backend, q, _norms.Head, _cfg.NormEps);
        q.Dispose();
        Tensor qScaled = new(qNormed.Shape, DType.F32);
        backend.AffineBroadcastLastDim(qScaled, qNormed, _qkScale!, null);
        qNormed.Dispose();
        Tensor kNormed = HypersphereOps.Normalized(backend, k, _norms.Head, _cfg.NormEps);
        k.Dispose();
        Tensor kFull = cache.KeyPrefix(slot);
        Tensor vFull = cache.ValuePrefix(slot);
        backend.KvCacheAppendDev(kFull, kNormed, 0, devicePos);
        backend.KvCacheAppendDev(vFull, v, 0, devicePos);
        kNormed.Dispose();
        v.Dispose();
        Tensor attn = new(new TensorShape(1, hq, 1, d), DType.F32);
        backend.FlashAttentionDev(attn, qScaled, kFull, vFull, 0, _cfg.KvGroup, causal: true, 0, AttentionScale, devicePos, 0f, Window);
        qScaled.Dispose();
        return GateAndProject(backend, attn, gate, 1);
    }

    private Tensor AttentionBatch(IBackend backend, Tensor hidden, int b, Tensor cos, Tensor sin, IKvCache[] caches, int slot)
    {
        int hq = _cfg.NumHeads, hkv = _cfg.NumKvHeads, d = _cfg.HeadDim;
        Tensor q = new(new TensorShape(1, b, hq, d), DType.F32);
        Tensor k = new(new TensorShape(1, b, hkv, d), DType.F32);
        Tensor v = new(new TensorShape(1, b, hkv, d), DType.F32);
        ProjectQkv(backend, q, k, v, hidden, b);
        Tensor gate = ProjectGate(backend, hidden, b);
        backend.ApplyRopeSingle(q, cos, sin, d);
        backend.ApplyRopeSingle(k, cos, sin, d);
        Tensor qNormed = HypersphereOps.Normalized(backend, q, _norms.Head, _cfg.NormEps);
        q.Dispose();
        Tensor kNormed = HypersphereOps.Normalized(backend, k, _norms.Head, _cfg.NormEps);
        k.Dispose();
        Tensor[] segments = new Tensor[b];
        for (int s = 0; s < b; s++)
        {
            // Each sequence's token sliced out of the batched projections lands directly in the head-major [1,H,1,D] layout.
            Tensor qHeads = new(new TensorShape(1, hq, 1, d), DType.F32);
            Tensor kHeads = new(new TensorShape(1, hkv, 1, d), DType.F32);
            Tensor vHeads = new(new TensorShape(1, hkv, 1, d), DType.F32);
            backend.SliceRows(qHeads, qNormed, s * hq);
            backend.SliceRows(kHeads, kNormed, s * hkv);
            backend.SliceRows(vHeads, v, s * hkv);
            Tensor qScaled = new(qHeads.Shape, DType.F32);
            backend.AffineBroadcastLastDim(qScaled, qHeads, _qkScale!, null);
            qHeads.Dispose();
            IKvCache cache = caches[s];
            cache.AppendStep(backend, slot, kHeads, vHeads);
            kHeads.Dispose();
            vHeads.Dispose();
            int position = cache.CurrentLength;
            Tensor segment = new(new TensorShape(1, hq, 1, d), DType.F32);
            backend.FlashAttention(segment, qScaled, cache.KeyPrefix(slot), cache.ValuePrefix(slot), position + 1, _cfg.KvGroup, causal: true,
                qOffset: position, AttentionScale, 0f, null, Window, null);
            qScaled.Dispose();
            segments[s] = segment;
        }
        qNormed.Dispose();
        kNormed.Dispose();
        v.Dispose();
        Tensor attnConcat = new(new TensorShape(b, hq, 1, d), DType.F32);
        backend.Concat(attnConcat, segments, dim: 0);
        foreach (Tensor segment in segments) segment.Dispose();
        return GateAndProject(backend, attnConcat, gate, b);
    }

    /// <summary>Projects Q/K/V from <paramref name="input"/> into the caller's pre-shaped tensors, one fused GEMV when the three weights share a dtype.</summary>
    private void ProjectQkv(IBackend backend, Tensor q, Tensor k, Tensor v, Tensor input, int rows)
    {
        if (_qkvW is null)
        {
            ProjectionOps.Project(backend, q, input, _qW!, null, _cfg.LowVramQuant);
            ProjectionOps.Project(backend, k, input, _kW!, null, _cfg.LowVramQuant);
            ProjectionOps.Project(backend, v, input, _vW!, null, _cfg.LowVramQuant);
            return;
        }
        int nq = _cfg.QDim, nk = _cfg.KvDim;
        using Tensor qkv = new(new TensorShape(1, rows, nq + 2 * nk), DType.F32);
        ProjectionOps.Project(backend, qkv, input, _qkvW, null, _cfg.LowVramQuant);
        backend.SliceLastDim(q, qkv, 0);
        backend.SliceLastDim(k, qkv, nq);
        backend.SliceLastDim(v, qkv, nq + nk);
    }

    /// <summary>The raw attention output gate logits <c>[1, rows, QDim]</c>.</summary>
    private Tensor ProjectGate(IBackend backend, Tensor input, int rows)
    {
        Tensor gate = new(new TensorShape(1, rows, _cfg.QDim), DType.F32);
        ProjectionOps.Project(backend, gate, input, _gateW!, null, _cfg.LowVramQuant);
        return gate;
    }

    /// <summary>Gates the merged attention output (<paramref name="attn"/>, any layout with <c>rows × QDim</c> elements in token-major order) by
    /// <c>sigmoid(gate ⊙ scale)</c>, unit-norms the gated vector and applies the output projection. Consumes both inputs.</summary>
    private Tensor GateAndProject(IBackend backend, Tensor attn, Tensor gate, int rows)
    {
        TensorShape merged = new(1, rows, _cfg.QDim);
        Tensor gateScaled = new(merged, DType.F32);
        backend.AffineBroadcastLastDim(gateScaled, gate, _gateScale!, null);
        gate.Dispose();
        backend.Sigmoid(gateScaled, gateScaled);
        Tensor gated = new(merged, DType.F32);
        backend.Mul(gated, attn, gateScaled);
        attn.Dispose();
        gateScaled.Dispose();
        Tensor normed = HypersphereOps.Normalized(backend, gated, _norms.Attention, _cfg.NormEps);
        gated.Dispose();
        Tensor output = new(new TensorShape(1, rows, _cfg.HiddenSize), DType.F32);
        ProjectionOps.Project(backend, output, normed, _oW!, null, _cfg.LowVramQuant);
        normed.Dispose();
        return output;
    }

    private Tensor Mlp(IBackend backend, Tensor hidden, int rows)
    {
        int ff = _cfg.IntermediateSize;
        Tensor act = new(new TensorShape(1, rows, ff), DType.F32);
        if (_gateUpW is not null)
        {
            using Tensor gateUp = new(new TensorShape(1, rows, 2 * ff), DType.F32);
            ProjectionOps.Project(backend, gateUp, hidden, _gateUpW, null, _cfg.LowVramQuant);
            using Tensor scaled = new(gateUp.Shape, DType.F32);
            backend.AffineBroadcastLastDim(scaled, gateUp, _suvScale!, null);
            backend.GluActivate(act, scaled, ff, gelu: false);
        }
        else
        {
            using Tensor gateRaw = new(act.Shape, DType.F32);
            using Tensor upRaw = new(act.Shape, DType.F32);
            ProjectionOps.Project(backend, gateRaw, hidden, _ffnGateW!, null, _cfg.LowVramQuant);
            ProjectionOps.Project(backend, upRaw, hidden, _ffnUpW!, null, _cfg.LowVramQuant);
            using Tensor gate = new(act.Shape, DType.F32);
            using Tensor up = new(act.Shape, DType.F32);
            backend.AffineBroadcastLastDim(gate, gateRaw, _ffnGateScale!, null);
            backend.AffineBroadcastLastDim(up, upRaw, _ffnUpScale!, null);
            backend.Silu(gate, gate);
            backend.Mul(act, gate, up);
        }
        Tensor normed = HypersphereOps.Normalized(backend, act, _norms.Ffn, _cfg.NormEps);
        act.Dispose();
        Tensor output = new(new TensorShape(1, rows, _cfg.HiddenSize), DType.F32);
        ProjectionOps.Project(backend, output, normed, _ffnDownW!, null, _cfg.LowVramQuant);
        normed.Dispose();
        return output;
    }

    /// <summary>The rank-2 weight under <paramref name="key"/>, checked against the <c>[rows, cols]</c> this layer projects with.</summary>
    private static Tensor Require(IReadOnlyDictionary<string, Tensor> w, string key, int rows, int cols)
    {
        if (!w.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"Looped transformer weight '{key}' is missing from the checkpoint.");
        if (t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
            throw new ArgumentException($"Looped transformer weight '{key}' is {t.Shape}; expected [{rows}, {cols}].");
        return t;
    }

    /// <summary>Reads the F32 vector under <paramref name="key"/> and bakes <c>v · factor</c>, squared or made non-negative as nGPT's forward does.</summary>
    private unsafe Tensor Bake(IReadOnlyDictionary<string, Tensor> w, string key, int length, float factor, bool square, bool abs)
    {
        if (!w.TryGetValue(key, out Tensor? source))
            throw new KeyNotFoundException($"Looped transformer scale '{key}' is missing from the checkpoint.");
        Tensor f32 = ProjectionOps.EnsureF32(source);
        try
        {
            if (f32.ElementCount != length)
                throw new ArgumentException($"Looped transformer scale '{key}' has {f32.ElementCount} values; expected {length}.");
            Tensor baked = new(new TensorShape(length), DType.F32);
            float* src = (float*)f32.DataPointer;
            float* dst = (float*)baked.DataPointer;
            for (int i = 0; i < length; i++)
            {
                float value = src[i] * factor;
                if (square) value *= value;
                if (abs) value = MathF.Abs(value);
                dst[i] = value;
            }
            _owned.Add(baked);
            return baked;
        }
        finally
        {
            if (!ReferenceEquals(f32, source)) f32.Dispose();
        }
    }

    private unsafe Tensor OneMinus(Tensor alpha)
    {
        int n = (int)alpha.ElementCount;
        Tensor result = new(new TensorShape(n), DType.F32);
        float* src = (float*)alpha.DataPointer;
        float* dst = (float*)result.DataPointer;
        for (int i = 0; i < n; i++) dst[i] = 1f - src[i];
        _owned.Add(result);
        return result;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }
}
