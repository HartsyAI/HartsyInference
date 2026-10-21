using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>One LTX-2.3 attention layer (<c>LTX2Attention</c> + <c>LTX2AudioVideoAttnProcessor</c>), single sample
/// (B=1) over <c>[S, dim]</c>. Generalizes <see cref="LtxVideoBlock"/>'s attention with three LTX-2 additions:
/// (1) the query, key/value, and output widths may all differ (so the same class serves video/audio self-attn,
/// text cross-attn, and the cross-modal a2v/v2a attentions); (2) per-head gating — <c>to_gate_logits</c> projects
/// the query input to one logit per head and the attention output is scaled by <c>2·sigmoid(logit)</c> (the factor
/// 2 makes zero-init logits give unit gates); (3) the query and key RoPE may be supplied separately (a2v/v2a use
/// video coords for one side and audio coords for the other).</summary>
///
/// <remarks>Flow: <c>gate = to_gate_logits(qInput)</c>; <c>q,k,v = to_q(qInput), to_k(kvInput), to_v(kvInput)</c>;
/// full-width <c>norm_q</c>/<c>norm_k</c> RMSNorm; optional interleaved RoPE on q (and k); SDPA over heads; flatten;
/// per-head gate multiply; <c>to_out</c>. QK-norm/RoPE span the full inner dim before the head split. Cross-attn
/// to text passes no RoPE.</remarks>
public sealed class LtxVideo2Attention
{
    private readonly int _qInDim, _kvInDim, _inner, _heads, _headDim, _outDim;
    private readonly float _qkEps;

    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk, _gateW, _gateB;
    private AttentionLayer? _layer;

    public LtxVideo2Attention(int qInDim, int kvInDim, int heads, int headDim, int outDim, float qkEps)
    {
        _qInDim = qInDim;
        _kvInDim = kvInDim;
        _heads = heads;
        _headDim = headDim;
        _inner = heads * headDim;
        _outDim = outDim;
        _qkEps = qkEps;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _qW = w[$"{p}.to_q.weight"]; w.TryGetValue($"{p}.to_q.bias", out _qB);
        _kW = w[$"{p}.to_k.weight"]; w.TryGetValue($"{p}.to_k.bias", out _kB);
        long qIn = _qW.Shape[_qW.Shape.Rank - 1], kvIn = _kW.Shape[_kW.Shape.Rank - 1];
        if (qIn != _qInDim || kvIn != _kvInDim)
            throw new InvalidOperationException(
                $"'{p}' attention weights are [{qIn}]/[{kvIn}] wide; the block declared q={_qInDim} kv={_kvInDim}.");
        _vW = w[$"{p}.to_v.weight"]; w.TryGetValue($"{p}.to_v.bias", out _vB);
        _oW = w[$"{p}.to_out.0.weight"]; w.TryGetValue($"{p}.to_out.0.bias", out _oB);
        _nq = TensorCasts.LoadF32(w, $"{p}.q_norm.weight");
        _nk = TensorCasts.LoadF32(w, $"{p}.k_norm.weight");
        // Per-head output gating is a 2.3-only feature; earlier LTX-2 (e.g. 19B) omits it (ungated attention).
        if (w.TryGetValue($"{p}.to_gate_logits.weight", out Tensor? gw)) { _gateW = gw; w.TryGetValue($"{p}.to_gate_logits.bias", out _gateB); }

        // Grouped by input so an int8 backend quantizes each input once; RMS-normed Q/K bound the scores, so F16 is allowed.
        AttentionLayerConfig config = new()
        {
            Heads = _heads, KvHeads = _heads, HeadDim = _headDim, OutDim = _outDim,
            Projection = AttentionProjection.GroupedByInput, QkNorm = AttentionQkNorm.FullWidth, FullWidthNormEps = _qkEps,
            Gate = _gateW is null ? AttentionGate.None : AttentionGate.PerHeadDoubledSigmoid, WidenKvToQuery = true,
            TokenMajorRoute = AttentionRoute.TokenMajor, TokenMajorKnob = EngineKnobs.Ltx2Tokenmajor, AllowF16 = true,
        };
        _layer = new AttentionLayer(config, new AttentionLayerWeights
        {
            Query = _qW, QueryBias = _qB, Key = _kW, KeyBias = _kB, Value = _vW, ValueBias = _vB, Output = _oW, OutputBias = _oB,
            Gate = _gateW, GateBias = _gateB, QueryNorm = _nq, KeyNorm = _nk,
        });
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk, _gateW, _gateB })
            if (t is not null) yield return t;
    }

    /// <summary>Attention over query rows <paramref name="qInput"/> <c>[Sq, qInDim]</c> attending to
    /// <paramref name="kvInput"/> <c>[Sk, kvInDim]</c>. Pass <paramref name="qRope"/>/<paramref name="kRope"/> null
    /// to skip RoPE (text cross-attn). Returns <c>[Sq, outDim]</c>; caller owns it.</summary>
    public Tensor Forward(IBackend backend, Tensor qInput, Tensor kvInput, LtxVideo2Rope? qRope, Tensor? qCos,
        Tensor? qSin, LtxVideo2Rope? kRope, Tensor? kCos, Tensor? kSin, Tensor? mask)
    {
        AttentionLayer layer = _layer ?? throw new InvalidOperationException("LtxVideo2Attention.LoadWeights has not run.");
        AttentionCallContext context = new()
        {
            QueryRotation = Rotation(qRope, qCos, qSin), KeyRotation = Rotation(kRope, kCos, kSin), Bias = mask,
        };
        return layer.Forward(backend, qInput, kvInput, in context);
    }

    /// <summary>LTX-2.x rotates split halves, fused with the norm; the LTX-0.9 lineage rotates interleaved pairs over the
    /// full width after it.</summary>
    private QkRotation Rotation(LtxVideo2Rope? rope, Tensor? cos, Tensor? sin)
    {
        if (rope is null)
            return QkRotation.None;
        if (rope.Flavor == LtxVideo2Rope.RopeType.Split)
            return new QkRotation(QkRotationKind.FullWidthSplit, cos, sin);
        if (rope.Dim != _inner)
            throw new ArgumentException($"x dim {_inner} != rope dim {rope.Dim}.", nameof(rope));
        return new QkRotation(QkRotationKind.FullWidthInterleaved, cos, sin);
    }
}
