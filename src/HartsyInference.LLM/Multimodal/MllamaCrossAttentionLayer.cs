using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Multimodal;

/// <summary>The gated cross-attention decoder layer of Llama-3.2-Vision (mllama), interleaved into the text decoder at layers <c>[3,8,13,18,23,28,33,38]</c> (11B) instead of splicing image tokens into the sequence like every other VLM here.</summary>
/// <remarks>
/// <code>
///   h = text + tanh(attn_gate) · o_proj( cross_attn( q=q_norm(q_proj(rms(text))), k=k_norm(k_proj(vision)), v=v_proj(vision) ) )
///   out = h    + tanh(mlp_gate)  · mlp( rms(h) )
/// </code>
/// The query comes from the text hidden states; the keys/values from the (fixed) vision features, so the attention is bidirectional over all vision tokens (no causal mask). Q/K are RMSNorm-ed per head. Two learned scalar gates (<c>cross_attn_attn_gate</c>, <c>cross_attn_mlp_gate</c>) are passed through <c>tanh</c>; both are ≈0 at init so a freshly-initialized layer is a no-op (a strong correctness signal). All math runs through <see cref="IBackend"/>; the attention reuses the decoder's GQA FlashAttention with <c>causal:false</c>.
/// </remarks>
public sealed unsafe class MllamaCrossAttentionLayer
{
    private readonly int _hidden, _numHeads, _numKvHeads, _headDim, _inter;
    private readonly float _rmsEps;
    private readonly bool _lowVram;
    private readonly AttentionLayer _attn;

    private Tensor _inNorm = null!, _postAttnNorm = null!;
    private Tensor _qW = null!, _kW = null!, _vW = null!, _oW = null!;
    private Tensor _qNorm = null!, _kNorm = null!;
    private Tensor _gateW = null!, _upW = null!, _downW = null!;
    private float _attnGate, _mlpGate;   // raw (pre-tanh) learned scalar gates

    public MllamaCrossAttentionLayer(int hidden, int numHeads, int numKvHeads, int headDim, int intermediate, float rmsEps, bool lowVram = false)
    {
        _hidden = hidden; _numHeads = numHeads; _numKvHeads = numKvHeads; _headDim = headDim;
        _inter = intermediate; _rmsEps = rmsEps; _lowVram = lowVram;
        _attn = new AttentionLayer(new AttentionLayerConfig
        {
            Heads = numHeads, KvHeads = numKvHeads, HeadDim = headDim, ComputeDType = DType.F32, Scale = 1f / MathF.Sqrt(headDim),
        });
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inNorm = w[$"{prefix}.input_layernorm.weight"];
        _postAttnNorm = w[$"{prefix}.post_attention_layernorm.weight"];
        _qW = w[$"{prefix}.cross_attn.q_proj.weight"];
        _kW = w[$"{prefix}.cross_attn.k_proj.weight"];
        _vW = w[$"{prefix}.cross_attn.v_proj.weight"];
        _oW = w[$"{prefix}.cross_attn.o_proj.weight"];
        _qNorm = w[$"{prefix}.cross_attn.q_norm.weight"];
        _kNorm = w[$"{prefix}.cross_attn.k_norm.weight"];
        _gateW = w[$"{prefix}.mlp.gate_proj.weight"];
        _upW = w[$"{prefix}.mlp.up_proj.weight"];
        _downW = w[$"{prefix}.mlp.down_proj.weight"];
        _attnGate = Scalar(w[$"{prefix}.cross_attn_attn_gate"]);
        _mlpGate = Scalar(w[$"{prefix}.cross_attn_mlp_gate"]);
    }

    /// <summary>Cross-attention + gated FFN: <paramref name="text"/> is <c>[1, t, hidden]</c> (the decoder query), <paramref name="vision"/> is <c>[1, L, hidden]</c> (the encoded image features); returns <c>[1, t, hidden]</c>.</summary>
    /// <param name="rope">Optional MOSS-VL tables: the query rows are rotated by <see cref="MultimodalRope.TextCos"/>, the key rows by the vision tables, and rows with a 0 <see cref="MultimodalRope.RowMask"/> contribute nothing. Null (the mllama default) applies no rotation and no row mask.</param>
    public Tensor Forward(IBackend backend, Tensor text, int t, Tensor vision, int visionLen, MultimodalRope? rope = null,
        CrossKvCache? kvCache = null, int layer = 0)
    {
        if (kvCache?.Get(layer) is { } cached) return Attend(backend, text, t, cached.K, cached.V, visionLen, rope);

        (Tensor kMh, Tensor vMh) = ProjectVision(backend, vision, visionLen, rope);
        if (kvCache is not null)
        {
            kvCache.Set(layer, kMh, vMh);   // the cache owns them for the rest of the request
            return Attend(backend, text, t, kMh, vMh, visionLen, rope);
        }
        try
        {
            return Attend(backend, text, t, kMh, vMh, visionLen, rope);
        }
        finally
        {
            kMh.Dispose();
            vMh.Dispose();
        }
    }

    /// <summary>Projects the vision sequence to head-major keys and values: <c>k_norm(k_proj(vision))</c> rotated by the vision tables, and
    /// <c>v_proj(vision)</c>. The caller owns both results.</summary>
    private (Tensor K, Tensor V) ProjectVision(IBackend backend, Tensor vision, int visionLen, MultimodalRope? rope)
    {
        int hkv = _numKvHeads, d = _headDim;
        Tensor k = new(new TensorShape(1, visionLen, hkv, d), DType.F32);
        GenericTransformer.Project(backend, k, vision, _kW, null, lowVram: _lowVram);
        Tensor kN = new(new TensorShape(1, visionLen, hkv, d), DType.F32);
        backend.RmsNorm(kN, k, _kNorm, _rmsEps);
        k.Dispose();
        if (rope is not null) backend.Rotary(RotaryCall.TokenMajor(kN, rope.VisionCos, rope.VisionSin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));
        Tensor v = new(new TensorShape(1, visionLen, hkv, d), DType.F32);
        GenericTransformer.Project(backend, v, vision, _vW, null, lowVram: _lowVram);

        // Head-major [1, heads, seq, d] for the GQA FlashAttention.
        Tensor kMh = new(new TensorShape(1, hkv, visionLen, d), DType.F32); backend.Permute0213(kMh, kN, visionLen, hkv, d);
        Tensor vMh = new(new TensorShape(1, hkv, visionLen, d), DType.F32); backend.Permute0213(vMh, v, visionLen, hkv, d);
        kN.Dispose(); v.Dispose();
        return (kMh, vMh);
    }

    /// <summary>The query side, attention over borrowed head-major vision keys and values (bidirectional: no causal mask), and the gated
    /// FFN. Does not dispose <paramref name="kMh"/> or <paramref name="vMh"/>.</summary>
    private Tensor Attend(IBackend backend, Tensor text, int t, Tensor kMh, Tensor vMh, int visionLen, MultimodalRope? rope)
    {
        int hq = _numHeads, d = _headDim;
        TensorShape flat = new(1, t, _hidden);

        // --- Cross attention ---
        Tensor pre = new(flat, DType.F32);
        backend.RmsNorm(pre, text, _inNorm, _rmsEps);

        Tensor q = new(new TensorShape(1, t, hq, d), DType.F32);
        GenericTransformer.Project(backend, q, pre, _qW, null, lowVram: _lowVram);
        pre.Dispose();
        Tensor qN = new(new TensorShape(1, t, hq, d), DType.F32);
        backend.RmsNorm(qN, q, _qNorm, _rmsEps);   // q-norm over the head dim
        q.Dispose();
        if (rope is not null) backend.Rotary(RotaryCall.TokenMajor(qN, rope.TextCos, rope.TextSin, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));

        Tensor qMh = new(new TensorShape(1, hq, t, d), DType.F32); backend.Permute0213(qMh, qN, t, hq, d);
        qN.Dispose();

        // Only the query is consumed here: the keys and values are borrowed (the kv cache or the caller disposes them).
        Tensor attn = _attn.AttendOverKvHeadMajor(backend, qMh, kMh, vMh, null, 0, 0,
            new KvAttentionCall { KvLength = visionLen, Bidirectional = true, Bias = rope?.CrossBias }, OperandOwnership.Query);
        Tensor attnFlat = _attn.MergeHeads(backend, attn, new TensorShape(1, t, hq * d));
        Tensor attnOut = new(flat, DType.F32);
        GenericTransformer.Project(backend, attnOut, attnFlat, _oW, null, lowVram: _lowVram);
        attnFlat.Dispose();
        attnOut = ApplyRowMask(backend, attnOut, rope);

        // Gated residual: text + tanh(attn_gate) · attnOut.
        Tensor h = new(flat, DType.F32);
        backend.Scale(attnOut, attnOut, MathF.Tanh(_attnGate));
        backend.Add(h, text, attnOut);
        attnOut.Dispose();

        // --- Gated FFN ---
        Tensor postNorm = new(flat, DType.F32);
        backend.RmsNorm(postNorm, h, _postAttnNorm, _rmsEps);
        Tensor mlp = ApplyRowMask(backend, SwiGlu(backend, postNorm, t), rope);
        postNorm.Dispose();
        backend.Scale(mlp, mlp, MathF.Tanh(_mlpGate));
        Tensor outp = new(flat, DType.F32);
        backend.Add(outp, h, mlp);
        h.Dispose(); mlp.Dispose();
        return outp;
    }

    /// <summary>Returns <paramref name="x"/> with each row scaled by the rope's row mask, disposing <paramref name="x"/>; returns it unchanged when there is no mask.</summary>
    private static Tensor ApplyRowMask(IBackend backend, Tensor x, MultimodalRope? rope)
    {
        if (rope?.RowMask is null) return x;
        Tensor masked = new(x.Shape, DType.F32);
        backend.Mul(masked, x, rope.RowMask);
        x.Dispose();
        return masked;
    }

    private Tensor SwiGlu(IBackend backend, Tensor x, int t)
    {
        TensorShape ff = new(1, t, _inter);
        Tensor gate = new(ff, DType.F32);
        GenericTransformer.Project(backend, gate, x, _gateW, null, lowVram: _lowVram);
        Tensor act = new(ff, DType.F32);
        backend.Silu(act, gate);
        gate.Dispose();
        Tensor up = new(ff, DType.F32);
        GenericTransformer.Project(backend, up, x, _upW, null, lowVram: _lowVram);
        Tensor comb = new(ff, DType.F32);
        backend.Mul(comb, act, up);
        act.Dispose(); up.Dispose();
        Tensor outp = new(new TensorShape(1, t, _hidden), DType.F32);
        GenericTransformer.Project(backend, outp, comb, _downW, null, lowVram: _lowVram);
        comb.Dispose();
        return outp;
    }

    /// <summary>The loaded weight tensors (for GPU residency / lifetime management by the host transformer).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _inNorm; yield return _postAttnNorm;
        yield return _qW; yield return _kW; yield return _vW; yield return _oW;
        yield return _qNorm; yield return _kNorm;
        yield return _gateW; yield return _upW; yield return _downW;
    }

    private static float Scalar(Tensor t)
    {
        if (t.DType != DType.F32) throw new NotSupportedException("mllama gate scalar must be F32.");
        return *(float*)t.DataPointer;
    }
}
