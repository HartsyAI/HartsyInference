using HartsyInference.Core.Backends;
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
public sealed unsafe class LtxVideo2Attention
{
    private readonly int _qInDim, _kvInDim, _inner, _heads, _headDim, _outDim;
    private readonly float _qkEps;

    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk, _gateW, _gateB;

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
        _vW = w[$"{p}.to_v.weight"]; w.TryGetValue($"{p}.to_v.bias", out _vB);
        _oW = w[$"{p}.to_out.0.weight"]; w.TryGetValue($"{p}.to_out.0.bias", out _oB);
        _nq = LoadF32(w, $"{p}.q_norm.weight");
        _nk = LoadF32(w, $"{p}.k_norm.weight");
        _gateW = w[$"{p}.to_gate_logits.weight"]; w.TryGetValue($"{p}.to_gate_logits.bias", out _gateB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk, _gateW, _gateB })
            if (t is not null) yield return t;
    }

    /// <summary>Attention over query rows <paramref name="qInput"/> <c>[Sq, qInDim]</c> attending to
    /// <paramref name="kvInput"/> <c>[Sk, kvInDim]</c>. Pass <paramref name="qRope"/>/<paramref name="kRope"/> null
    /// to skip RoPE (text cross-attn). Returns <c>[Sq, outDim]</c>; caller owns it.</summary>
    public Tensor Forward(IBackend backend, Tensor qInput, Tensor kvInput,
        LtxVideo2Rope? qRope, Tensor? qCos, Tensor? qSin,
        LtxVideo2Rope? kRope, Tensor? kCos, Tensor? kSin, Tensor? mask)
    {
        int sq = (int)qInput.Shape[0];
        int sk = (int)kvInput.Shape[0];

        // Gate logits on the (modulated, normed) query input — one logit per head.
        Tensor gateLogits = new(new TensorShape(sq, _heads), DType.F32);
        backend.Linear(gateLogits, qInput, _gateW!, _gateB);

        Tensor q = new(new TensorShape(sq, _inner), DType.F32);
        backend.Linear(q, qInput, _qW!, _qB);
        Tensor k = new(new TensorShape(sk, _inner), DType.F32);
        backend.Linear(k, kvInput, _kW!, _kB);
        Tensor v = new(new TensorShape(sk, _inner), DType.F32);
        backend.Linear(v, kvInput, _vW!, _vB);

        // Full-width QK-RMSNorm (across heads), then optional interleaved RoPE before the head split.
        Tensor qn = new(q.Shape, DType.F32); backend.RmsNorm(qn, q, _nq!, _qkEps); q.Dispose();
        Tensor kn = new(k.Shape, DType.F32); backend.RmsNorm(kn, k, _nk!, _qkEps); k.Dispose();
        if (qRope is not null) qRope.ApplyRotary(qn, qCos!, qSin!);
        if (kRope is not null) kRope.ApplyRotary(kn, kCos!, kSin!);

        Tensor qMh = ToBhsd(qn, sq); qn.Dispose();
        Tensor kMh = ToBhsd(kn, sk); kn.Dispose();
        Tensor vMh = ToBhsd(v, sk); v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attn = new(new TensorShape(1, _heads, sq, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = FromBhsd(attn, sq); attn.Dispose();   // [Sq, inner]
        ApplyGate(flat, gateLogits, sq);
        gateLogits.Dispose();

        Tensor outT = new(new TensorShape(sq, _outDim), DType.F32);
        backend.Linear(outT, flat, _oW!, _oB);
        flat.Dispose();
        return outT;
    }

    /// <summary>Per-head output gating: each head's slice scaled by <c>2·sigmoid(logit_head)</c>.</summary>
    private void ApplyGate(Tensor flat, Tensor gateLogits, int sq)
    {
        float* fp = (float*)flat.DataPointer;
        float* gl = (float*)gateLogits.DataPointer;
        for (int i = 0; i < sq; i++)
            for (int h = 0; h < _heads; h++)
            {
                float gate = 2.0f / (1.0f + MathF.Exp(-gl[i * _heads + h]));
                float* row = fp + (long)i * _inner + (long)h * _headDim;
                for (int d = 0; d < _headDim; d++) row[d] *= gate;
            }
    }

    private Tensor ToBhsd(Tensor x, int s)
    {
        Tensor o = new(new TensorShape(1, _heads, s, _headDim), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int i = 0; i < s; i++)
            for (int h = 0; h < _heads; h++)
            {
                long src = (long)i * _inner + (long)h * _headDim;
                long dst = ((long)h * s + i) * _headDim;
                Buffer.MemoryCopy(xp + src, op + dst, (long)_headDim * 4, (long)_headDim * 4);
            }
        return o;
    }

    private Tensor FromBhsd(Tensor x, int s)
    {
        Tensor o = new(new TensorShape(s, _inner), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int h = 0; h < _heads; h++)
            for (int i = 0; i < s; i++)
            {
                long src = ((long)h * s + i) * _headDim;
                long dst = (long)i * _inner + (long)h * _headDim;
                Buffer.MemoryCopy(xp + src, op + dst, (long)_headDim * 4, (long)_headDim * 4);
            }
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
