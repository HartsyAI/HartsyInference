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
        // Per-head output gating is a 2.3-only feature; earlier LTX-2 (e.g. 19B) omits it (ungated attention).
        if (w.TryGetValue($"{p}.to_gate_logits.weight", out Tensor? gw)) { _gateW = gw; w.TryGetValue($"{p}.to_gate_logits.bias", out _gateB); }
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

        // Gate logits on the (modulated, normed) query input — one logit per head. Absent on ungated (pre-2.3) checkpoints.
        Tensor? gateLogits = null;
        if (_gateW is not null)
        {
            gateLogits = new(new TensorShape(sq, _heads), DType.F32);
            backend.Linear(gateLogits, qInput, _gateW!, _gateB);
        }

        Tensor q = new(new TensorShape(sq, _inner), DType.F32);
        backend.Linear(q, qInput, _qW!, _qB);
        Tensor k = new(new TensorShape(sk, _inner), DType.F32);
        backend.Linear(k, kvInput, _kW!, _kB);
        Tensor v = new(new TensorShape(sk, _inner), DType.F32);
        backend.Linear(v, kvInput, _vW!, _vB);

        // Full-width QK-RMSNorm (across heads), then optional interleaved RoPE before the head split.
        Tensor qn = new(q.Shape, DType.F32); backend.RmsNorm(qn, q, _nq!, _qkEps); q.Dispose();
        Tensor kn = new(k.Shape, DType.F32); backend.RmsNorm(kn, k, _nk!, _qkEps); k.Dispose();
        if (qRope is not null) qRope.ApplyRotary(backend, qn, qCos!, qSin!);
        if (kRope is not null) kRope.ApplyRotary(backend, kn, kCos!, kSin!);

        Tensor qMh = ToBhsd(backend, qn, sq); qn.Dispose();
        Tensor kMh = ToBhsd(backend, kn, sk); kn.Dispose();
        Tensor vMh = ToBhsd(backend, v, sk); v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attn = new(new TensorShape(1, _heads, sq, _headDim), DType.F32);
        // allowF16: Q and K are RMS-normed above → bounded pre-softmax scores → F16 attention is safe and halves the
        // (dominant) score-matrix traffic. Engine keeps F32 when a mask is present.
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, scale, allowF16: true);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = FromBhsd(backend, attn, sq); attn.Dispose();   // [Sq, inner]
        if (gateLogits is not null) { flat = ApplyGate(backend, flat, gateLogits, sq); gateLogits.Dispose(); }

        Tensor outT = new(new TensorShape(sq, _outDim), DType.F32);
        backend.Linear(outT, flat, _oW!, _oB);
        flat.Dispose();
        return outT;
    }

    /// <summary>Per-head output gating on-device: each head's slice scaled by <c>2·sigmoid(logit_head)</c>. The
    /// per-(row,head) gate is expanded to <c>[Sq, inner]</c> via a constant 0/1 block matrix GEMM (exact copy — one
    /// term per output element), then applied with an elementwise multiply. Was a host <c>DataPointer</c> loop that
    /// drained the SDPA output mid-chain, 6×/block.</summary>
    private Tensor ApplyGate(IBackend backend, Tensor flat, Tensor gateLogits, int sq)
    {
        Tensor sig = new(new TensorShape(sq, _heads), DType.F32);
        backend.Sigmoid(sig, gateLogits);
        Tensor sig2 = new(new TensorShape(sq, _heads), DType.F32);
        backend.Scale(sig2, sig, 2f);
        sig.Dispose();
        Tensor expand = HeadExpandWeights.GetOrAdd((_heads, _headDim), BuildHeadExpand);
        Tensor gateFull = new(new TensorShape(sq, _inner), DType.F32);
        backend.Linear(gateFull, sig2, expand, null);
        sig2.Dispose();
        Tensor o = new(new TensorShape(sq, _inner), DType.F32);
        backend.Mul(o, flat, gateFull);
        gateFull.Dispose();
        flat.Dispose();
        return o;
    }

    // One constant [inner, heads] expansion matrix per (heads, headDim) head layout, shared across every attention
    // instance (per-instance copies would cost ~150 MB across 48 dual-stream blocks). Never disposed: process-lifetime
    // constants totaling <1 MB, re-uploaded on demand if a backend context is torn down.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<(int Heads, int HeadDim), Tensor> HeadExpandWeights = new();

    private static Tensor BuildHeadExpand((int Heads, int HeadDim) key)
    {
        Tensor w = new(new TensorShape(key.Heads * key.HeadDim, key.Heads), DType.F32);
        float* p = (float*)w.DataPointer;
        long total = (long)key.Heads * key.HeadDim * key.Heads;
        for (long i = 0; i < total; i++) p[i] = 0f;
        for (int h = 0; h < key.Heads; h++)
            for (int d = 0; d < key.HeadDim; d++)
                p[((long)h * key.HeadDim + d) * key.Heads + h] = 1f;
        return w;
    }

    // [s, inner]=[s, heads, headDim] → [1, heads, s, headDim], GPU-resident via Permute0213 (was a host DataPointer
    // loop = a D2H sync + host copy + H2D per call, ×3/attention — the same host-excursion that dominated Wan/Flux).
    private Tensor ToBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new(new TensorShape(1, _heads, s, _headDim), DType.F32);
        backend.Permute0213(o, x, s, _heads, _headDim);
        return o;
    }

    // [1, heads, s, headDim] → [s, inner] (inverse of ToBhsd), GPU-resident via Permute0213.
    private Tensor FromBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new(new TensorShape(s, _inner), DType.F32);
        backend.Permute0213(o, x, _heads, s, _headDim);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
