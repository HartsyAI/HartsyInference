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
        // Activation dtype FOLLOWS THE INPUT (the Krea2 pattern) rather than being hardcoded, so the block decides
        // once and the text-connector path — whose surrounding ops are F32 and which runs once per prompt — keeps the
        // F32 baseline. Parameters stay F32 throughout: the F16 RmsNorm/RoPE/Sigmoid kernels take an F16 activation
        // with F32 per-channel params, and SDPA requires an F32 additive mask on either path.
        DType act = qInput.DType;

        // Gate logits on the (modulated, normed) query input — one logit per head. Absent on ungated (pre-2.3) checkpoints.
        Tensor? gateLogits = null;
        if (_gateW is not null)
        {
            gateLogits = new(new TensorShape(sq, _heads), act);
            backend.Linear(gateLogits, qInput, _gateW!, _gateB);
        }

        Tensor q = new(new TensorShape(sq, _inner), act);
        backend.Linear(q, qInput, _qW!, _qB);
        Tensor k = new(new TensorShape(sk, _inner), kvInput.DType);
        backend.Linear(k, kvInput, _kW!, _kB);
        Tensor v = new(new TensorShape(sk, _inner), kvInput.DType);
        backend.Linear(v, kvInput, _vW!, _vB);

        // Full-width QK-RMSNorm (across heads) -> optional RoPE -> head-major, fused into ONE pass per tensor.
        // The unfused sequence read and wrote each [S, inner] tensor three times (RmsNorm, then in-place rope,
        // then Permute0213); at 4992 tokens x 4096 inner that was the single largest non-GEMM cost in the step.
        // SDPA requires output/K/V to share Q's dtype, and a cross-attention K/V stream can arrive F32 (F32 text
        // connectors) against an F16 query — reconcile onto Q's dtype here rather than at each call site.
        Tensor qMh = NormRopeHeadMajor(backend, q, _nq!, qRope, qCos, qSin, sq, act);
        Tensor kMh = NormRopeHeadMajor(backend, k, _nk!, kRope, kCos, kSin, sk, act);
        Tensor vMh = ToBhsd(backend, v, sk, act); v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attn = new(new TensorShape(1, _heads, sq, _headDim), act);
        // allowF16: Q and K are RMS-normed above → bounded pre-softmax scores → F16 attention is safe and halves the
        // (dominant) score-matrix traffic. Engine keeps F32 when a mask is present. Redundant once act is F16 (the
        // native-F16 branch is cast-free and needs no gate), but load-bearing on the F32 connector path.
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, scale, allowF16: true);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = FromBhsd(backend, attn, sq, act); attn.Dispose();   // [Sq, inner]
        if (gateLogits is not null)
        {
            backend.Ltx2HeadGate(flat, gateLogits, sq, _heads, _headDim);
            gateLogits.Dispose();
        }

        Tensor outT = new(new TensorShape(sq, _outDim), act);
        backend.Linear(outT, flat, _oW!, _oB);
        flat.Dispose();
        return outT;
    }

    /// <summary>Norm + optional RoPE + head-major in one backend op. Falls back to the explicit three-op
    /// sequence for an INTERLEAVED rope, which the fused op does not serve (LTX-0.9 lineage; LTX-2.x is Split),
    /// and widens an F32 cross-attention K/V stream onto the query's dtype first, since SDPA requires the pair
    /// to match and the fused op emits its input's dtype.</summary>
    private Tensor NormRopeHeadMajor(IBackend backend, Tensor x, Tensor normW,
        LtxVideo2Rope? rope, Tensor? cos, Tensor? sin, int s, DType act)
    {
        Tensor src = x.DType == act ? x : Cast(backend, x, act);
        if (!ReferenceEquals(src, x)) x.Dispose();
        if (rope is not null && rope.Flavor == LtxVideo2Rope.RopeType.Interleaved)
        {
            Tensor normed = new(src.Shape, src.DType);
            backend.RmsNorm(normed, src, normW, _qkEps);
            src.Dispose();
            rope.ApplyRotary(backend, normed, cos!, sin!);
            Tensor mh = ToBhsd(backend, normed, s, act);
            normed.Dispose();
            return mh;
        }
        Tensor o = new(new TensorShape(1, _heads, s, _headDim), act);
        backend.Ltx2QkNormRopeHeadMajor(o, src, normW, rope is null ? null : cos, rope is null ? null : sin,
            s, _heads, _headDim, _qkEps);
        src.Dispose();
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
    private Tensor ToBhsd(IBackend backend, Tensor x, int s, DType act)
    {
        // Permute0213 is a pure address shuffle with no dtype guard, so it cannot convert: cast first when the
        // stream's dtype differs from the attention's, then shuffle.
        Tensor src = x.DType == act ? x : Cast(backend, x, act);
        Tensor o = new(new TensorShape(1, _heads, s, _headDim), act);
        backend.Permute0213(o, src, s, _heads, _headDim);
        if (!ReferenceEquals(src, x)) src.Dispose();
        return o;
    }

    /// <summary>Widens an F32 cross-attention K/V stream to the query's F16. Only this direction occurs: the block
    /// streams share one dtype, and the only mismatch is an F32 text connector feeding an F16 block.</summary>
    private static Tensor Cast(IBackend backend, Tensor x, DType target)
    {
        if (target != DType.F16 || x.DType != DType.F32)
            throw new NotSupportedException($"LTX-2 attention only widens an F32 K/V stream to F16, got {x.DType} to {target}.");
        Tensor o = new(x.Shape, target);
        backend.CastToF16(o, x);
        return o;
    }

    // [1, heads, s, headDim] → [s, inner] (inverse of ToBhsd), GPU-resident via Permute0213.
    private Tensor FromBhsd(IBackend backend, Tensor x, int s, DType act)
    {
        Tensor o = new(new TensorShape(s, _inner), act);
        backend.Permute0213(o, x, _heads, s, _headDim);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
