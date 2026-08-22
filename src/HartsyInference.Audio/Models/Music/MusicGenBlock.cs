using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>One MusicGen decoder layer (HF <c>MusicgenDecoderLayer</c> naming): pre-norm causal self-attention → pre-norm cross-attention against the T5 states → pre-norm GELU MLP.</summary>
/// <remarks>Bias-free attention projections (MusicGen default), bias on the MLP + LayerNorms. AR decoding is incremental:
/// one cache-capturing prefill <see cref="Forward"/> then O(T) per-token <see cref="ForwardStep"/> calls
/// against the <see cref="MusicGenKvCache"/> (cross K/V projected once via <see cref="PrimeCross"/>).
/// Reuses the WhisperOps attention helpers and the backend SDPA op.</remarks>
public sealed unsafe class MusicGenBlock : IDisposable
{
    private readonly MusicGenConfig _cfg;
    private int _disposed;

    private Tensor? _selfLnW, _selfLnB;
    private Tensor? _selfQW, _selfKW, _selfVW, _selfOW;
    private Tensor? _crossLnW, _crossLnB;
    private Tensor? _crossQW, _crossKW, _crossVW, _crossOW;
    private Tensor? _finalLnW, _finalLnB;
    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

    public MusicGenBlock(MusicGenConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // The q/k/v/o projections and the FFN fc1/fc2 are the bulk of the decoder's weight mass. They are consumed
        // ONLY via backend.Linear (WhisperOps.ProjectLinear), which casts a non-F32 weight to the GEMM dtype on the
        // GPU — so we keep them in their native checkpoint dtype (fp16) instead of upconverting to F32 here. That
        // halves their host-RAM footprint (the AudioGen load OOM). Norm gains/biases + the small FFN biases stay
        // F32: they're tiny, precision-sensitive, and backend.LayerNorm reads the F32 gains directly.
        _selfLnW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn_layer_norm.weight"]);
        _selfLnB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn_layer_norm.bias"]);
        _selfQW = w[$"{prefix}.self_attn.q_proj.weight"];
        _selfKW = w[$"{prefix}.self_attn.k_proj.weight"];
        _selfVW = w[$"{prefix}.self_attn.v_proj.weight"];
        _selfOW = w[$"{prefix}.self_attn.out_proj.weight"];
        _crossLnW = WhisperOps.EnsureF32(w[$"{prefix}.encoder_attn_layer_norm.weight"]);
        _crossLnB = WhisperOps.EnsureF32(w[$"{prefix}.encoder_attn_layer_norm.bias"]);
        _crossQW = w[$"{prefix}.encoder_attn.q_proj.weight"];
        _crossKW = w[$"{prefix}.encoder_attn.k_proj.weight"];
        _crossVW = w[$"{prefix}.encoder_attn.v_proj.weight"];
        _crossOW = w[$"{prefix}.encoder_attn.out_proj.weight"];
        _finalLnW = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.weight"]);
        _finalLnB = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.bias"]);
        // MusicGen FFN linears are bias-free (the HF checkpoint has no fc1.bias/fc2.bias); load if present.
        _fc1W = w[$"{prefix}.fc1.weight"];
        _fc1B = w.TryGetValue($"{prefix}.fc1.bias", out Tensor? fc1b) ? WhisperOps.EnsureF32(fc1b) : null;
        _fc2W = w[$"{prefix}.fc2.weight"];
        _fc2B = w.TryGetValue($"{prefix}.fc2.bias", out Tensor? fc2b) ? WhisperOps.EnsureF32(fc2b) : null;
    }

    /// <summary>Runs the layer over <paramref name="hidden"/> <c>[1, T, hidden]</c> against <paramref name="cross"/> <c>[1, T_text, hidden]</c>; pass <paramref name="cacheOut"/> to capture the self-attn K/V rows (prefill), enabling incremental continuation via <see cref="ForwardStep"/>.</summary>
    /// <param name="causalMask">The <c>[1,1,T,T]</c> additive causal mask; null when T==1.</param>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor cross, Tensor? causalMask,
        MusicGenKvCache.MusicGenKvLayer? cacheOut = null, int cacheOffset = 0)
    {
        int t = (int)hidden.Shape[1];
        int tt = (int)cross.Shape[1];
        int d = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int hd = _cfg.HeadDim;
        float scale = 1f / MathF.Sqrt(hd);
        TensorShape inShape = new(1, t, d);
        TensorShape mhT = new(1, nh, t, hd);

        // --- Self-attention ---
        Tensor normed = new(inShape, DType.F32);
        backend.LayerNorm(normed, hidden, _selfLnW!, _selfLnB!, 1e-5f);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _selfQW!, null, 1, t, d, d);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _selfKW!, null, 1, t, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _selfVW!, null, 1, t, d, d);
        normed.Dispose();
        Tensor qh = new(mhT, DType.F32), kh = new(mhT, DType.F32), vh = new(mhT, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qh, q, 1, t, nh, hd);
        WhisperOps.ReshapeToMultiHead4D(kh, k, 1, t, nh, hd);
        WhisperOps.ReshapeToMultiHead4D(vh, v, 1, t, nh, hd);
        q.Dispose(); k.Dispose(); v.Dispose();
        // Prefill capture: append the prompt's head-major self-attn K/V into the device cache (in place, no
        // host round-trip) so decoding can continue via ForwardStep.
        if (cacheOut is not null)
        {
            backend.KvCacheAppend(cacheOut.K, kh, cacheOffset);
            backend.KvCacheAppend(cacheOut.V, vh, cacheOffset);
        }
        Tensor attn = new(mhT, DType.F32);
        backend.ScaledDotProductAttention(attn, qh, kh, vh, causalMask, scale);
        qh.Dispose(); kh.Dispose(); vh.Dispose();
        Tensor merged = new(inShape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(merged, attn, 1, t, nh, hd);
        attn.Dispose();
        Tensor selfProj = WhisperOps.ProjectLinear(backend, merged, _selfOW!, null, 1, t, d, d);
        merged.Dispose();
        Tensor res1 = new(inShape, DType.F32);
        backend.Add(res1, hidden, selfProj);
        selfProj.Dispose();

        // --- Cross-attention to T5 ---
        Tensor normed2 = new(inShape, DType.F32);
        backend.LayerNorm(normed2, res1, _crossLnW!, _crossLnB!, 1e-5f);
        Tensor cq = WhisperOps.ProjectLinear(backend, normed2, _crossQW!, null, 1, t, d, d);
        normed2.Dispose();
        Tensor ck = WhisperOps.ProjectLinear(backend, cross, _crossKW!, null, 1, tt, d, d);
        Tensor cv = WhisperOps.ProjectLinear(backend, cross, _crossVW!, null, 1, tt, d, d);
        TensorShape mhTt = new(1, nh, tt, hd);
        Tensor cqh = new(mhT, DType.F32), ckh = new(mhTt, DType.F32), cvh = new(mhTt, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(cqh, cq, 1, t, nh, hd);
        WhisperOps.ReshapeToMultiHead4D(ckh, ck, 1, tt, nh, hd);
        WhisperOps.ReshapeToMultiHead4D(cvh, cv, 1, tt, nh, hd);
        cq.Dispose(); ck.Dispose(); cv.Dispose();
        Tensor cattn = new(mhT, DType.F32);
        backend.ScaledDotProductAttention(cattn, cqh, ckh, cvh, mask: null, scale);
        cqh.Dispose(); ckh.Dispose(); cvh.Dispose();
        Tensor cmerged = new(inShape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(cmerged, cattn, 1, t, nh, hd);
        cattn.Dispose();
        Tensor cproj = WhisperOps.ProjectLinear(backend, cmerged, _crossOW!, null, 1, t, d, d);
        cmerged.Dispose();
        Tensor res2 = new(inShape, DType.F32);
        backend.Add(res2, res1, cproj);
        res1.Dispose(); cproj.Dispose();

        // --- GELU MLP ---
        Tensor normed3 = new(inShape, DType.F32);
        backend.LayerNorm(normed3, res2, _finalLnW!, _finalLnB!, 1e-5f);
        Tensor fc1 = WhisperOps.ProjectLinear(backend, normed3, _fc1W!, _fc1B, 1, t, d, _cfg.FfnDim);
        normed3.Dispose();
        Tensor act = new(fc1.Shape, DType.F32);
        backend.Gelu(act, fc1);
        fc1.Dispose();
        Tensor fc2 = WhisperOps.ProjectLinear(backend, act, _fc2W!, _fc2B, 1, t, _cfg.FfnDim, d);
        act.Dispose();
        Tensor outT = new(inShape, DType.F32);
        backend.Add(outT, res2, fc2);
        res2.Dispose(); fc2.Dispose();
        return outT;
    }

    /// <summary>Projects the text states' cross-attn K/V into the cache once per generation — they depend only on the encoder output, so no per-step recompute.</summary>
    public void PrimeCross(IBackend backend, Tensor cross, MusicGenKvCache.MusicGenKvLayer cache)
    {
        int b = (int)cross.Shape[0];   // 1 (no CFG) or 2 (CFG: [0]=cond text, [1]=null/zeros)
        int tt = (int)cross.Shape[1];
        int d = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int hd = _cfg.HeadDim;
        Tensor ck = WhisperOps.ProjectLinear(backend, cross, _crossKW!, null, b, tt, d, d);
        Tensor cv = WhisperOps.ProjectLinear(backend, cross, _crossVW!, null, b, tt, d, d);
        Tensor ckh = new(new TensorShape(b, nh, tt, hd), DType.F32);
        Tensor cvh = new(new TensorShape(b, nh, tt, hd), DType.F32);
        backend.Permute0213(ckh, ck, tt, nh, hd);   // [b,tt,d] → head-major [b,nh,tt,hd]
        backend.Permute0213(cvh, cv, tt, nh, hd);
        ck.Dispose(); cv.Dispose();
        cache.CrossK?.Dispose(); cache.CrossV?.Dispose();
        cache.CrossK = ckh;
        cache.CrossV = cvh;
    }

    /// <summary>Single-token decode step against the cache, fully on-device. Input/output are <c>[1, 1, hidden]</c>.</summary>
    /// <remarks>Projections run on the backend; the 1×L self-attention appends the new K/V into the device cache at <paramref name="pos"/>
    /// (<see cref="IBackend.KvCacheAppend"/>, no host copy) then runs <see cref="IBackend.FlashAttention"/> over
    /// rows <c>[0, pos]</c> (causal, qOffset=pos — no explicit mask); the 1×T_text cross-attention runs
    /// FlashAttention over the pre-projected device text K/V (rows <c>[0, crossLen)</c>, non-causal). Numerics
    /// match the previous CPU path (same 1/√headDim scale, same causal prefix).</remarks>
    public Tensor ForwardStep(IBackend backend, Tensor x, MusicGenKvCache.MusicGenKvLayer cache, int pos, int crossLen, ulong devicePos = 0)
    {
        int b = (int)x.Shape[0];   // 1 (no CFG) or 2 (CFG cond+uncond decoded together)
        int d = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int hd = _cfg.HeadDim;
        float scale = 1f / MathF.Sqrt(hd);
        TensorShape one = new(b, 1, d);
        TensorShape mh1 = new(b, nh, 1, hd);

        // --- Self-attention (incremental, device-resident KV) ---
        Tensor normed = new(one, DType.F32);
        backend.LayerNorm(normed, x, _selfLnW!, _selfLnB!, 1e-5f);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _selfQW!, null, b, 1, d, d);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _selfKW!, null, b, 1, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _selfVW!, null, b, 1, d, d);
        normed.Dispose();
        Tensor qh = new(mh1, DType.F32), kh = new(mh1, DType.F32), vh = new(mh1, DType.F32);
        backend.Permute0213(qh, q, 1, nh, hd);   // [b,1,d] → head-major [b,nh,1,hd]
        backend.Permute0213(kh, k, 1, nh, hd);
        backend.Permute0213(vh, v, 1, nh, hd);
        q.Dispose(); k.Dispose(); v.Dispose();
        backend.KvCacheAppendDev(cache.K, kh, pos, devicePos);   // in-place device append at row `pos` (device or host)
        backend.KvCacheAppendDev(cache.V, vh, pos, devicePos);
        kh.Dispose(); vh.Dispose();
        Tensor attn = new(mh1, DType.F32);
        backend.FlashAttentionDev(attn, qh, cache.K, cache.V, kvLen: pos + 1, kvGroup: 1, causal: true, qOffset: pos, scale, devicePos);
        qh.Dispose();
        Tensor attnFlat = new(one, DType.F32);
        backend.Permute0213(attnFlat, attn, nh, 1, hd);   // [b,nh,1,hd] → [b,1,d]
        attn.Dispose();
        Tensor selfProj = WhisperOps.ProjectLinear(backend, attnFlat, _selfOW!, null, b, 1, d, d);
        attnFlat.Dispose();
        Tensor res1 = new(one, DType.F32);
        backend.Add(res1, x, selfProj);
        selfProj.Dispose();

        // --- Cross-attention (device-resident pre-projected text K/V; each batch elem attends its own K/V) ---
        Tensor normed2 = new(one, DType.F32);
        backend.LayerNorm(normed2, res1, _crossLnW!, _crossLnB!, 1e-5f);
        Tensor cq = WhisperOps.ProjectLinear(backend, normed2, _crossQW!, null, b, 1, d, d);
        normed2.Dispose();
        Tensor cqh = new(mh1, DType.F32);
        backend.Permute0213(cqh, cq, 1, nh, hd);
        cq.Dispose();
        Tensor cattn = new(mh1, DType.F32);
        backend.FlashAttention(cattn, cqh, cache.CrossK!, cache.CrossV!, kvLen: crossLen, kvGroup: 1, causal: false, qOffset: 0, scale);
        cqh.Dispose();
        Tensor cattnFlat = new(one, DType.F32);
        backend.Permute0213(cattnFlat, cattn, nh, 1, hd);
        cattn.Dispose();
        Tensor cproj = WhisperOps.ProjectLinear(backend, cattnFlat, _crossOW!, null, b, 1, d, d);
        cattnFlat.Dispose();
        Tensor res2 = new(one, DType.F32);
        backend.Add(res2, res1, cproj);
        res1.Dispose(); cproj.Dispose();

        // --- GELU MLP ---
        Tensor normed3 = new(one, DType.F32);
        backend.LayerNorm(normed3, res2, _finalLnW!, _finalLnB!, 1e-5f);
        Tensor fc1 = WhisperOps.ProjectLinear(backend, normed3, _fc1W!, _fc1B, b, 1, d, _cfg.FfnDim);
        normed3.Dispose();
        Tensor act = new(fc1.Shape, DType.F32);
        backend.Gelu(act, fc1);
        fc1.Dispose();
        Tensor fc2 = WhisperOps.ProjectLinear(backend, act, _fc2W!, _fc2B, b, 1, _cfg.FfnDim, d);
        act.Dispose();
        Tensor outT = new(one, DType.F32);
        backend.Add(outT, res2, fc2);
        res2.Dispose(); fc2.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_selfLnW, _selfLnB, _selfQW, _selfKW, _selfVW, _selfOW,
            _crossLnW, _crossLnB, _crossQW, _crossKW, _crossVW, _crossOW,
            _finalLnW, _finalLnB, _fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
