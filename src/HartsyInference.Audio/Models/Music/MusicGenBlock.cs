using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>One MusicGen decoder layer (HF <c>MusicgenDecoderLayer</c> naming): pre-norm causal
/// self-attention → pre-norm cross-attention against the T5 states → pre-norm GELU MLP. Bias-free
/// attention projections (MusicGen default), bias on the MLP + LayerNorms. AR decoding is incremental:
/// one cache-capturing prefill <see cref="Forward"/> then O(T) per-token <see cref="ForwardStep"/> calls
/// against the <see cref="MusicGenKvCache"/> (cross K/V projected once via <see cref="PrimeCross"/>).
/// Reuses the WhisperOps attention helpers and the backend SDPA op.</summary>
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
        _selfLnW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn_layer_norm.weight"]);
        _selfLnB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn_layer_norm.bias"]);
        _selfQW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.q_proj.weight"]);
        _selfKW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.k_proj.weight"]);
        _selfVW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.v_proj.weight"]);
        _selfOW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.out_proj.weight"]);
        _crossLnW = WhisperOps.EnsureF32(w[$"{prefix}.encoder_attn_layer_norm.weight"]);
        _crossLnB = WhisperOps.EnsureF32(w[$"{prefix}.encoder_attn_layer_norm.bias"]);
        _crossQW = WhisperOps.EnsureF32(w[$"{prefix}.encoder_attn.q_proj.weight"]);
        _crossKW = WhisperOps.EnsureF32(w[$"{prefix}.encoder_attn.k_proj.weight"]);
        _crossVW = WhisperOps.EnsureF32(w[$"{prefix}.encoder_attn.v_proj.weight"]);
        _crossOW = WhisperOps.EnsureF32(w[$"{prefix}.encoder_attn.out_proj.weight"]);
        _finalLnW = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.weight"]);
        _finalLnB = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.bias"]);
        // MusicGen FFN linears are bias-free (the HF checkpoint has no fc1.bias/fc2.bias); load if present.
        _fc1W = WhisperOps.EnsureF32(w[$"{prefix}.fc1.weight"]);
        _fc1B = w.TryGetValue($"{prefix}.fc1.bias", out Tensor? fc1b) ? WhisperOps.EnsureF32(fc1b) : null;
        _fc2W = WhisperOps.EnsureF32(w[$"{prefix}.fc2.weight"]);
        _fc2B = w.TryGetValue($"{prefix}.fc2.bias", out Tensor? fc2b) ? WhisperOps.EnsureF32(fc2b) : null;
    }

    /// <summary>Runs the layer. <paramref name="hidden"/> is <c>[1, T, hidden]</c>; <paramref name="cross"/>
    /// is the projected T5 states <c>[1, T_text, hidden]</c>; <paramref name="causalMask"/> is the
    /// <c>[1,1,T,T]</c> additive mask (null for T==1). Pass <paramref name="cacheOut"/> to capture the
    /// self-attn K/V rows (prefill), enabling incremental continuation via <see cref="ForwardStep"/>.</summary>
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
        // Prefill capture: stash the prompt's self-attn K/V rows so decoding can continue via ForwardStep.
        if (cacheOut is not null)
        {
            new ReadOnlySpan<float>((void*)k.DataPointer, t * d).CopyTo(cacheOut.K.AsSpan(cacheOffset * d));
            new ReadOnlySpan<float>((void*)v.DataPointer, t * d).CopyTo(cacheOut.V.AsSpan(cacheOffset * d));
        }
        Tensor qh = new(mhT, DType.F32), kh = new(mhT, DType.F32), vh = new(mhT, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qh, q, 1, t, nh, hd);
        WhisperOps.ReshapeToMultiHead4D(kh, k, 1, t, nh, hd);
        WhisperOps.ReshapeToMultiHead4D(vh, v, 1, t, nh, hd);
        q.Dispose(); k.Dispose(); v.Dispose();
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

    /// <summary>Projects the text states' cross-attn K/V into the cache once per generation — they depend
    /// only on the encoder output, so no per-step recompute.</summary>
    public void PrimeCross(IBackend backend, Tensor cross, MusicGenKvCache.MusicGenKvLayer cache)
    {
        int tt = (int)cross.Shape[1];
        int d = _cfg.Hidden;
        Tensor ck = WhisperOps.ProjectLinear(backend, cross, _crossKW!, null, 1, tt, d, d);
        Tensor cv = WhisperOps.ProjectLinear(backend, cross, _crossVW!, null, 1, tt, d, d);
        cache.CrossK = new float[tt * d];
        cache.CrossV = new float[tt * d];
        new ReadOnlySpan<float>((void*)ck.DataPointer, tt * d).CopyTo(cache.CrossK);
        new ReadOnlySpan<float>((void*)cv.DataPointer, tt * d).CopyTo(cache.CrossV);
        ck.Dispose(); cv.Dispose();
    }

    /// <summary>Single-token decode step against the cache: projections run on the backend (weights stay
    /// device-resident); the 1×L self-attention runs on CPU over the cached prefix (new K/V appended at
    /// <paramref name="pos"/>, attention covers rows <c>[0, pos]</c> — causal by construction, no mask); the
    /// 1×T_text cross-attention runs on CPU over the pre-projected text K/V (rows <c>[0, crossLen)</c>).
    /// Input and output are <c>[1, 1, hidden]</c>.</summary>
    public Tensor ForwardStep(IBackend backend, Tensor x, MusicGenKvCache.MusicGenKvLayer cache, int pos, int crossLen)
    {
        int d = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int hd = _cfg.HeadDim;
        float scale = 1f / MathF.Sqrt(hd);
        TensorShape one = new(1, 1, d);

        // --- Self-attention (incremental) ---
        Tensor normed = new(one, DType.F32);
        backend.LayerNorm(normed, x, _selfLnW!, _selfLnB!, 1e-5f);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _selfQW!, null, 1, 1, d, d);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _selfKW!, null, 1, 1, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _selfVW!, null, 1, 1, d, d);
        normed.Dispose();
        new ReadOnlySpan<float>((void*)k.DataPointer, d).CopyTo(cache.K.AsSpan(pos * d));
        new ReadOnlySpan<float>((void*)v.DataPointer, d).CopyTo(cache.V.AsSpan(pos * d));
        k.Dispose(); v.Dispose();
        Tensor attnFlat = new(one, DType.F32);
        AttendCpu((float*)q.DataPointer, cache.K, cache.V, pos + 1, (float*)attnFlat.DataPointer, nh, hd, scale);
        q.Dispose();
        Tensor selfProj = WhisperOps.ProjectLinear(backend, attnFlat, _selfOW!, null, 1, 1, d, d);
        attnFlat.Dispose();
        Tensor res1 = new(one, DType.F32);
        backend.Add(res1, x, selfProj);
        selfProj.Dispose();

        // --- Cross-attention (pre-projected text K/V) ---
        Tensor normed2 = new(one, DType.F32);
        backend.LayerNorm(normed2, res1, _crossLnW!, _crossLnB!, 1e-5f);
        Tensor cq = WhisperOps.ProjectLinear(backend, normed2, _crossQW!, null, 1, 1, d, d);
        normed2.Dispose();
        Tensor cattnFlat = new(one, DType.F32);
        AttendCpu((float*)cq.DataPointer, cache.CrossK, cache.CrossV, crossLen, (float*)cattnFlat.DataPointer, nh, hd, scale);
        cq.Dispose();
        Tensor cproj = WhisperOps.ProjectLinear(backend, cattnFlat, _crossOW!, null, 1, 1, d, d);
        cattnFlat.Dispose();
        Tensor res2 = new(one, DType.F32);
        backend.Add(res2, res1, cproj);
        res1.Dispose(); cproj.Dispose();

        // --- GELU MLP ---
        Tensor normed3 = new(one, DType.F32);
        backend.LayerNorm(normed3, res2, _finalLnW!, _finalLnB!, 1e-5f);
        Tensor fc1 = WhisperOps.ProjectLinear(backend, normed3, _fc1W!, _fc1B, 1, 1, d, _cfg.FfnDim);
        normed3.Dispose();
        Tensor act = new(fc1.Shape, DType.F32);
        backend.Gelu(act, fc1);
        fc1.Dispose();
        Tensor fc2 = WhisperOps.ProjectLinear(backend, act, _fc2W!, _fc2B, 1, 1, _cfg.FfnDim, d);
        act.Dispose();
        Tensor outT = new(one, DType.F32);
        backend.Add(outT, res2, fc2);
        res2.Dispose(); fc2.Dispose();
        return outT;
    }

    /// <summary>Single-query multi-head attention on CPU: <paramref name="q"/> <c>[hidden]</c> against
    /// token-major K/V rows <c>[len, hidden]</c>, softmax per head, result into <paramref name="outP"/>.</summary>
    private static void AttendCpu(float* q, float[] kRows, float[] vRows, int len, float* outP, int nh, int hd, float scale)
    {
        int h = nh * hd;
        Span<float> scores = len <= 4096 ? stackalloc float[len] : new float[len];
        for (int head = 0; head < nh; head++)
        {
            int hOff = head * hd;
            float* qh = q + hOff;
            // scores = q · K_head over the rows, softmax in place
            float max = float.NegativeInfinity;
            for (int l = 0; l < len; l++)
            {
                float dot = 0f;
                int rowOff = l * h + hOff;
                for (int j = 0; j < hd; j++) dot += qh[j] * kRows[rowOff + j];
                dot *= scale;
                scores[l] = dot;
                if (dot > max) max = dot;
            }
            float sum = 0f;
            for (int l = 0; l < len; l++)
            {
                float e = MathF.Exp(scores[l] - max);
                scores[l] = e;
                sum += e;
            }
            float inv = 1f / sum;
            float* oh = outP + hOff;
            for (int j = 0; j < hd; j++) oh[j] = 0f;
            for (int l = 0; l < len; l++)
            {
                float w = scores[l] * inv;
                int rowOff = l * h + hOff;
                for (int j = 0; j < hd; j++) oh[j] += w * vRows[rowOff + j];
            }
        }
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
