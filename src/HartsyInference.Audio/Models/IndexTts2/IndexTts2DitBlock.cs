using HartsyInference.Audio.Models.DiT;
using HartsyInference.Audio.Models.Moonshine;   // RotaryEmbedding (interleaved convention)
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>One block of IndexTTS-2's S2Mel DiT — a LLaMA/"gpt-fast"-style block (RoPE attention + SwiGLU FFN,
/// <see cref="AdaptiveLayerNorm"/> in place of plain RMSNorm, non-causal/bidirectional) with an optional U-ViT
/// skip-connection input. Real source (<c>gpt_fast/model.py:TransformerBlock</c>):
/// <code>
/// def forward(self, x, c, ..., skip_in_x=None):
///     if uvit_skip_connection and skip_in_x is not None:
///         x = self.skip_in_linear(cat([x, skip_in_x], dim=-1))
///     h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask, input_pos)
///     out = h + self.feed_forward(self.ffn_norm(h, c))
///     return out
/// </code>
/// Attention is fused QKV (one <c>wqkv</c> weight, no bias anywhere) — confirmed from the real checkpoint's
/// own <c>attention.wqkv.weight [3*hidden, hidden]</c>/<c>attention.wo.weight [hidden, hidden]</c>, no
/// <c>.bias</c> keys. <c>skip_in_linear</c> is present on EVERY block in the real checkpoint (the module is
/// unconditionally constructed) but is only ever invoked on the stack's receiving half — see
/// <see cref="IndexTts2Dit"/>.</summary>
internal sealed unsafe class IndexTts2DitBlock
{
    private readonly int _hidden, _heads, _headDim, _ffnDim;

    private readonly AdaptiveLayerNorm _attentionNorm, _ffnNorm;
    private Tensor? _wq, _wk, _wv;   // views into the fused wqkv weight — no copy.
    private Tensor? _wqkvOwned;      // the owned fused tensor the views above slice.
    private Tensor? _wo;
    private Tensor? _w1, _w2, _w3;
    private Tensor? _skipInW, _skipInB;

    public IndexTts2DitBlock(int hidden, int heads, int ffnDim)
    {
        _hidden = hidden;
        _heads = heads;
        _headDim = hidden / heads;
        _ffnDim = ffnDim;
        _attentionNorm = new AdaptiveLayerNorm(hidden);
        _ffnNorm = new AdaptiveLayerNorm(hidden);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _attentionNorm.LoadWeights(
            WhisperOps.EnsureF32(w[$"{prefix}.attention_norm.norm.weight"]),
            WhisperOps.EnsureF32(w[$"{prefix}.attention_norm.project_layer.weight"]),
            WhisperOps.EnsureF32(w[$"{prefix}.attention_norm.project_layer.bias"]));
        _ffnNorm.LoadWeights(
            WhisperOps.EnsureF32(w[$"{prefix}.ffn_norm.norm.weight"]),
            WhisperOps.EnsureF32(w[$"{prefix}.ffn_norm.project_layer.weight"]),
            WhisperOps.EnsureF32(w[$"{prefix}.ffn_norm.project_layer.bias"]));

        _wqkvOwned = WhisperOps.EnsureF32(w[$"{prefix}.attention.wqkv.weight"]);
        _wq = _wqkvOwned.SliceRows(0, _hidden);
        _wk = _wqkvOwned.SliceRows(_hidden, _hidden);
        _wv = _wqkvOwned.SliceRows(2 * _hidden, _hidden);
        _wo = WhisperOps.EnsureF32(w[$"{prefix}.attention.wo.weight"]);

        _w1 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w1.weight"]);
        _w2 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w2.weight"]);
        _w3 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w3.weight"]);

        if (w.TryGetValue($"{prefix}.skip_in_linear.weight", out Tensor? skipW))
        {
            _skipInW = WhisperOps.EnsureF32(skipW);
            _skipInB = WhisperOps.EnsureF32(w[$"{prefix}.skip_in_linear.bias"]);
        }
    }

    /// <summary><paramref name="x"/> is <c>[1, T, hidden]</c>; <paramref name="embedding"/> is the shared
    /// <c>[1, hidden]</c> timestep embedding; <paramref name="skipInX"/>, when given, is an earlier (emitting)
    /// block's output consumed via <c>skip_in_linear</c> before this block's own attention. Returns a new
    /// <c>[1, T, hidden]</c> tensor — the caller decides whether to keep it on the U-ViT emit stack.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor embedding, int t, Tensor ropeCos, Tensor ropeSin, Tensor? skipInX)
    {
        Tensor h = x;
        bool ownH = false;
        if (skipInX is not null && _skipInW is not null)
        {
            Tensor cat = ConcatLastDim(x, skipInX, t, _hidden, _hidden);
            h = WhisperOps.ProjectLinear(backend, cat, _skipInW, _skipInB, 1, t, 2 * _hidden, _hidden);
            cat.Dispose();
            ownH = true;
        }

        Tensor normed = _attentionNorm.Forward(backend, h, embedding);
        Tensor attnOut = Attention(backend, normed, t, ropeCos, ropeSin);
        normed.Dispose();

        Tensor afterAttn = AddLastDim(h, attnOut, t, _hidden);
        attnOut.Dispose();
        if (ownH) h.Dispose();

        Tensor normed2 = _ffnNorm.Forward(backend, afterAttn, embedding);
        Tensor ffnOut = Ffn(backend, normed2, t);
        normed2.Dispose();

        Tensor output = AddLastDim(afterAttn, ffnOut, t, _hidden);
        ffnOut.Dispose();
        afterAttn.Dispose();
        return output;
    }

    private Tensor Attention(IBackend backend, Tensor normed, int t, Tensor ropeCos, Tensor ropeSin)
    {
        TensorShape packed = new(1, t, _heads, _headDim);
        Tensor q = new(packed, DType.F32);
        Tensor k = new(packed, DType.F32);
        Tensor v = new(packed, DType.F32);
        backend.Linear(q, normed, _wq!, null);
        backend.Linear(k, normed, _wk!, null);
        backend.Linear(v, normed, _wv!, null);

        backend.WanRopeInterleaved(q, ropeCos, ropeSin, t, _heads, _headDim);
        backend.WanRopeInterleaved(k, ropeCos, ropeSin, t, _heads, _headDim);

        TensorShape mh = new(1, _heads, t, _headDim);
        Tensor qMh = new(mh, DType.F32), kMh = new(mh, DType.F32), vMh = new(mh, DType.F32);
        backend.Permute0213(qMh, q, t, _heads, _headDim);
        backend.Permute0213(kMh, k, t, _heads, _headDim);
        backend.Permute0213(vMh, v, t, _heads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        float scale = 1f / MathF.Sqrt(_headDim);
        Tensor attnOut = new(mh, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor merged = new(new TensorShape(1, t, _hidden), DType.F32);
        backend.Permute0213(merged, attnOut, _heads, t, _headDim);
        attnOut.Dispose();

        Tensor projected = WhisperOps.ProjectLinear(backend, merged, _wo!, null, 1, t, _hidden, _hidden);
        merged.Dispose();
        return projected;
    }

    private Tensor Ffn(IBackend backend, Tensor normed, int t)
    {
        Tensor gate = WhisperOps.ProjectLinear(backend, normed, _w1!, null, 1, t, _hidden, _ffnDim);
        Tensor silu = new(gate.Shape, DType.F32);
        backend.Silu(silu, gate);
        gate.Dispose();

        Tensor up = WhisperOps.ProjectLinear(backend, normed, _w3!, null, 1, t, _hidden, _ffnDim);
        float* sp = (float*)silu.DataPointer;
        float* up_ = (float*)up.DataPointer;
        long n = silu.ElementCount;
        for (long i = 0; i < n; i++) sp[i] *= up_[i];
        up.Dispose();

        Tensor down = WhisperOps.ProjectLinear(backend, silu, _w2!, null, 1, t, _ffnDim, _hidden);
        silu.Dispose();
        return down;
    }

    private static Tensor ConcatLastDim(Tensor a, Tensor b, int t, int aDim, int bDim)
    {
        Tensor result = new(new TensorShape(1, t, aDim + bDim), DType.F32);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float* rp = (float*)result.DataPointer;
        for (int i = 0; i < t; i++)
        {
            Buffer.MemoryCopy(ap + (long)i * aDim, rp + (long)i * (aDim + bDim), (long)aDim * 4, (long)aDim * 4);
            Buffer.MemoryCopy(bp + (long)i * bDim, rp + (long)i * (aDim + bDim) + aDim, (long)bDim * 4, (long)bDim * 4);
        }
        return result;
    }

    private static Tensor AddLastDim(Tensor a, Tensor b, int t, int dim)
    {
        Tensor result = new(a.Shape, DType.F32);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float* rp = (float*)result.DataPointer;
        long n = (long)t * dim;
        for (long i = 0; i < n; i++) rp[i] = ap[i] + bp[i];
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _attentionNorm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ffnNorm.EnumerateWeights()) yield return t;
        if (_wqkvOwned is not null) yield return _wqkvOwned;
        if (_wo is not null) yield return _wo;
        if (_w1 is not null) yield return _w1;
        if (_w2 is not null) yield return _w2;
        if (_w3 is not null) yield return _w3;
        if (_skipInW is not null) yield return _skipInW;
        if (_skipInB is not null) yield return _skipInB;
    }
}
