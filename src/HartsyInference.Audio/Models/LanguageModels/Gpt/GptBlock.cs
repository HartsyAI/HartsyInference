using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.LanguageModels.Gpt;

/// <summary>One GPT-2 pre-norm block: <c>x + Attn(LN1(x))</c> then <c>x + MLP(LN2(x))</c>, fused QKV projection, all linears <c>bias=False</c>; checkpoint key scheme follows HF Bark (<c>layernorm_1</c>, <c>attn.att_proj</c>, <c>attn.out_proj</c>, <c>layernorm_2</c>, <c>mlp.in_proj</c>, <c>mlp.out_proj</c>).</summary>
public sealed unsafe class GptBlock : IDisposable
{
    private readonly GptConfig _cfg;
    private int _disposed;

    private Tensor? _ln1G, _ln1B, _ln2G, _ln2B;
    private Tensor? _attW, _outW;         // att_proj [3H, H], out_proj [H, H]
    private Tensor? _mlpInW, _mlpOutW;    // [4H, H], [H, 4H]

    public GptBlock(GptConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _ln1G = WhisperOps.EnsureF32(w[$"{prefix}.layernorm_1.weight"]);
        _ln1B = LoadBiasOrZero(w, $"{prefix}.layernorm_1.bias", _ln1G);
        _attW = WhisperOps.EnsureF32(w[$"{prefix}.attn.att_proj.weight"]);
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.attn.out_proj.weight"]);
        _ln2G = WhisperOps.EnsureF32(w[$"{prefix}.layernorm_2.weight"]);
        _ln2B = LoadBiasOrZero(w, $"{prefix}.layernorm_2.bias", _ln2G);
        _mlpInW = WhisperOps.EnsureF32(w[$"{prefix}.mlp.in_proj.weight"]);
        _mlpOutW = WhisperOps.EnsureF32(w[$"{prefix}.mlp.out_proj.weight"]);
    }

    /// <summary>Full-sequence forward with NO KV cache — used for the bidirectional Bark-Fine stage (<paramref name="causalMask"/> null) and parity-debug teacher forcing (causal mask); the incremental AR path goes through <see cref="ForwardCached"/> instead.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor? causalMask)
    {
        int t = (int)x.Shape[1];
        int h = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int d = _cfg.HeadDim;

        // ── Attention ──
        Tensor ln1 = new(x.Shape, DType.F32);
        backend.LayerNorm(ln1, x, _ln1G!, _ln1B!, 1e-5f);
        Tensor qkv = WhisperOps.ProjectLinear(backend, ln1, _attW!, bias: null, 1, t, h, 3 * h);
        ln1.Dispose();

        // Split the fused QKV on-device (last-dim slices), then permute [1,t,h] → [1,nh,t,d]. Keeping the
        // split/permute/residual on the backend (rather than host loops) avoids a device→host sync per layer —
        // the non-causal Bark-Fine stage runs six full 1024-frame passes, so this glue dominated otherwise.
        Tensor q = new(new TensorShape(1, t, h), DType.F32);
        Tensor k = new(new TensorShape(1, t, h), DType.F32);
        Tensor v = new(new TensorShape(1, t, h), DType.F32);
        backend.SliceLastDim(q, qkv, 0);
        backend.SliceLastDim(k, qkv, h);
        backend.SliceLastDim(v, qkv, 2 * h);
        qkv.Dispose();

        Tensor qMh = new(new TensorShape(1, nh, t, d), DType.F32);
        Tensor kMh = new(new TensorShape(1, nh, t, d), DType.F32);
        Tensor vMh = new(new TensorShape(1, nh, t, d), DType.F32);
        backend.Permute0213(qMh, q, t, nh, d);
        backend.Permute0213(kMh, k, t, nh, d);
        backend.Permute0213(vMh, v, t, nh, d);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn = new(new TensorShape(1, nh, t, d), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, causalMask, 1f / MathF.Sqrt(d));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, t, h), DType.F32);
        backend.Permute0213(attnFlat, attn, nh, t, d);
        attn.Dispose();
        Tensor attnOut = WhisperOps.ProjectLinear(backend, attnFlat, _outW!, bias: null, 1, t, h, h);
        attnFlat.Dispose();

        Tensor res1 = new(x.Shape, DType.F32);
        backend.Add(res1, x, attnOut);
        attnOut.Dispose();

        // ── MLP ──
        Tensor ln2 = new(res1.Shape, DType.F32);
        backend.LayerNorm(ln2, res1, _ln2G!, _ln2B!, 1e-5f);
        Tensor fc = WhisperOps.ProjectLinear(backend, ln2, _mlpInW!, bias: null, 1, t, h, _cfg.MlpDim);
        ln2.Dispose();
        backend.Gelu(fc, fc);
        Tensor proj = WhisperOps.ProjectLinear(backend, fc, _mlpOutW!, bias: null, 1, t, _cfg.MlpDim, h);
        fc.Dispose();

        Tensor res2 = new(res1.Shape, DType.F32);
        backend.Add(res2, res1, proj);
        res1.Dispose();
        proj.Dispose();
        return res2;
    }

    /// <summary>GPU-resident cached forward for prefill (<c>t≥1</c>) and incremental decode (<c>t=1</c>): everything after the QKV projection — split, head-permute, K/V cache append, causal FlashAttention over the valid prefix — stays device-side, so no <c>DataPointer</c> read crosses back to the host mid-forward.</summary>
    /// <remarks>Replaces an older per-step host attention loop that synced device→host every layer. <paramref name="qOffset"/> is the absolute position of this call's first query, equal to the cache's committed length; the backbone advances the shared length once, after all layers.</remarks>
    public Tensor ForwardCached(IBackend backend, Tensor x, IKvCache cache, int layerIndex, int qOffset)
    {
        int t = (int)x.Shape[1];
        int h = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int d = _cfg.HeadDim;

        // ── Attention ──
        Tensor ln1 = new(x.Shape, DType.F32);
        backend.LayerNorm(ln1, x, _ln1G!, _ln1B!, 1e-5f);
        Tensor qkv = WhisperOps.ProjectLinear(backend, ln1, _attW!, bias: null, 1, t, h, 3 * h);
        ln1.Dispose();

        // Split the fused QKV on-device (last-dim slices), then permute [1,t,h] → [1,nh,t,d].
        Tensor q = new(new TensorShape(1, t, h), DType.F32);
        Tensor k = new(new TensorShape(1, t, h), DType.F32);
        Tensor v = new(new TensorShape(1, t, h), DType.F32);
        backend.SliceLastDim(q, qkv, 0);
        backend.SliceLastDim(k, qkv, h);
        backend.SliceLastDim(v, qkv, 2 * h);
        qkv.Dispose();

        Tensor qMh = new(new TensorShape(1, nh, t, d), DType.F32);
        Tensor kMh = new(new TensorShape(1, nh, t, d), DType.F32);
        Tensor vMh = new(new TensorShape(1, nh, t, d), DType.F32);
        backend.Permute0213(qMh, q, t, nh, d);
        backend.Permute0213(kMh, k, t, nh, d);
        backend.Permute0213(vMh, v, t, nh, d);
        q.Dispose(); k.Dispose(); v.Dispose();

        cache.AppendStep(backend, layerIndex, kMh, vMh);
        kMh.Dispose(); vMh.Dispose();

        // FlashAttention over the valid prefix [0, qOffset+t): MHA (group 1), causal via the absolute query
        // offset. The cache buffer's seq stride exceeds the valid length, so the valid key count is passed
        // explicitly.
        Tensor attn = new(new TensorShape(1, nh, t, d), DType.F32);
        backend.FlashAttention(attn, qMh, cache.KeyPrefix(layerIndex), cache.ValuePrefix(layerIndex),
            qOffset + t, kvGroup: 1, causal: true, qOffset: qOffset, 1f / MathF.Sqrt(d));
        qMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, t, h), DType.F32);
        backend.Permute0213(attnFlat, attn, nh, t, d);
        attn.Dispose();
        Tensor attnOut = WhisperOps.ProjectLinear(backend, attnFlat, _outW!, bias: null, 1, t, h, h);
        attnFlat.Dispose();

        Tensor res1 = new(x.Shape, DType.F32);
        backend.Add(res1, x, attnOut);
        attnOut.Dispose();

        // ── MLP ──
        Tensor ln2 = new(res1.Shape, DType.F32);
        backend.LayerNorm(ln2, res1, _ln2G!, _ln2B!, 1e-5f);
        Tensor fc = WhisperOps.ProjectLinear(backend, ln2, _mlpInW!, bias: null, 1, t, h, _cfg.MlpDim);
        ln2.Dispose();
        backend.Gelu(fc, fc);
        Tensor proj = WhisperOps.ProjectLinear(backend, fc, _mlpOutW!, bias: null, 1, t, _cfg.MlpDim, h);
        fc.Dispose();

        Tensor res2 = new(res1.Shape, DType.F32);
        backend.Add(res2, res1, proj);
        res1.Dispose();
        proj.Dispose();
        return res2;
    }

    /// <summary>Returns a zero tensor (matching <paramref name="weightLike"/>'s shape) when the checkpoint has no bias — HF Bark's <c>BarkLayerNorm</c> uses <c>bias=False</c>.</summary>
    internal static Tensor LoadBiasOrZero(IReadOnlyDictionary<string, Tensor> w, string key, Tensor weightLike)
    {
        if (w.TryGetValue(key, out Tensor? b)) return WhisperOps.EnsureF32(b);
        Tensor zero = new(weightLike.Shape, DType.F32);
        float* p = (float*)zero.DataPointer;
        for (long i = 0; i < zero.ElementCount; i++) p[i] = 0f;
        return zero;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_ln1G, _ln1B, _attW, _outW, _ln2G, _ln2B, _mlpInW, _mlpOutW];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
