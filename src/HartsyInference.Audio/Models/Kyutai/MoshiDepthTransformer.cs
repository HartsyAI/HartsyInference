using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Moshi/Kyutai depth transformer ("depformer"): given one temporal-frame context vector it
/// autoregressively predicts the <see cref="MoshiDepthConfig.DepQ"/> Mimi codebooks. It is RoPE-free (the
/// codebook index is the position, captured by per-step weights), uses a per-codebook input projection
/// (multi-linear) over low-rank token embeddings, runs a small per-step-weighted transformer stack, and
/// projects to per-codebook logits.
///
/// <para>Structural scaffold (validation-pending): the data flow, shapes, and per-step weight-set selection
/// are exact; the precise checkpoint key layout for the per-step weight sets, low-rank embedding
/// factorization, and multi-linear projections is reconciled on first weight load. Attention reuses
/// <see cref="IBackend.ScaledDotProductAttention"/>; projections reuse <see cref="WhisperOps.ProjectLinear"/>.</para></summary>
public sealed unsafe class MoshiDepthTransformer : IDisposable
{
    private readonly MoshiDepthConfig _cfg;

    // Per-codebook-step (DepQ): low-rank embedding A [vocab, lowRank], input projection [dim, lowRank|mainDim],
    // and output head [vocab, dim].
    private readonly Tensor?[] _embedLowRank;     // [DepQ] each [CodebookVocab, LowRankEmb]
    private readonly Tensor?[] _inProj;           // [DepQ] each [Dim, in] (in = MainDim for k=0, LowRankEmb else)
    private readonly Tensor?[] _outHead;          // [DepQ] each [CodebookVocab, Dim]

    // Per-weight-set (NumWeightSets) × per-layer (NumLayers) transformer block weights.
    private readonly Tensor?[,] _qW, _kW, _vW, _oW, _gateW, _upW, _downW, _inNorm, _postNorm;
    private int _disposed;

    public MoshiDepthConfig Config => _cfg;

    public MoshiDepthTransformer(MoshiDepthConfig cfg)
    {
        _cfg = cfg;
        _embedLowRank = new Tensor?[cfg.DepQ];
        _inProj = new Tensor?[cfg.DepQ];
        _outHead = new Tensor?[cfg.DepQ];
        int sets = cfg.NumWeightSets, layers = cfg.NumLayers;
        _qW = new Tensor?[sets, layers]; _kW = new Tensor?[sets, layers]; _vW = new Tensor?[sets, layers];
        _oW = new Tensor?[sets, layers]; _gateW = new Tensor?[sets, layers]; _upW = new Tensor?[sets, layers];
        _downW = new Tensor?[sets, layers]; _inNorm = new Tensor?[sets, layers]; _postNorm = new Tensor?[sets, layers];
    }

    /// <summary>Loads the depformer weights. Keys follow the moshi <c>depformer.*</c> layout; the exact
    /// per-step-weight / low-rank / multi-linear key spelling is the checkpoint-gated reconcile item.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "depformer")
    {
        for (int k = 0; k < _cfg.DepQ; k++)
        {
            _embedLowRank[k] = WhisperOps.EnsureF32(w[$"{prefix}.emb.{k}.weight"]);
            _inProj[k] = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.{k}.weight"]);
            _outHead[k] = WhisperOps.EnsureF32(w[$"{prefix}.linears.{k}.weight"]);
        }
        for (int s = 0; s < _cfg.NumWeightSets; s++)
            for (int l = 0; l < _cfg.NumLayers; l++)
            {
                string p = $"{prefix}.layers.{s}.{l}";
                _qW[s, l] = WhisperOps.EnsureF32(w[$"{p}.self_attn.q_proj.weight"]);
                _kW[s, l] = WhisperOps.EnsureF32(w[$"{p}.self_attn.k_proj.weight"]);
                _vW[s, l] = WhisperOps.EnsureF32(w[$"{p}.self_attn.v_proj.weight"]);
                _oW[s, l] = WhisperOps.EnsureF32(w[$"{p}.self_attn.o_proj.weight"]);
                _gateW[s, l] = WhisperOps.EnsureF32(w[$"{p}.mlp.gate_proj.weight"]);
                _upW[s, l] = WhisperOps.EnsureF32(w[$"{p}.mlp.up_proj.weight"]);
                _downW[s, l] = WhisperOps.EnsureF32(w[$"{p}.mlp.down_proj.weight"]);
                _inNorm[s, l] = WhisperOps.EnsureF32(w[$"{p}.input_layernorm.weight"]);
                _postNorm[s, l] = WhisperOps.EnsureF32(w[$"{p}.post_attention_layernorm.weight"]);
            }
    }

    /// <summary>Autoregressively predicts the DepQ codebook tokens for a single temporal frame.
    /// <paramref name="context"/> is the temporal hidden state <c>[1,1,MainDim]</c>.</summary>
    public int[] GenerateFrame(IBackend backend, Tensor context, float temperature, int topK, float topP, ref uint rng)
    {
        int dim = _cfg.Dim, heads = _cfg.NumHeads, hd = _cfg.HeadDim, depQ = _cfg.DepQ;
        int[] codes = new int[depQ];

        // Depth KV cache (grown per codebook step): [heads, depQ, hd] for K and V.
        Tensor kCache = new(new TensorShape(1, heads, depQ, hd), DType.F32);
        Tensor vCache = new(new TensorShape(1, heads, depQ, hd), DType.F32);
        try
        {
            int prevToken = 0;
            for (int k = 0; k < depQ; k++)
            {
                Tensor x = StepInput(backend, context, prevToken, k);     // [1,1,dim]
                int set = _cfg.WeightSet(k);
                for (int l = 0; l < _cfg.NumLayers; l++)
                    x = Block(backend, x, set, l, k, kCache, vCache);

                Tensor logits = WhisperOps.ProjectLinear(backend, x, _outHead[k]!, null, 1, 1, dim, _cfg.CodebookVocab);
                x.Dispose();
                int tok = NucleusSampler.Draw(new Span<float>((void*)logits.DataPointer, _cfg.CodebookVocab),
                    _cfg.CodebookVocab, temperature, topK, topP, ref rng);
                logits.Dispose();
                codes[k] = tok;
                prevToken = tok;
            }
            return codes;
        }
        finally
        {
            kCache.Dispose();
            vCache.Dispose();
        }
    }

    /// <summary>Step input: codebook 0 projects the temporal context; later steps project the previous
    /// codebook's low-rank embedding. Result is <c>[1,1,dim]</c>.</summary>
    private Tensor StepInput(IBackend backend, Tensor context, int prevToken, int k)
    {
        if (k == 0)
            return WhisperOps.ProjectLinear(backend, context, _inProj[0]!, null, 1, 1, _cfg.MainDim, _cfg.Dim);

        // Low-rank embedding lookup → [1,1,lowRank] → in_proj → [1,1,dim].
        Tensor emb = new(new TensorShape(1, 1, _cfg.LowRankEmb), DType.F32);
        float* ep = (float*)_embedLowRank[k]!.DataPointer + (long)prevToken * _cfg.LowRankEmb;
        Buffer.MemoryCopy(ep, (void*)emb.DataPointer, _cfg.LowRankEmb * 4, _cfg.LowRankEmb * 4);
        Tensor x = WhisperOps.ProjectLinear(backend, emb, _inProj[k]!, null, 1, 1, _cfg.LowRankEmb, _cfg.Dim);
        emb.Dispose();
        return x;
    }

    /// <summary>One depformer transformer block at codebook position <paramref name="k"/> using weight set
    /// <paramref name="set"/> — RMSNorm + no-RoPE causal attention over depth positions 0..k + SwiGLU.</summary>
    private Tensor Block(IBackend backend, Tensor hidden, int set, int layer, int k, Tensor kCache, Tensor vCache)
    {
        int dim = _cfg.Dim, heads = _cfg.NumHeads, hd = _cfg.HeadDim;
        TensorShape shape = new(1, 1, dim);

        Tensor preAttn = new(shape, DType.F32);
        backend.RmsNorm(preAttn, hidden, _inNorm[set, layer]!, _cfg.RmsNormEps);

        Tensor q = WhisperOps.ProjectLinear(backend, preAttn, _qW[set, layer]!, null, 1, 1, dim, dim);
        Tensor kc = WhisperOps.ProjectLinear(backend, preAttn, _kW[set, layer]!, null, 1, 1, dim, dim);
        Tensor vc = WhisperOps.ProjectLinear(backend, preAttn, _vW[set, layer]!, null, 1, 1, dim, dim);
        preAttn.Dispose();

        // Write this position's K/V into the depth cache and read the [0..k] prefix.
        WriteHeadStep(kCache, kc, k, heads, hd); kc.Dispose();
        WriteHeadStep(vCache, vc, k, heads, hd); vc.Dispose();
        Tensor qMh = new(new TensorShape(1, heads, 1, hd), DType.F32);
        FlatToHeads(qMh, q, heads, hd); q.Dispose();
        Tensor kPre = LeadingPrefix(kCache, heads, k + 1, hd);
        Tensor vPre = LeadingPrefix(vCache, heads, k + 1, hd);

        float scale = 1f / MathF.Sqrt(hd);
        Tensor attn = new(new TensorShape(1, heads, 1, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kPre, vPre, null, scale);
        qMh.Dispose(); kPre.Dispose(); vPre.Dispose();

        Tensor attnFlat = new(shape, DType.F32);
        HeadsToFlat(attnFlat, attn, heads, hd); attn.Dispose();
        Tensor attnOut = WhisperOps.ProjectLinear(backend, attnFlat, _oW[set, layer]!, null, 1, 1, dim, dim);
        attnFlat.Dispose();

        Tensor afterAttn = new(shape, DType.F32);
        backend.Add(afterAttn, hidden, attnOut);
        attnOut.Dispose();
        hidden.Dispose();

        Tensor preMlp = new(shape, DType.F32);
        backend.RmsNorm(preMlp, afterAttn, _postNorm[set, layer]!, _cfg.RmsNormEps);
        Tensor mlpOut = SwiGlu(backend, preMlp, set, layer);
        preMlp.Dispose();

        Tensor result = new(shape, DType.F32);
        backend.Add(result, afterAttn, mlpOut);
        afterAttn.Dispose(); mlpOut.Dispose();
        return result;
    }

    private Tensor SwiGlu(IBackend backend, Tensor x, int set, int layer)
    {
        int dim = _cfg.Dim, ffn = _cfg.FfnDim;
        Tensor gate = WhisperOps.ProjectLinear(backend, x, _gateW[set, layer]!, null, 1, 1, dim, ffn);
        Tensor up = WhisperOps.ProjectLinear(backend, x, _upW[set, layer]!, null, 1, 1, dim, ffn);
        backend.Silu(gate, gate);     // in-place, matching Qwen2Mlp
        Tensor act = new(new TensorShape(1, 1, ffn), DType.F32);
        backend.Mul(act, gate, up);
        gate.Dispose(); up.Dispose();
        Tensor outT = WhisperOps.ProjectLinear(backend, act, _downW[set, layer]!, null, 1, 1, ffn, dim);
        act.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in _embedLowRank) if (t is not null) yield return t;
        foreach (Tensor? t in _inProj) if (t is not null) yield return t;
        foreach (Tensor? t in _outHead) if (t is not null) yield return t;
        foreach (Tensor?[,] grp in new[] { _qW, _kW, _vW, _oW, _gateW, _upW, _downW, _inNorm, _postNorm })
            foreach (Tensor? t in grp) if (t is not null) yield return t;
    }

    // [1,1,heads*hd] → [1,heads,1,hd]
    private static void FlatToHeads(Tensor outT, Tensor input, int heads, int hd)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int h = 0; h < heads; h++)
            Buffer.MemoryCopy(ip + (long)h * hd, op + (long)h * hd, hd * 4, hd * 4);
    }

    // [1,heads,1,hd] → [1,1,heads*hd]
    private static void HeadsToFlat(Tensor outT, Tensor input, int heads, int hd)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int h = 0; h < heads; h++)
            Buffer.MemoryCopy(ip + (long)h * hd, op + (long)h * hd, hd * 4, hd * 4);
    }

    // Write a [1,1,heads*hd] projection into cache [1,heads,depQ,hd] at depth position `pos`.
    private void WriteHeadStep(Tensor cache, Tensor proj, int pos, int heads, int hd)
    {
        float* cp = (float*)cache.DataPointer;
        float* pp = (float*)proj.DataPointer;
        int depQ = _cfg.DepQ;
        for (int h = 0; h < heads; h++)
        {
            long dst = ((long)h * depQ + pos) * hd;
            Buffer.MemoryCopy(pp + (long)h * hd, cp + dst, hd * 4, hd * 4);
        }
    }

    // Copy the first `len` depth positions of cache [1,heads,depQ,hd] into a packed [1,heads,len,hd].
    private Tensor LeadingPrefix(Tensor cache, int heads, int len, int hd)
    {
        Tensor outT = new(new TensorShape(1, heads, len, hd), DType.F32);
        float* cp = (float*)cache.DataPointer;
        float* op = (float*)outT.DataPointer;
        int depQ = _cfg.DepQ;
        for (int h = 0; h < heads; h++)
        {
            long src = (long)h * depQ * hd;
            long dst = (long)h * len * hd;
            Buffer.MemoryCopy(cp + src, op + dst, (long)len * hd * 4, (long)len * hd * 4);
        }
        return outT;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
