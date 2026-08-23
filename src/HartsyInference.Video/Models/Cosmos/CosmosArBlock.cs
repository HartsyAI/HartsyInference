using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Video.Models.Cosmos;

/// <summary>One Cosmos AR transformer layer: pre-norm GQA self-attention (3D-RoPE + optional QK-norm, KV-cached) →
/// optional per-layer non-causal T5 cross-attention → SwiGLU FFN, each a residual sub-layer. Mirrors the LLM
/// decoder attention idiom (project into head layout → QK-norm → RoPE → <see cref="IBackend.Permute0213"/> →
/// <see cref="IBackend.FlashAttention"/>) but adds the distinct V2W cross-attention over external T5 embeddings.</summary>
public sealed unsafe class CosmosArBlock
{
    private readonly CosmosArConfig _cfg;
    private readonly bool _hasCross;

    private Tensor _attnNorm = null!;
    private Tensor _wq = null!, _wk = null!, _wv = null!, _wo = null!;
    private Tensor? _qNorm, _kNorm;
    private Tensor? _crossNorm, _cwq, _cwk, _cwv, _cwo;
    private Tensor _ffnNorm = null!;
    private Tensor _w1 = null!, _w2 = null!, _w3 = null!;   // gate / down / up

    public CosmosArBlock(CosmosArConfig cfg, int layerIndex)
    {
        _cfg = cfg;
        _hasCross = cfg.HasCrossAttn(layerIndex);
    }

    /// <summary>Loads this layer's weights from a Cosmos-named state dict under <paramref name="prefix"/>
    /// (e.g. <c>layers.3</c>). Norms are forced F32; projections keep their loaded dtype for the matmul backend.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _attnNorm = EnsureF32(w[$"{prefix}.attention_norm.weight"]);
        _wq = w[$"{prefix}.attention.wq.weight"];
        _wk = w[$"{prefix}.attention.wk.weight"];
        _wv = w[$"{prefix}.attention.wv.weight"];
        _wo = w[$"{prefix}.attention.wo.weight"];
        if (_cfg.QkNorm)
        {
            _qNorm = EnsureF32(w[$"{prefix}.attention.q_norm.weight"]);
            _kNorm = EnsureF32(w[$"{prefix}.attention.k_norm.weight"]);
        }
        if (_hasCross)
        {
            _crossNorm = EnsureF32(w[$"{prefix}.cross_attention_norm.weight"]);
            _cwq = w[$"{prefix}.cross_attention.wq.weight"];
            _cwk = w[$"{prefix}.cross_attention.wk.weight"];
            _cwv = w[$"{prefix}.cross_attention.wv.weight"];
            _cwo = w[$"{prefix}.cross_attention.wo.weight"];
        }
        _ffnNorm = EnsureF32(w[$"{prefix}.ffn_norm.weight"]);
        _w1 = w[$"{prefix}.feed_forward.w1.weight"];
        _w2 = w[$"{prefix}.feed_forward.w2.weight"];
        _w3 = w[$"{prefix}.feed_forward.w3.weight"];
    }

    /// <summary>All weight tensors for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _attnNorm; yield return _wq; yield return _wk; yield return _wv; yield return _wo;
        if (_qNorm is not null) yield return _qNorm;
        if (_kNorm is not null) yield return _kNorm;
        if (_hasCross) { yield return _crossNorm!; yield return _cwq!; yield return _cwk!; yield return _cwv!; yield return _cwo!; }
        yield return _ffnNorm; yield return _w1; yield return _w2; yield return _w3;
    }

    /// <summary>Runs the layer over <paramref name="hidden"/> <c>[1, t, dim]</c>. <paramref name="cos"/>/
    /// <paramref name="sin"/> are the 3D-RoPE slice <c>[1, t, head_dim]</c> for this token range; <paramref name="context"/>
    /// is the T5 embedding <c>[1, ctxLen, context_dim]</c> (null on the base 4B path). Returns <c>[1, t, dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, int t, int posStart, IKvCache cache, int layerIndex,
        Tensor cos, Tensor sin, Tensor? context, int ctxLen)
    {
        int hq = _cfg.NumHeads, hkv = _cfg.NumKvHeads, d = _cfg.HeadDim, group = _cfg.KvGroup, dim = _cfg.Dim;
        TensorShape flat = new(1, t, dim);

        // --- Self-attention ---
        Tensor pre = new(flat, DType.F32);
        backend.RmsNorm(pre, hidden, _attnNorm, _cfg.NormEps);

        Tensor q = new(new TensorShape(1, t, hq, d), DType.F32);
        Tensor k = new(new TensorShape(1, t, hkv, d), DType.F32);
        Tensor v = new(new TensorShape(1, t, hkv, d), DType.F32);
        Project(backend, q, pre, _wq, null);
        Project(backend, k, pre, _wk, null);
        Project(backend, v, pre, _wv, null);
        pre.Dispose();

        if (_cfg.QkNorm)
        {
            Tensor qN = new(new TensorShape(1, t, hq, d), DType.F32);
            backend.RmsNorm(qN, q, _qNorm!, _cfg.NormEps);
            q.Dispose(); q = qN;
            Tensor kN = new(new TensorShape(1, t, hkv, d), DType.F32);
            backend.RmsNorm(kN, k, _kNorm!, _cfg.NormEps);
            k.Dispose(); k = kN;
        }

        backend.ApplyRopeSingle(q, cos, sin, d);
        backend.ApplyRopeSingle(k, cos, sin, d);

        Tensor qMh = new(new TensorShape(1, hq, t, d), DType.F32); backend.Permute0213(qMh, q, t, hq, d);
        Tensor kMh = new(new TensorShape(1, hkv, t, d), DType.F32); backend.Permute0213(kMh, k, t, hkv, d);
        Tensor vMh = new(new TensorShape(1, hkv, t, d), DType.F32); backend.Permute0213(vMh, v, t, hkv, d);
        q.Dispose(); k.Dispose(); v.Dispose();

        cache.AppendStep(backend, layerIndex, kMh, vMh);
        kMh.Dispose(); vMh.Dispose();
        Tensor kFull = cache.KeyPrefix(layerIndex);
        Tensor vFull = cache.ValuePrefix(layerIndex);

        int kvLen = posStart + t;
        Tensor attn = new(new TensorShape(1, hq, t, d), DType.F32);
        backend.FlashAttention(attn, qMh, kFull, vFull, kvLen, group, causal: true, qOffset: posStart, 1f / MathF.Sqrt(d));
        qMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, t, hq * d), DType.F32);
        backend.Permute0213(attnFlat, attn, hq, t, d);
        attn.Dispose();
        Tensor attnOut = new(flat, DType.F32);
        Project(backend, attnOut, attnFlat, _wo, null);
        attnFlat.Dispose();

        Tensor h = new(flat, DType.F32);
        backend.Add(h, hidden, attnOut);
        attnOut.Dispose();

        // --- T5 cross-attention (non-causal, no RoPE) ---
        if (_hasCross && context is not null)
        {
            Tensor cout = CrossAttention(backend, h, t, context, ctxLen);
            Tensor h2 = new(flat, DType.F32);
            backend.Add(h2, h, cout);
            h.Dispose(); cout.Dispose();
            h = h2;
        }

        // --- SwiGLU FFN ---
        Tensor fpre = new(flat, DType.F32);
        backend.RmsNorm(fpre, h, _ffnNorm, _cfg.NormEps);
        Tensor ffnOut = SwiGlu(backend, fpre, t);
        fpre.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, h, ffnOut);
        h.Dispose(); ffnOut.Dispose();
        return result;
    }

    private Tensor CrossAttention(IBackend backend, Tensor h, int t, Tensor context, int ctxLen)
    {
        int hq = _cfg.NumHeads, hkv = _cfg.NumKvHeads, d = _cfg.HeadDim, group = _cfg.KvGroup, dim = _cfg.Dim;
        TensorShape flat = new(1, t, dim);

        Tensor pre = new(flat, DType.F32);
        backend.RmsNorm(pre, h, _crossNorm!, _cfg.NormEps);

        Tensor q = new(new TensorShape(1, t, hq, d), DType.F32);
        Project(backend, q, pre, _cwq!, null);
        pre.Dispose();
        Tensor k = new(new TensorShape(1, ctxLen, hkv, d), DType.F32);
        Project(backend, k, context, _cwk!, null);
        Tensor v = new(new TensorShape(1, ctxLen, hkv, d), DType.F32);
        Project(backend, v, context, _cwv!, null);

        Tensor qMh = new(new TensorShape(1, hq, t, d), DType.F32); backend.Permute0213(qMh, q, t, hq, d);
        Tensor kMh = new(new TensorShape(1, hkv, ctxLen, d), DType.F32); backend.Permute0213(kMh, k, ctxLen, hkv, d);
        Tensor vMh = new(new TensorShape(1, hkv, ctxLen, d), DType.F32); backend.Permute0213(vMh, v, ctxLen, hkv, d);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn = new(new TensorShape(1, hq, t, d), DType.F32);
        backend.FlashAttention(attn, qMh, kMh, vMh, ctxLen, group, causal: false, qOffset: 0, 1f / MathF.Sqrt(d));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, t, hq * d), DType.F32);
        backend.Permute0213(attnFlat, attn, hq, t, d);
        attn.Dispose();
        Tensor cout = new(flat, DType.F32);
        Project(backend, cout, attnFlat, _cwo!, null);
        attnFlat.Dispose();
        return cout;
    }

    private Tensor SwiGlu(IBackend backend, Tensor x, int t)
    {
        TensorShape ff = new(1, t, _cfg.FfnHiddenSize);
        Tensor gate = new(ff, DType.F32);
        Project(backend, gate, x, _w1, null);
        Tensor act = new(ff, DType.F32);
        backend.Silu(act, gate);
        gate.Dispose();
        Tensor up = new(ff, DType.F32);
        Project(backend, up, x, _w3, null);
        Tensor comb = new(ff, DType.F32);
        backend.Mul(comb, act, up);
        act.Dispose(); up.Dispose();
        Tensor outp = new(new TensorShape(1, t, _cfg.Dim), DType.F32);
        Project(backend, outp, comb, _w2, null);
        comb.Dispose();
        return outp;
    }

    // Float weights take the cuBLAS Linear path (quantized weights aren't used on this AR path). Mirrors the LLM
    // decoder's internal projection dispatch, which is assembly-internal and not reachable across the package edge.
    private static void Project(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias)
        => backend.Linear(output, input, weight, bias);

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
