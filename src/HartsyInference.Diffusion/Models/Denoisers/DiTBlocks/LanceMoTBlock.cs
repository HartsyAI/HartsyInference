using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Lance MoT (Mixture-of-Tokens) decoder layer (<c>Qwen2MoTDecoderLayer</c> in upstream <c>qwen2_navit.py</c>). A Qwen2.5-VL decoder block with **two complete parameter sets** per layer: the "und" set (standard <c>q/k/v/o_proj</c>, <c>mlp</c>, norms) for understanding tokens (text + ViT-semantic) and the "gen" set (<c>*_moe_gen</c>) for generation tokens (clean/noisy VAE). Tokens are routed deterministically by modality role — NOT a learned gate — then a SINGLE joint attention runs over the whole packed sequence.
///
/// <para>First-pass routing uses re-batched gather/scatter (the research doc's recommended approach): gather each role's token subset, run its path's linears, scatter back. Q/K/V/O and both norms and the SwiGLU MLP are all role-routed; RoPE + the joint GQA attention run over the full sequence. Reuses <see cref="QkNorm"/> and <see cref="Multimodal3DRope"/>.</para>
///
/// <para>Operates on a 2-D packed sequence <c>[seq, hidden]</c> (B=1; the joint sequence is one sample). GQA: 16 query heads : 2 KV heads (factor 8). Optional per-head QK-RMSNorm when the checkpoint ships <c>q_norm</c>/<c>k_norm</c>.</para></summary>
public sealed unsafe class LanceMoTBlock
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _ffn;
    private readonly float _eps;
    private readonly bool _qkNorm;

    // "und" path
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW;
    private Tensor? _gateW, _upW, _downW;
    private Tensor? _inNorm, _postNorm;
    private QkNorm? _qNorm, _kNorm;
    // "gen" path (_moe_gen)
    private Tensor? _qWg, _qBg, _kWg, _kBg, _vWg, _vBg, _oWg;
    private Tensor? _gateWg, _upWg, _downWg;
    private Tensor? _inNormg, _postNormg;
    private QkNorm? _qNormg, _kNormg;

    public LanceMoTBlock(int hidden, int numHeads, int numKvHeads, int ffn, float eps, bool qkNorm)
    {
        _hidden = hidden;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = hidden / numHeads;
        _ffn = ffn;
        _eps = eps;
        _qkNorm = qkNorm;
        if (qkNorm)
        {
            _qNorm = new QkNorm(_headDim, eps);
            _kNorm = new QkNorm(_headDim, eps);
            _qNormg = new QkNorm(_headDim, eps);
            _kNormg = new QkNorm(_headDim, eps);
        }
    }

    /// <summary>Loads both parameter sets. <paramref name="prefix"/> is e.g. <c>"layers.0"</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // und path
        _qW = weights[$"{prefix}.self_attn.q_proj.weight"];
        weights.TryGetValue($"{prefix}.self_attn.q_proj.bias", out _qB);
        _kW = weights[$"{prefix}.self_attn.k_proj.weight"];
        weights.TryGetValue($"{prefix}.self_attn.k_proj.bias", out _kB);
        _vW = weights[$"{prefix}.self_attn.v_proj.weight"];
        weights.TryGetValue($"{prefix}.self_attn.v_proj.bias", out _vB);
        _oW = weights[$"{prefix}.self_attn.o_proj.weight"];
        _gateW = weights[$"{prefix}.mlp.gate_proj.weight"];
        _upW = weights[$"{prefix}.mlp.up_proj.weight"];
        _downW = weights[$"{prefix}.mlp.down_proj.weight"];
        _inNorm = LoadF32(weights, $"{prefix}.input_layernorm.weight");
        _postNorm = LoadF32(weights, $"{prefix}.post_attention_layernorm.weight");

        // gen path
        _qWg = weights[$"{prefix}.self_attn.q_proj_moe_gen.weight"];
        weights.TryGetValue($"{prefix}.self_attn.q_proj_moe_gen.bias", out _qBg);
        _kWg = weights[$"{prefix}.self_attn.k_proj_moe_gen.weight"];
        weights.TryGetValue($"{prefix}.self_attn.k_proj_moe_gen.bias", out _kBg);
        _vWg = weights[$"{prefix}.self_attn.v_proj_moe_gen.weight"];
        weights.TryGetValue($"{prefix}.self_attn.v_proj_moe_gen.bias", out _vBg);
        _oWg = weights[$"{prefix}.self_attn.o_proj_moe_gen.weight"];
        _gateWg = weights[$"{prefix}.mlp_moe_gen.gate_proj.weight"];
        _upWg = weights[$"{prefix}.mlp_moe_gen.up_proj.weight"];
        _downWg = weights[$"{prefix}.mlp_moe_gen.down_proj.weight"];
        _inNormg = LoadF32(weights, $"{prefix}.input_layernorm_moe_gen.weight");
        _postNormg = LoadF32(weights, $"{prefix}.post_attention_layernorm_moe_gen.weight");

        if (_qkNorm)
        {
            _qNorm!.LoadWeights(weights[$"{prefix}.self_attn.q_norm.weight"]);
            _kNorm!.LoadWeights(weights[$"{prefix}.self_attn.k_norm.weight"]);
            _qNormg!.LoadWeights(weights[$"{prefix}.self_attn.q_norm_moe_gen.weight"]);
            _kNormg!.LoadWeights(weights[$"{prefix}.self_attn.k_norm_moe_gen.weight"]);
        }
    }

    /// <summary>Enumerates all weights of both paths for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _kW, _kB, _vW, _vB, _oW, _gateW, _upW, _downW, _inNorm, _postNorm,
            _qWg, _qBg, _kWg, _kBg, _vWg, _vBg, _oWg, _gateWg, _upWg, _downWg, _inNormg, _postNormg })
            if (t is not null) yield return t;
        if (_qkNorm)
            foreach (QkNorm n in new[] { _qNorm!, _kNorm!, _qNormg!, _kNormg! })
                foreach (Tensor t in n.EnumerateWeights()) yield return t;
    }

    /// <summary>Forward over a packed <c>[seq, hidden]</c> sequence. <paramref name="undIdx"/> / <paramref name="genIdx"/> partition the rows by modality role; <paramref name="cos"/>/<paramref name="sin"/> are <c>[seq, headDim]</c>; <paramref name="mask"/> is an optional additive attention mask <c>[1,1,seq,seq]</c> (null = full attention).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor cos, Tensor sin, Multimodal3DRope rope,
        int[] undIdx, int[] genIdx, Tensor? mask)
    {
        int seq = (int)x.Shape[0];

        // ── input norm (role-routed) ──
        Tensor normed = RoleNorm(backend, x, undIdx, genIdx, _inNorm!, _inNormg!, seq);
        // ── attention (role-routed QKV/O, joint GQA SDPA) ──
        Tensor attn = Attention(backend, normed, cos, sin, rope, undIdx, genIdx, mask, seq);
        normed.Dispose();
        Tensor afterAttn = AddRows(x, attn, seq);
        attn.Dispose();

        // ── post-attn norm + SwiGLU MLP (role-routed) ──
        Tensor normed2 = RoleNorm(backend, afterAttn, undIdx, genIdx, _postNorm!, _postNormg!, seq);
        Tensor mlp = RoleMlp(backend, normed2, undIdx, genIdx, seq);
        normed2.Dispose();
        Tensor result = AddRows(afterAttn, mlp, seq);
        afterAttn.Dispose();
        mlp.Dispose();
        return result;
    }

    private Tensor Attention(IBackend backend, Tensor normed, Tensor cos, Tensor sin, Multimodal3DRope rope,
        int[] undIdx, int[] genIdx, Tensor? mask, int seq)
    {
        TensorShape qShape = new TensorShape(seq, _numHeads, _headDim);
        TensorShape kvShape = new TensorShape(seq, _numKvHeads, _headDim);
        Tensor q = new Tensor(qShape, DType.F32);
        Tensor k = new Tensor(kvShape, DType.F32);
        Tensor v = new Tensor(kvShape, DType.F32);

        // Project each role's subset with its path's weights, scatter into the joint Q/K/V.
        ProjectRole(backend, normed, undIdx, _qW!, _qB, _kW!, _kB, _vW!, _vB, _qNorm, _kNorm, q, k, v);
        ProjectRole(backend, normed, genIdx, _qWg!, _qBg, _kWg!, _kBg, _vWg!, _vBg, _qNormg, _kNormg, q, k, v);

        // RoPE over the full sequence (Q and K).
        rope.ApplyRotary(q, cos, sin);
        rope.ApplyRotary(k, cos, sin);

        // GQA: expand K/V to numHeads, permute to [1, H, seq, D], joint SDPA.
        Tensor qMh = PermuteToBhsd(q, seq, _numHeads);
        Tensor kMh = ExpandKvToBhsd(k, seq);
        Tensor vMh = ExpandKvToBhsd(v, seq);
        q.Dispose(); k.Dispose(); v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(new TensorShape(1, _numHeads, seq, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, mask, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = PermuteFromBhsd(attnOut, seq);   // [seq, hidden]
        attnOut.Dispose();

        // o_proj (role-routed).
        Tensor outProj = new Tensor(new TensorShape(seq, _hidden), DType.F32);
        new Span<float>((float*)outProj.DataPointer, checked((int)((long)seq * _hidden))).Clear();
        ProjectOut(backend, flat, undIdx, _oW!, outProj);
        ProjectOut(backend, flat, genIdx, _oWg!, outProj);
        flat.Dispose();
        return outProj;
    }

    private void ProjectRole(IBackend backend, Tensor normed, int[] idx,
        Tensor qW, Tensor? qB, Tensor kW, Tensor? kB, Tensor vW, Tensor? vB,
        QkNorm? qNorm, QkNorm? kNorm, Tensor qDst, Tensor kDst, Tensor vDst)
    {
        int n = idx.Length;
        if (n == 0) return;
        Tensor sub = Gather(normed, idx, _hidden);

        Tensor qSub = new Tensor(new TensorShape(n, _numHeads * _headDim), DType.F32);
        Tensor kSub = new Tensor(new TensorShape(n, _numKvHeads * _headDim), DType.F32);
        Tensor vSub = new Tensor(new TensorShape(n, _numKvHeads * _headDim), DType.F32);
        backend.Linear(qSub, sub, qW, qB);
        backend.Linear(kSub, sub, kW, kB);
        backend.Linear(vSub, sub, vW, vB);
        sub.Dispose();

        if (_qkNorm)
        {
            Tensor qn = new Tensor(qSub.Shape, DType.F32);
            Tensor kn = new Tensor(kSub.Shape, DType.F32);
            qNorm!.Forward(qn, qSub, n * _numHeads);
            kNorm!.Forward(kn, kSub, n * _numKvHeads);
            qSub.Dispose(); kSub.Dispose();
            qSub = qn; kSub = kn;
        }

        ScatterHeads(qSub, idx, qDst, _numHeads);
        ScatterHeads(kSub, idx, kDst, _numKvHeads);
        ScatterHeads(vSub, idx, vDst, _numKvHeads);
        qSub.Dispose(); kSub.Dispose(); vSub.Dispose();
    }

    private void ProjectOut(IBackend backend, Tensor flat, int[] idx, Tensor oW, Tensor dst)
    {
        int n = idx.Length;
        if (n == 0) return;
        Tensor sub = Gather(flat, idx, _hidden);
        Tensor outSub = new Tensor(new TensorShape(n, _hidden), DType.F32);
        backend.Linear(outSub, sub, oW, null);
        sub.Dispose();
        Scatter(outSub, idx, dst, _hidden);
        outSub.Dispose();
    }

    private Tensor RoleMlp(IBackend backend, Tensor normed, int[] undIdx, int[] genIdx, int seq)
    {
        Tensor outT = new Tensor(new TensorShape(seq, _hidden), DType.F32);
        new Span<float>((float*)outT.DataPointer, checked((int)((long)seq * _hidden))).Clear();
        SwiGluRole(backend, normed, undIdx, _gateW!, _upW!, _downW!, outT);
        SwiGluRole(backend, normed, genIdx, _gateWg!, _upWg!, _downWg!, outT);
        return outT;
    }

    private void SwiGluRole(IBackend backend, Tensor normed, int[] idx, Tensor gateW, Tensor upW, Tensor downW, Tensor dst)
    {
        int n = idx.Length;
        if (n == 0) return;
        Tensor sub = Gather(normed, idx, _hidden);
        Tensor gate = new Tensor(new TensorShape(n, _ffn), DType.F32);
        backend.Linear(gate, sub, gateW, null);
        Tensor act = new Tensor(new TensorShape(n, _ffn), DType.F32);
        backend.Silu(act, gate);
        gate.Dispose();
        Tensor up = new Tensor(new TensorShape(n, _ffn), DType.F32);
        backend.Linear(up, sub, upW, null);
        sub.Dispose();
        Tensor prod = new Tensor(new TensorShape(n, _ffn), DType.F32);
        backend.Mul(prod, act, up);
        act.Dispose(); up.Dispose();
        Tensor outSub = new Tensor(new TensorShape(n, _hidden), DType.F32);
        backend.Linear(outSub, prod, downW, null);
        prod.Dispose();
        Scatter(outSub, idx, dst, _hidden);
        outSub.Dispose();
    }

    private Tensor RoleNorm(IBackend backend, Tensor x, int[] undIdx, int[] genIdx, Tensor undW, Tensor genW, int seq)
    {
        Tensor outT = new Tensor(new TensorShape(seq, _hidden), DType.F32);
        new Span<float>((float*)outT.DataPointer, checked((int)((long)seq * _hidden))).Clear();
        NormRole(backend, x, undIdx, undW, outT);
        NormRole(backend, x, genIdx, genW, outT);
        return outT;
    }

    private void NormRole(IBackend backend, Tensor x, int[] idx, Tensor normW, Tensor dst)
    {
        int n = idx.Length;
        if (n == 0) return;
        Tensor sub = Gather(x, idx, _hidden);
        Tensor normed = new Tensor(sub.Shape, DType.F32);
        backend.RmsNorm(normed, sub, normW, _eps);
        sub.Dispose();
        Scatter(normed, idx, dst, _hidden);
        normed.Dispose();
    }

    // ── row gather/scatter + head reshapes ──
    private static Tensor Gather(Tensor src, int[] idx, int dim)
    {
        Tensor outT = new Tensor(new TensorShape(idx.Length, dim), DType.F32);
        float* s = (float*)src.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int i = 0; i < idx.Length; i++)
            Buffer.MemoryCopy(s + (long)idx[i] * dim, d + (long)i * dim, (long)dim * 4, (long)dim * 4);
        return outT;
    }

    private static void Scatter(Tensor src, int[] idx, Tensor dst, int dim)
    {
        float* s = (float*)src.DataPointer;
        float* d = (float*)dst.DataPointer;
        for (int i = 0; i < idx.Length; i++)
            Buffer.MemoryCopy(s + (long)i * dim, d + (long)idx[i] * dim, (long)dim * 4, (long)dim * 4);
    }

    /// <summary>Scatter a <c>[n, heads*headDim]</c> sub-buffer into a <c>[seq, heads, headDim]</c> joint tensor at the given row indexes.</summary>
    private void ScatterHeads(Tensor src, int[] idx, Tensor dst, int heads)
    {
        int rowLen = heads * _headDim;
        float* s = (float*)src.DataPointer;
        float* d = (float*)dst.DataPointer;
        for (int i = 0; i < idx.Length; i++)
            Buffer.MemoryCopy(s + (long)i * rowLen, d + (long)idx[i] * rowLen, (long)rowLen * 4, (long)rowLen * 4);
    }

    private Tensor PermuteToBhsd(Tensor x, int seq, int heads)
    {
        Tensor outT = new Tensor(new TensorShape(1, heads, seq, _headDim), DType.F32);
        float* s = (float*)x.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int p = 0; p < seq; p++)
            for (int h = 0; h < heads; h++)
            {
                long sOff = ((long)p * heads + h) * _headDim;
                long dOff = ((long)h * seq + p) * _headDim;
                Buffer.MemoryCopy(s + sOff, d + dOff, (long)_headDim * 4, (long)_headDim * 4);
            }
        return outT;
    }

    /// <summary>Expand <c>[seq, numKv, headDim]</c> to <c>[1, numHeads, seq, headDim]</c> by repeating each KV head <c>numHeads/numKv</c> times (GQA).</summary>
    private Tensor ExpandKvToBhsd(Tensor kv, int seq)
    {
        int rep = _numHeads / _numKvHeads;
        Tensor outT = new Tensor(new TensorShape(1, _numHeads, seq, _headDim), DType.F32);
        float* s = (float*)kv.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int p = 0; p < seq; p++)
            for (int kvh = 0; kvh < _numKvHeads; kvh++)
            {
                long sOff = ((long)p * _numKvHeads + kvh) * _headDim;
                for (int r = 0; r < rep; r++)
                {
                    int h = kvh * rep + r;
                    long dOff = ((long)h * seq + p) * _headDim;
                    Buffer.MemoryCopy(s + sOff, d + dOff, (long)_headDim * 4, (long)_headDim * 4);
                }
            }
        return outT;
    }

    private Tensor PermuteFromBhsd(Tensor x, int seq)
    {
        Tensor outT = new Tensor(new TensorShape(seq, _hidden), DType.F32);
        float* s = (float*)x.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int h = 0; h < _numHeads; h++)
            for (int p = 0; p < seq; p++)
            {
                long sOff = ((long)h * seq + p) * _headDim;
                long dOff = (long)p * _hidden + (long)h * _headDim;
                Buffer.MemoryCopy(s + sOff, d + dOff, (long)_headDim * 4, (long)_headDim * 4);
            }
        return outT;
    }

    private static Tensor AddRows(Tensor a, Tensor b, int seq)
    {
        Tensor outT = new Tensor(a.Shape, DType.F32);
        long n = a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        float* po = (float*)outT.DataPointer;
        for (long i = 0; i < n; i++) po[i] = pa[i] + pb[i];
        return outT;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
