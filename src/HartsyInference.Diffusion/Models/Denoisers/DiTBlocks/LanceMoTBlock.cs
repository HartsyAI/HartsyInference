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

    /// <summary>Forward over a packed <c>[seq, hidden]</c> sequence laid out as three contiguous role segments:
    /// und rows <c>[0, nPre)</c>, gen rows <c>[nPre, nPre+nGen)</c>, und rows <c>[nPre+nGen, seq)</c> (the layout
    /// <c>LancePipelineCommon.BuildGenSequence</c> always produces — <see cref="Denoisers.LanceTransformer"/>
    /// validates it). <paramref name="cos"/>/<paramref name="sin"/> are <c>[1, seq, headDim]</c>;
    /// <paramref name="mask"/> is an optional additive attention mask <c>[1,1,seq,seq]</c> (null = full attention).</summary>
    // GPU-residency rewrite: the old path routed roles via host gather/scatter memcpys (DataPointer reads), plus
    // host RoPE / head permutes / GQA expand / residual adds — every one a full-stream D2H drain, many times per
    // block × 36 blocks × CFG, which was the ~12 s/step cost. Contiguous role segments make routing a pair of
    // device SliceRows/Concat ops; RoPE is the ApplyRopeSingle rotate-half kernel (per-tensor, so GQA's 16:2 head
    // mismatch is fine); permutes are Permute0213; the GQA expand is RepeatKvHeads.
    public Tensor Forward(IBackend backend, Tensor x, Tensor cos, Tensor sin, int nPre, int nGen, Tensor? mask)
    {
        int seq = (int)x.Shape[0];

        Tensor normed = RoleNorm(backend, x, nPre, nGen, _inNorm!, _inNormg!, seq);
        Tensor attn = Attention(backend, normed, cos, sin, nPre, nGen, mask, seq);
        normed.Dispose();
        Tensor afterAttn = new Tensor(x.Shape, DType.F32);
        backend.Add(afterAttn, x, attn);
        attn.Dispose();

        Tensor normed2 = RoleNorm(backend, afterAttn, nPre, nGen, _postNorm!, _postNormg!, seq);
        Tensor mlp = RoleMlp(backend, normed2, nPre, nGen, seq);
        normed2.Dispose();
        Tensor result = new Tensor(x.Shape, DType.F32);
        backend.Add(result, afterAttn, mlp);
        afterAttn.Dispose();
        mlp.Dispose();
        return result;
    }

    private Tensor Attention(IBackend backend, Tensor normed, Tensor cos, Tensor sin,
        int nPre, int nGen, Tensor? mask, int seq)
    {
        float scale = 1.0f / MathF.Sqrt(_headDim);

        // Role-routed Q/K/V over the three contiguous segments, concatenated back into joint [1, seq, H, D]
        // ([B, L, H, D] views over the flat rows — dim-0 concat is a contiguous device copy).
        (Tensor q, Tensor k, Tensor v) = ProjectQkvJoint(backend, normed, nPre, nGen, seq);

        // RoPE (rotate-half, cos/sin broadcast over heads) on the pre-permute layout; per-tensor kernel because
        // GQA Q (16 heads) and K (2 heads) stride differently.
        backend.ApplyRopeSingle(q, cos, sin);
        backend.ApplyRopeSingle(k, cos, sin);

        // [1, L, H, D] → [1, H, L, D]; GQA-expand K/V to the query head count.
        int rep = _numHeads / _numKvHeads;
        Tensor qMh = new Tensor(new TensorShape(1, _numHeads, seq, _headDim), DType.F32);
        backend.Permute0213(qMh, q, seq, _numHeads, _headDim);
        q.Dispose();
        Tensor kSmall = new Tensor(new TensorShape(1, _numKvHeads, seq, _headDim), DType.F32);
        backend.Permute0213(kSmall, k, seq, _numKvHeads, _headDim);
        k.Dispose();
        Tensor vSmall = new Tensor(new TensorShape(1, _numKvHeads, seq, _headDim), DType.F32);
        backend.Permute0213(vSmall, v, seq, _numKvHeads, _headDim);
        v.Dispose();

        Tensor kMh = new Tensor(new TensorShape(1, _numHeads, seq, _headDim), DType.F32);
        backend.RepeatKvHeads(kMh, kSmall, _numKvHeads, rep);
        kSmall.Dispose();
        Tensor vMh = new Tensor(new TensorShape(1, _numHeads, seq, _headDim), DType.F32);
        backend.RepeatKvHeads(vMh, vSmall, _numKvHeads, rep);
        vSmall.Dispose();

        // Joint SDPA. allowF16: Lance ships QK-RMSNorm (bounded scores) and the additive mask is finite (-1e9),
        // so the cuDNN fused engine (mask as an fp32 bias score-modifier) is range-safe.
        Tensor attnOut = new Tensor(new TensorShape(1, _numHeads, seq, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, mask, scale, allowF16: true);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = new Tensor(new TensorShape(seq, _hidden), DType.F32);
        backend.Permute0213(flat, attnOut, _numHeads, seq, _headDim);
        attnOut.Dispose();

        // o_proj (role-routed, bias-less).
        Tensor outProj = RoleLinear(backend, flat, nPre, nGen, seq, _hidden, _oW!, null, _oWg!, null);
        flat.Dispose();
        return outProj;
    }

    /// <summary>Role-routed QKV: each contiguous segment goes through its role's projections (+ optional per-head
    /// QK-RMSNorm), then the segments are concatenated back in sequence order. Returns joint
    /// <c>q [1, seq, H, D]</c>, <c>k/v [1, seq, Hkv, D]</c>.</summary>
    private (Tensor q, Tensor k, Tensor v) ProjectQkvJoint(IBackend backend, Tensor normed, int nPre, int nGen, int seq)
    {
        Tensor[] qSegs = new Tensor[3];
        Tensor[] kSegs = new Tensor[3];
        Tensor[] vSegs = new Tensor[3];
        int segCount = 0;
        int offset = 0;
        Span<int> segLens = stackalloc int[3] { nPre, nGen, seq - nPre - nGen };
        for (int s = 0; s < 3; s++)
        {
            int n = segLens[s];
            if (n == 0) continue;
            bool gen = s == 1;
            Tensor sub = new Tensor(new TensorShape(n, _hidden), DType.F32);
            backend.SliceRows(sub, normed, offset);

            Tensor qSub = new Tensor(new TensorShape(n, _numHeads, _headDim), DType.F32);
            backend.Linear(qSub, sub, gen ? _qWg! : _qW!, gen ? _qBg : _qB);
            Tensor kSub = new Tensor(new TensorShape(n, _numKvHeads, _headDim), DType.F32);
            backend.Linear(kSub, sub, gen ? _kWg! : _kW!, gen ? _kBg : _kB);
            Tensor vSub = new Tensor(new TensorShape(n, _numKvHeads, _headDim), DType.F32);
            backend.Linear(vSub, sub, gen ? _vWg! : _vW!, gen ? _vBg : _vB);
            sub.Dispose();

            if (_qkNorm)
            {
                QkNorm qNorm = gen ? _qNormg! : _qNorm!;
                QkNorm kNorm = gen ? _kNormg! : _kNorm!;
                Tensor qn = new Tensor(qSub.Shape, DType.F32);
                backend.RmsNorm(qn, qSub, qNorm.Weight, qNorm.Eps);
                qSub.Dispose();
                qSub = qn;
                Tensor kn = new Tensor(kSub.Shape, DType.F32);
                backend.RmsNorm(kn, kSub, kNorm.Weight, kNorm.Eps);
                kSub.Dispose();
                kSub = kn;
            }

            qSegs[segCount] = qSub;
            kSegs[segCount] = kSub;
            vSegs[segCount] = vSub;
            segCount++;
            offset += n;
        }

        Tensor q = ConcatSegments(backend, qSegs.AsSpan(0, segCount), new TensorShape(1, seq, _numHeads, _headDim));
        Tensor k = ConcatSegments(backend, kSegs.AsSpan(0, segCount), new TensorShape(1, seq, _numKvHeads, _headDim));
        Tensor v = ConcatSegments(backend, vSegs.AsSpan(0, segCount), new TensorShape(1, seq, _numKvHeads, _headDim));
        for (int i = 0; i < segCount; i++)
        {
            qSegs[i].Dispose();
            kSegs[i].Dispose();
            vSegs[i].Dispose();
        }
        return (q, k, v);
    }

    /// <summary>Role-routed Linear over the three contiguous segments → concatenated <c>[seq, outDim]</c>.</summary>
    private Tensor RoleLinear(IBackend backend, Tensor input, int nPre, int nGen, int seq, int outDim,
        Tensor undW, Tensor? undB, Tensor genW, Tensor? genB)
    {
        Tensor[] segs = new Tensor[3];
        int segCount = 0;
        int offset = 0;
        Span<int> segLens = stackalloc int[3] { nPre, nGen, seq - nPre - nGen };
        for (int s = 0; s < 3; s++)
        {
            int n = segLens[s];
            if (n == 0) continue;
            bool gen = s == 1;
            Tensor sub = new Tensor(new TensorShape(n, _hidden), DType.F32);
            backend.SliceRows(sub, input, offset);
            Tensor outSub = new Tensor(new TensorShape(n, outDim), DType.F32);
            backend.Linear(outSub, sub, gen ? genW : undW, gen ? genB : undB);
            sub.Dispose();
            segs[segCount++] = outSub;
            offset += n;
        }
        Tensor result = ConcatSegments(backend, segs.AsSpan(0, segCount), new TensorShape(seq, outDim));
        for (int i = 0; i < segCount; i++) segs[i].Dispose();
        return result;
    }

    private Tensor RoleMlp(IBackend backend, Tensor normed, int nPre, int nGen, int seq)
    {
        Tensor[] segs = new Tensor[3];
        int segCount = 0;
        int offset = 0;
        Span<int> segLens = stackalloc int[3] { nPre, nGen, seq - nPre - nGen };
        for (int s = 0; s < 3; s++)
        {
            int n = segLens[s];
            if (n == 0) continue;
            bool gen = s == 1;
            Tensor sub = new Tensor(new TensorShape(n, _hidden), DType.F32);
            backend.SliceRows(sub, normed, offset);
            segs[segCount++] = SwiGlu(backend, sub, n,
                gen ? _gateWg! : _gateW!, gen ? _upWg! : _upW!, gen ? _downWg! : _downW!);
            sub.Dispose();
            offset += n;
        }
        Tensor result = ConcatSegments(backend, segs.AsSpan(0, segCount), new TensorShape(seq, _hidden));
        for (int i = 0; i < segCount; i++) segs[i].Dispose();
        return result;
    }

    private Tensor SwiGlu(IBackend backend, Tensor sub, int n, Tensor gateW, Tensor upW, Tensor downW)
    {
        Tensor gate = new Tensor(new TensorShape(n, _ffn), DType.F32);
        backend.Linear(gate, sub, gateW, null);
        Tensor act = new Tensor(new TensorShape(n, _ffn), DType.F32);
        backend.Silu(act, gate);
        gate.Dispose();
        Tensor up = new Tensor(new TensorShape(n, _ffn), DType.F32);
        backend.Linear(up, sub, upW, null);
        Tensor prod = new Tensor(new TensorShape(n, _ffn), DType.F32);
        backend.Mul(prod, act, up);
        act.Dispose();
        up.Dispose();
        Tensor outSub = new Tensor(new TensorShape(n, _hidden), DType.F32);
        backend.Linear(outSub, prod, downW, null);
        prod.Dispose();
        return outSub;
    }

    private Tensor RoleNorm(IBackend backend, Tensor x, int nPre, int nGen, Tensor undW, Tensor genW, int seq)
    {
        Tensor[] segs = new Tensor[3];
        int segCount = 0;
        int offset = 0;
        Span<int> segLens = stackalloc int[3] { nPre, nGen, seq - nPre - nGen };
        for (int s = 0; s < 3; s++)
        {
            int n = segLens[s];
            if (n == 0) continue;
            Tensor sub = new Tensor(new TensorShape(n, _hidden), DType.F32);
            backend.SliceRows(sub, x, offset);
            Tensor normedSeg = new Tensor(sub.Shape, DType.F32);
            backend.RmsNorm(normedSeg, sub, s == 1 ? genW : undW, _eps);
            sub.Dispose();
            segs[segCount++] = normedSeg;
            offset += n;
        }
        Tensor result = ConcatSegments(backend, segs.AsSpan(0, segCount), new TensorShape(seq, _hidden));
        for (int i = 0; i < segCount; i++) segs[i].Dispose();
        return result;
    }

    /// <summary>Concatenates row segments into <paramref name="shape"/> (single segment = device copy). Dim-0
    /// concat is a contiguous device copy regardless of the output view's rank.</summary>
    private static Tensor ConcatSegments(IBackend backend, ReadOnlySpan<Tensor> segs, TensorShape shape)
    {
        Tensor result = new Tensor(shape, DType.F32);
        if (segs.Length == 1)
            backend.CopyInto(result, segs[0]);
        else
            backend.Concat(result, segs, 0);
        return result;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
