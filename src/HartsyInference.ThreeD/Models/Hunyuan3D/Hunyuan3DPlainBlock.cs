using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>One <c>HunYuanDiTBlock</c> of Hunyuan3D-2.1: optional U-skip, self-attention, cross-attention to the image tokens, then a dense MLP or a mixture of experts.</summary>
/// <remarks>Batch 1 only; classifier-free guidance runs as two forwards. Activations are F32.</remarks>
internal sealed unsafe class Hunyuan3DPlainBlock
{
    private const float Eps = 1e-6f;
    private readonly int _width, _heads, _headDim, _experts, _topK;
    private readonly bool _skip, _moe;
    private readonly AttentionLayer _attn;
    private Tensor? _skipW, _skipB, _skipNormW, _skipNormB;
    private Tensor? _n1W, _n1B, _n2W, _n2B, _n3W, _n3B;
    private Tensor? _sqW, _skW, _svW, _sqNorm, _skNorm, _soW, _soB;
    private Tensor? _cqW, _ckW, _cvW, _cqNorm, _ckNorm, _coW, _coB;
    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;
    private Tensor? _gateW;
    private Tensor?[]? _expertFc1W, _expertFc1B, _expertFc2W, _expertFc2B;
    private bool _halfWeights;

    internal Hunyuan3DPlainBlock(int width, int heads, bool skip, bool moe, int experts, int topK)
    {
        _width = width; _heads = heads; _headDim = width / heads;
        _skip = skip; _moe = moe; _experts = experts; _topK = topK;
        _attn = new AttentionLayer(new AttentionLayerConfig
        {
            Heads = heads, KvHeads = heads, HeadDim = _headDim, ComputeDType = DType.F32, AllowF16 = true,
        });
    }

    internal bool HasSkip => _skip;

    /// <summary>Hidden width of the dense MLP (or of the shared expert in a MoE block), read from the loaded weights.</summary>
    internal int HiddenDim => (int)_fc1W!.Shape[0];

    internal void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p, bool halfWeights)
    {
        _halfWeights = halfWeights;
        _n1W = F(w, $"{p}.norm1.weight"); _n1B = F(w, $"{p}.norm1.bias");
        _n2W = F(w, $"{p}.norm2.weight"); _n2B = F(w, $"{p}.norm2.bias");
        _n3W = F(w, $"{p}.norm3.weight"); _n3B = F(w, $"{p}.norm3.bias");
        // Upstream concatenates the projections and views the result as (heads, 3 * headDim), so a head's "q", "k" and "v" are three
        // consecutive channel groups of the fused row, not the head's own slices of q, k and v. The weights were trained that way; the
        // rows are regrouped once here so plain per-projection Linears reproduce it.
        Tensor q = F(w, $"{p}.attn1.to_q.weight"), k = F(w, $"{p}.attn1.to_k.weight"), v = F(w, $"{p}.attn1.to_v.weight");
        (_sqW, _skW, _svW) = Regroup(q, k, v, 3);
        Release(q, w[$"{p}.attn1.to_q.weight"]); Release(k, w[$"{p}.attn1.to_k.weight"]); Release(v, w[$"{p}.attn1.to_v.weight"]);
        _sqNorm = F(w, $"{p}.attn1.q_norm.weight"); _skNorm = F(w, $"{p}.attn1.k_norm.weight");
        _soW = F(w, $"{p}.attn1.out_proj.weight"); _soB = F(w, $"{p}.attn1.out_proj.bias");
        _cqW = F(w, $"{p}.attn2.to_q.weight");
        Tensor ck = F(w, $"{p}.attn2.to_k.weight"), cv = F(w, $"{p}.attn2.to_v.weight");
        (_ckW, _cvW, _) = Regroup(ck, cv, null, 2);
        Release(ck, w[$"{p}.attn2.to_k.weight"]); Release(cv, w[$"{p}.attn2.to_v.weight"]);
        _cqNorm = F(w, $"{p}.attn2.q_norm.weight"); _ckNorm = F(w, $"{p}.attn2.k_norm.weight");
        _coW = F(w, $"{p}.attn2.out_proj.weight"); _coB = F(w, $"{p}.attn2.out_proj.bias");
        if (_skip)
        {
            _skipW = F(w, $"{p}.skip_linear.weight"); _skipB = F(w, $"{p}.skip_linear.bias");
            _skipNormW = F(w, $"{p}.skip_norm.weight"); _skipNormB = F(w, $"{p}.skip_norm.bias");
        }

        if (!_moe)
        {
            _fc1W = F(w, $"{p}.mlp.fc1.weight"); _fc1B = F(w, $"{p}.mlp.fc1.bias");
            _fc2W = F(w, $"{p}.mlp.fc2.weight"); _fc2B = F(w, $"{p}.mlp.fc2.bias");
            return;
        }

        _gateW = F(w, $"{p}.moe.gate.weight");
        _fc1W = F(w, $"{p}.moe.shared_experts.net.0.proj.weight"); _fc1B = F(w, $"{p}.moe.shared_experts.net.0.proj.bias");
        _fc2W = F(w, $"{p}.moe.shared_experts.net.2.weight"); _fc2B = F(w, $"{p}.moe.shared_experts.net.2.bias");
        _expertFc1W = new Tensor?[_experts]; _expertFc1B = new Tensor?[_experts];
        _expertFc2W = new Tensor?[_experts]; _expertFc2B = new Tensor?[_experts];
        for (int e = 0; e < _experts; e++)
        {
            _expertFc1W[e] = F(w, $"{p}.moe.experts.{e}.net.0.proj.weight"); _expertFc1B[e] = F(w, $"{p}.moe.experts.{e}.net.0.proj.bias");
            _expertFc2W[e] = F(w, $"{p}.moe.experts.{e}.net.2.weight"); _expertFc2B[e] = F(w, $"{p}.moe.experts.{e}.net.2.bias");
        }
    }

    internal IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _skipW, _skipB, _skipNormW, _skipNormB, _n1W, _n1B, _n2W, _n2B, _n3W, _n3B, _sqW, _skW, _svW, _sqNorm, _skNorm, _soW, _soB,
            _cqW, _ckW, _cvW, _cqNorm, _ckNorm, _coW, _coB, _fc1W, _fc1B, _fc2W, _fc2B, _gateW,
        ];
        foreach (Tensor? t in all)
        {
            if (t is not null) yield return t;
        }

        if (_expertFc1W is null) yield break;
        for (int e = 0; e < _experts; e++)
        {
            foreach (Tensor? t in new[] { _expertFc1W[e], _expertFc1B![e], _expertFc2W![e], _expertFc2B![e] })
            {
                if (t is not null) yield return t;
            }
        }
    }

    /// <summary>Runs the block on <paramref name="x"/> <c>[1,S,W]</c> (consumed) with image tokens <paramref name="context"/> and, for blocks after the middle, the matching <paramref name="skipValue"/> (not consumed).</summary>
    internal Tensor Forward(IBackend backend, Tensor x, Tensor context, Tensor? skipValue)
    {
        int s = (int)x.Shape[1];
        TensorShape flat = new(1, s, _width);
        if (_skip)
        {
            Tensor cat = new(new TensorShape(1, s, 2 * _width), DType.F32);
            backend.Concat(cat, [skipValue!, x], 2);
            x.Dispose();
            Tensor lin = new(flat, DType.F32);
            backend.Linear(lin, cat, _skipW!, _skipB!);
            cat.Dispose();
            x = new(flat, DType.F32);
            backend.LayerNorm(x, lin, _skipNormW!, _skipNormB!, Eps);
            lin.Dispose();
        }

        // x += self_attn(norm1(x))
        Tensor n1 = new(flat, DType.F32);
        backend.LayerNorm(n1, x, _n1W!, _n1B!, Eps);
        Tensor a1 = Attention(backend, n1, n1, s, s, _sqW!, _skW!, _svW!, _sqNorm!, _skNorm!, _soW!, _soB!);
        n1.Dispose();
        Tensor x1 = new(flat, DType.F32);
        backend.Add(x1, x, a1);
        x.Dispose();
        a1.Dispose();

        // x += cross_attn(norm2(x), image tokens)
        Tensor n2 = new(flat, DType.F32);
        backend.LayerNorm(n2, x1, _n2W!, _n2B!, Eps);
        Tensor a2 = Attention(backend, n2, context, s, (int)context.Shape[1], _cqW!, _ckW!, _cvW!, _cqNorm!, _ckNorm!, _coW!, _coB!);
        n2.Dispose();
        Tensor x2 = new(flat, DType.F32);
        backend.Add(x2, x1, a2);
        x1.Dispose();
        a2.Dispose();

        // x += mlp_or_moe(norm3(x))
        Tensor n3 = new(flat, DType.F32);
        backend.LayerNorm(n3, x2, _n3W!, _n3B!, Eps);
        Tensor ff = _moe ? Moe(backend, n3, s) : Mlp(backend, n3, _fc1W!, _fc1B!, _fc2W!, _fc2B!, s);
        n3.Dispose();
        Tensor x3 = new(flat, DType.F32);
        backend.Add(x3, x2, ff);
        x2.Dispose();
        ff.Dispose();
        return x3;
    }

    private Tensor Attention(IBackend backend, Tensor q, Tensor kv, int sq, int sk, Tensor qW, Tensor kW, Tensor vW,
        Tensor qNorm, Tensor kNorm, Tensor oW, Tensor oB)
    {
        TensorShape qHeads = new(1, sq, _heads, _headDim), kHeads = new(1, sk, _heads, _headDim);
        Tensor qp = new(qHeads, DType.F32);
        backend.Linear(qp, q, qW, null);
        Tensor kp = new(kHeads, DType.F32);
        backend.Linear(kp, kv, kW, null);
        Tensor vp = new(kHeads, DType.F32);
        backend.Linear(vp, kv, vW, null);
        Tensor qn = new(qHeads, DType.F32);
        backend.RmsNorm(qn, qp, qNorm, Eps);
        qp.Dispose();
        Tensor kn = new(kHeads, DType.F32);
        backend.RmsNorm(kn, kp, kNorm, Eps);
        kp.Dispose();
        Tensor qm = new(new TensorShape(1, _heads, sq, _headDim), DType.F32);
        backend.Permute0213(qm, qn, sq, _heads, _headDim);
        qn.Dispose();
        Tensor km = new(new TensorShape(1, _heads, sk, _headDim), DType.F32);
        backend.Permute0213(km, kn, sk, _heads, _headDim);
        kn.Dispose();
        Tensor vm = new(new TensorShape(1, _heads, sk, _headDim), DType.F32);
        backend.Permute0213(vm, vp, sk, _heads, _headDim);
        vp.Dispose();
        Tensor attn = _attn.AttendHeadMajorOperands(backend, qm, km, vm, default, OperandOwnership.All);
        Tensor merged = _attn.MergeHeads(backend, attn, new TensorShape(1, sq, _width));
        Tensor o = new(new TensorShape(1, sq, _width), DType.F32);
        backend.Linear(o, merged, oW, oB);
        merged.Dispose();
        return o;
    }

    private Tensor Mlp(IBackend backend, Tensor input, Tensor fc1W, Tensor fc1B, Tensor fc2W, Tensor fc2B, int rows)
    {
        int hidden = (int)fc1W.Shape[0];
        Tensor h = new(new TensorShape(1, rows, hidden), DType.F32);
        backend.Linear(h, input, fc1W, fc1B);
        Tensor act = new(h.Shape, DType.F32);
        backend.GeluErf(act, h); // upstream FeedForward uses torch's exact (erf) GELU, not the tanh form
        h.Dispose();
        Tensor o = new(new TensorShape(1, rows, _width), DType.F32);
        backend.Linear(o, act, fc2W, fc2B);
        act.Dispose();
        return o;
    }

    /// <summary>Shared expert on every token plus, per token, its top-k routed experts weighted by the softmax gate scores (not renormalised).</summary>
    internal Tensor Moe(IBackend backend, Tensor input, int s)
    {
        Tensor logits = new(new TensorShape(1, s, _experts), DType.F32);
        backend.Linear(logits, input, _gateW!, null);
        List<int>[] rows = new List<int>[_experts];
        List<float>[] weights = new List<float>[_experts];
        for (int e = 0; e < _experts; e++)
        {
            rows[e] = [];
            weights[e] = [];
        }

        float* lp = (float*)logits.DataPointer;
        float[] scores = new float[_experts];
        int[] chosen = new int[_topK];
        for (int t = 0; t < s; t++)
        {
            float max = float.MinValue, sum = 0f;
            for (int e = 0; e < _experts; e++) max = MathF.Max(max, lp[(long)t * _experts + e]);
            for (int e = 0; e < _experts; e++)
            {
                scores[e] = MathF.Exp(lp[(long)t * _experts + e] - max);
                sum += scores[e];
            }

            for (int e = 0; e < _experts; e++) scores[e] /= sum;
            for (int k = 0; k < _topK; k++)
            {
                int best = -1;
                for (int e = 0; e < _experts; e++)
                {
                    bool taken = false;
                    for (int j = 0; j < k; j++) taken |= chosen[j] == e;
                    if (!taken && (best < 0 || scores[e] > scores[best])) best = e;
                }

                chosen[k] = best;
                rows[best].Add(t);
                weights[best].Add(scores[best]);
            }
        }

        logits.Dispose();
        Tensor output = Mlp(backend, input, _fc1W!, _fc1B!, _fc2W!, _fc2B!, s);
        for (int e = 0; e < _experts; e++)
        {
            int count = rows[e].Count;
            if (count == 0) continue;
            int[] index = [.. rows[e]];
            Tensor gathered = new(new TensorShape(1, count, _width), DType.F32);
            backend.GatherRows(gathered, input, index);
            Tensor y = Mlp(backend, gathered, _expertFc1W![e]!, _expertFc1B![e]!, _expertFc2W![e]!, _expertFc2B![e]!, count);
            gathered.Dispose();
            backend.ScatterAddRows(output, y, index, [.. weights[e]]);
            y.Dispose();
        }

        return output;
    }

    /// <summary>Regroups the rows of <paramref name="parts"/> stacked as one fused projection so that head <c>h</c> reads channels
    /// <c>[h * parts * headDim, (h + 1) * parts * headDim)</c> of the fused row as its <paramref name="parts"/> consecutive groups.</summary>
    internal (Tensor First, Tensor Second, Tensor? Third) Regroup(Tensor a, Tensor b, Tensor? c, int parts)
    {
        Tensor[] sources = c is null ? [a, b] : [a, b, c];
        int rows = (int)a.Shape[0], cols = (int)a.Shape[1];
        long rowBytes = cols * a.DType.SizeInBytes;
        Tensor[] outputs = new Tensor[parts];
        for (int part = 0; part < parts; part++)
        {
            outputs[part] = new Tensor(new TensorShape(rows, cols), a.DType);
            for (int head = 0; head < _heads; head++)
            {
                for (int j = 0; j < _headDim; j++)
                {
                    int fused = head * parts * _headDim + part * _headDim + j;
                    Tensor source = sources[fused / rows];
                    Buffer.MemoryCopy((byte*)source.DataPointer + (long)(fused % rows) * rowBytes,
                        (byte*)outputs[part].DataPointer + ((long)head * _headDim + j) * rowBytes, rowBytes, rowBytes);
                }
            }
        }

        return (outputs[0], outputs[1], parts > 2 ? outputs[2] : null);
    }

    // F32 casts made only to feed Regroup are freed at once instead of waiting for the finalizer (about 1.4 GB over 21 blocks on the CPU path).
    private static void Release(Tensor made, Tensor original)
    {
        if (!ReferenceEquals(made, original))
            made.Dispose();
    }

    // Matrices stay F16 when the backend multiplies F16 weights directly; vectors (biases, norms) are always F32.
    private Tensor F(IReadOnlyDictionary<string, Tensor> w, string key) =>
        _halfWeights && w[key].DType == DType.F16 && w[key].Shape.Rank == 2 ? w[key] : Hunyuan3DShapeVae.F32(w[key]);
}
