using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>ACE-Step v1.5's <c>AceStepAttention</c> — Qwen3-style GQA with per-head q/k RMSNorm, optional split-half
/// RoPE (self-attention only; cross-attention skips it per the reference), and an optional additive mask (the
/// bidirectional ±window sliding mask). One implementation shared by the DiT layers and both condition encoders;
/// projections are bias-less (<c>attention_bias=False</c>).</summary>
internal sealed unsafe class AceStep15Attention
{
    private readonly AceStep15Config _c;

    private Tensor? _qW, _kW, _vW, _oW, _qNorm, _kNorm;

    public AceStep15Attention(AceStep15Config config) => _c = config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _qW = w[$"{prefix}.q_proj.weight"];
        _kW = w[$"{prefix}.k_proj.weight"];
        _vW = w[$"{prefix}.v_proj.weight"];
        _oW = w[$"{prefix}.o_proj.weight"];
        _qNorm = TensorCasts.EnsureF32(w[$"{prefix}.q_norm.weight"]);
        _kNorm = TensorCasts.EnsureF32(w[$"{prefix}.k_norm.weight"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _kW, _vW, _oW, _qNorm, _kNorm })
            if (t is not null) yield return t;
    }

    /// <summary>Attention over <paramref name="x"/> <c>[1, S, H]</c>. Pass <paramref name="crossKv"/> for
    /// cross-attention (keys/values from <c>[1, L, H]</c>, no RoPE); pass RoPE tables for self-attention.</summary>
    // GPU-residency rewrite (mirrors Krea2Attention / the verified Flux/Hunyuan ports): every glue op — head split,
    // per-head QK-RMSNorm, split-half RoPE, GQA K/V repeat, head merge — runs as an IBackend op so the activation
    // chain stays device-resident (the old ReshapeTo/FromMultiHead + host ApplyRopeSplitHalf + host RepeatKvHeads
    // loops D2H-synced every intermediate around every GEMM). Head split = declaring Q/K/V directly as [1, S, H, D]
    // (byte-identical to [1, S, H·D]) so RmsNorm normalizes over headDim with no reshape; RoPE runs on that
    // pre-permute layout via ApplyRopeSingle (broadcast over heads → GQA-correct per tensor); then Permute0213 to
    // [1, H, S, D] for SDPA. Activation dtype follows the input (F16 on the HARTSY_DIT_F16 path; norm/rope tables
    // stay F32). Q and K carry per-head RMSNorm, so pre-softmax scores are bounded and the F16 SDPA path is safe.
    public Tensor Forward(IBackend backend, Tensor x, Tensor? crossKv, float[]? ropeCos, float[]? ropeSin, Tensor? mask)
    {
        int s = (int)x.Shape[1];
        Tensor kvSrc = crossKv ?? x;
        int l = (int)kvSrc.Shape[1];
        int heads = _c.NumHeads, kvHeads = _c.NumKvHeads, hd = _c.HeadDim;
        DType act = x.DType;

        // Q/K/V declared [1, S, H, D] (byte-identical to [1, S, H·D]) so RmsNorm normalizes over headDim (= _qNorm).
        Tensor q = new Tensor(new TensorShape(1, s, heads, hd), act);
        backend.Linear(q, x, _qW!, null);
        Tensor k = new Tensor(new TensorShape(1, l, kvHeads, hd), act);
        backend.Linear(k, kvSrc, _kW!, null);
        Tensor v = new Tensor(new TensorShape(1, l, kvHeads, hd), act);
        backend.Linear(v, kvSrc, _vW!, null);

        Tensor qN = new Tensor(q.Shape, act);
        backend.RmsNorm(qN, q, _qNorm!, _c.RmsNormEps);
        q.Dispose();
        Tensor kN = new Tensor(k.Shape, act);
        backend.RmsNorm(kN, k, _kNorm!, _c.RmsNormEps);
        k.Dispose();

        if (ropeCos is not null && ropeSin is not null)
        {
            // Split-half RoPE on the [1, S, H, D] layout. The half-width host table is expanded to [1, S, headDim]
            // with duplicated halves so ApplyRopeSingle's c[i]/c[i+half] both read the pair angle — bit-identical to
            // the old host ApplyRopeSplitHalf (a·cos − b·sin, b·cos + a·sin over pairs (i, i+half)). Self-attention:
            // l == s, so Q and K share one table.
            Tensor cosDev = ExpandRope(ropeCos, s, hd);
            Tensor sinDev = ExpandRope(ropeSin, s, hd);
            backend.ApplyRopeSingle(qN, cosDev, sinDev, hd);
            backend.ApplyRopeSingle(kN, cosDev, sinDev, hd);
            cosDev.Dispose(); sinDev.Dispose();
        }

        Tensor qMh = new Tensor(new TensorShape(1, heads, s, hd), act);
        backend.Permute0213(qMh, qN, s, heads, hd);
        qN.Dispose();
        Tensor kMh = new Tensor(new TensorShape(1, kvHeads, l, hd), act);
        backend.Permute0213(kMh, kN, l, kvHeads, hd);
        kN.Dispose();
        Tensor vMh = new Tensor(new TensorShape(1, kvHeads, l, hd), act);
        backend.Permute0213(vMh, v, l, kvHeads, hd);
        v.Dispose();

        Tensor kRep = kMh, vRep = vMh;
        if (kvHeads != heads)
        {
            int group = heads / kvHeads;
            kRep = new Tensor(new TensorShape(1, heads, l, hd), act);
            backend.RepeatKvHeads(kRep, kMh, kvHeads, group);
            vRep = new Tensor(new TensorShape(1, heads, l, hd), act);
            backend.RepeatKvHeads(vRep, vMh, kvHeads, group);
            kMh.Dispose(); vMh.Dispose();
        }

        Tensor attn = new Tensor(new TensorShape(1, heads, s, hd), act);
        backend.ScaledDotProductAttention(attn, qMh, kRep, vRep, mask, 1f / MathF.Sqrt(hd), allowF16: act == DType.F16);
        qMh.Dispose(); kRep.Dispose(); vRep.Dispose();

        Tensor attnFlat = new Tensor(new TensorShape(1, s, heads * hd), act);
        backend.Permute0213(attnFlat, attn, heads, s, hd);
        attn.Dispose();
        Tensor projected = new Tensor(new TensorShape(1, s, _c.HiddenSize), act);
        backend.Linear(projected, attnFlat, _oW!, null);
        attnFlat.Dispose();
        return projected;
    }

    /// <summary>Expands a half-width RoPE table <c>[seqLen·headDim/2]</c> to a device tensor <c>[1, seqLen, headDim]</c>
    /// with the second half duplicating the first, so <see cref="IBackend.ApplyRopeSingle"/> (which reads
    /// <c>c[i]</c> and <c>c[i+half]</c>) applies the same pair angle to both elements — matching split-half RoPE.</summary>
    private static Tensor ExpandRope(float[] halfTable, int seqLen, int headDim)
    {
        int half = headDim / 2;
        Tensor t = new Tensor(new TensorShape(1, seqLen, headDim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int s = 0; s < seqLen; s++)
            for (int i = 0; i < half; i++)
            {
                float val = halfTable[s * half + i];
                p[s * headDim + i] = val;
                p[s * headDim + i + half] = val;
            }
        return t;
    }

    /// <summary>Half-dim RoPE tables (<c>cos/sin [seqLen, headDim/2]</c>) for the Llama/Qwen split-half rotation.</summary>
    public static (float[] Cos, float[] Sin) BuildRopeTables(int seqLen, int headDim, double theta)
    {
        int half = headDim / 2;
        float[] cos = new float[seqLen * half];
        float[] sin = new float[seqLen * half];
        for (int p = 0; p < seqLen; p++)
            for (int k = 0; k < half; k++)
            {
                double freq = 1.0 / Math.Pow(theta, (double)(2 * k) / headDim);
                cos[p * half + k] = (float)Math.Cos(p * freq);
                sin[p * half + k] = (float)Math.Sin(p * freq);
            }
        return (cos, sin);
    }

    /// <summary>Bidirectional sliding-window additive mask <c>[1, 1, S, S]</c> (broadcast over heads): 0 where
    /// <c>|i − j| ≤ window</c>, −1e30 elsewhere (the reference masks <c>abs(diff) &gt; sliding_window</c>, non-causal).</summary>
    public static Tensor BuildSlidingMask(int seqLen, int window)
    {
        Tensor mask = new Tensor(new TensorShape(1, 1, seqLen, seqLen), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < seqLen; j++)
                p[(long)i * seqLen + j] = Math.Abs(i - j) <= window ? 0f : -1e30f;
        return mask;
    }
}
