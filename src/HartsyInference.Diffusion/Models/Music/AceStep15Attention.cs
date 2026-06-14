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
        _qNorm = EnsureF32(w[$"{prefix}.q_norm.weight"]);
        _kNorm = EnsureF32(w[$"{prefix}.k_norm.weight"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _kW, _vW, _oW, _qNorm, _kNorm })
            if (t is not null) yield return t;
    }

    /// <summary>Attention over <paramref name="x"/> <c>[1, S, H]</c>. Pass <paramref name="crossKv"/> for
    /// cross-attention (keys/values from <c>[1, L, H]</c>, no RoPE); pass RoPE tables for self-attention.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor? crossKv, float[]? ropeCos, float[]? ropeSin, Tensor? mask)
    {
        int s = (int)x.Shape[1];
        Tensor kvSrc = crossKv ?? x;
        int l = (int)kvSrc.Shape[1];
        int heads = _c.NumHeads, kvHeads = _c.NumKvHeads, hd = _c.HeadDim;

        Tensor qFlat = new Tensor(new TensorShape(1, s, heads * hd), DType.F32);
        Tensor kFlat = new Tensor(new TensorShape(1, l, kvHeads * hd), DType.F32);
        Tensor vFlat = new Tensor(new TensorShape(1, l, kvHeads * hd), DType.F32);
        backend.Linear(qFlat, x, _qW!, null);
        backend.Linear(kFlat, kvSrc, _kW!, null);
        backend.Linear(vFlat, kvSrc, _vW!, null);

        Tensor qMh = DiTUtils.ReshapeToMultiHead(qFlat, 1, s, heads, hd);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(kFlat, 1, l, kvHeads, hd);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(vFlat, 1, l, kvHeads, hd);
        qFlat.Dispose(); kFlat.Dispose(); vFlat.Dispose();

        Tensor qNormed = new Tensor(qMh.Shape, DType.F32);
        Tensor kNormed = new Tensor(kMh.Shape, DType.F32);
        backend.RmsNorm(qNormed, qMh, _qNorm!, _c.RmsNormEps);
        backend.RmsNorm(kNormed, kMh, _kNorm!, _c.RmsNormEps);
        qMh.Dispose(); kMh.Dispose();

        if (ropeCos is not null && ropeSin is not null)
        {
            ApplyRopeSplitHalf(qNormed, ropeCos, ropeSin, heads, s, hd);
            ApplyRopeSplitHalf(kNormed, ropeCos, ropeSin, kvHeads, l, hd);
        }

        Tensor kRep = kNormed, vRep = vMh;
        if (kvHeads != heads)
        {
            int group = heads / kvHeads;
            kRep = RepeatKvHeads(kNormed, kvHeads, group, l, hd);
            vRep = RepeatKvHeads(vMh, kvHeads, group, l, hd);
            kNormed.Dispose(); vMh.Dispose();
        }

        Tensor attn = new Tensor(new TensorShape(1, heads, s, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qNormed, kRep, vRep, mask, 1f / MathF.Sqrt(hd));
        qNormed.Dispose(); kRep.Dispose(); vRep.Dispose();

        Tensor merged = DiTUtils.ReshapeFromMultiHead(attn, 1, s, heads, hd);
        attn.Dispose();
        Tensor projected = new Tensor(new TensorShape(1, s, _c.HiddenSize), DType.F32);
        backend.Linear(projected, merged, _oW!, null);
        merged.Dispose();
        return projected;
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

    private static void ApplyRopeSplitHalf(Tensor x, float[] cos, float[] sin, int heads, int seqLen, int headDim)
    {
        int half = headDim / 2;
        float* xp = (float*)x.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < seqLen; s++)
            {
                long off = ((long)h * seqLen + s) * headDim;
                int posOff = s * half;
                for (int i = 0; i < half; i++)
                {
                    float c = cos[posOff + i], si = sin[posOff + i];
                    float a = xp[off + i], b = xp[off + i + half];
                    xp[off + i] = a * c - b * si;
                    xp[off + i + half] = b * c + a * si;
                }
            }
    }

    private static Tensor RepeatKvHeads(Tensor input, int kvHeads, int groupSize, int seqLen, int headDim)
    {
        Tensor output = new Tensor(new TensorShape(1, kvHeads * groupSize, seqLen, headDim), DType.F32);
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long perHead = (long)seqLen * headDim;
        for (int h = 0; h < kvHeads; h++)
            for (int g = 0; g < groupSize; g++)
                Buffer.MemoryCopy(inPtr + h * perHead, outPtr + ((long)h * groupSize + g) * perHead,
                    perHead * 4, perHead * 4);
        return output;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
