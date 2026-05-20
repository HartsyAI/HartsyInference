using SharpInference.Audio.Models.Moonshine;  // RotaryEmbedding (interleaved x_transformers-style)
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.F5Tts;

/// <summary>F5-TTS DiT block. Standard "DiT-XL style" pre-norm transformer with
/// AdaLayerNorm-Zero modulation driven by the timestep embedding. Pattern:
/// <code>
///   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(AdaLNLinear(silu(t_emb)), 6)
///   h = attn(modulate(LayerNorm(x), shift_msa, scale_msa))
///   x = x + gate_msa * h
///   h = ff(modulate(LayerNorm(x), shift_mlp, scale_mlp))
///   x = x + gate_mlp * h
/// </code>
///
/// <para>Attention uses RoPE on Q, K (interleaved / x_transformers convention — reused
/// from <see cref="RotaryEmbedding"/>). FFN is plain GELU MLP with <c>ff_mult=2</c>
/// (smaller than the typical 4x — note this when comparing magnitudes to other DiTs).</para></summary>
internal sealed unsafe class F5DitBlock
{
    private readonly F5TtsConfig _cfg;

    private Tensor? _attnNormLinW, _attnNormLinB;   // [6*dim, dim] / [6*dim]
    private Tensor? _qW, _qB;
    private Tensor? _kW, _kB;
    private Tensor? _vW, _vB;
    private Tensor? _oW, _oB;
    private Tensor? _ffW1, _ffB1;                    // [ff_inner, dim] / [ff_inner]
    private Tensor? _ffW2, _ffB2;                    // [dim, ff_inner] / [dim]

    public F5DitBlock(F5TtsConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _attnNormLinW = WhisperOps.EnsureF32(w[$"{prefix}.attn_norm.linear.weight"]);
        _attnNormLinB = WhisperOps.EnsureF32(w[$"{prefix}.attn_norm.linear.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_q.weight"]);
        _qB = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_q.bias"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_k.weight"]);
        _kB = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_k.bias"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_v.weight"]);
        _vB = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_v.bias"]);
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_out.0.weight"]);
        _oB = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_out.0.bias"]);
        _ffW1 = WhisperOps.EnsureF32(w[$"{prefix}.ff.ff.0.0.weight"]);
        _ffB1 = WhisperOps.EnsureF32(w[$"{prefix}.ff.ff.0.0.bias"]);
        _ffW2 = WhisperOps.EnsureF32(w[$"{prefix}.ff.ff.2.weight"]);
        _ffB2 = WhisperOps.EnsureF32(w[$"{prefix}.ff.ff.2.bias"]);
    }

    /// <summary>Forward: <paramref name="x"/> <c>[1, T, dim]</c> channels-last,
    /// <paramref name="timeEmb"/> <c>[1, dim]</c> per-batch timestep embedding,
    /// <paramref name="ropeCos"/> / <paramref name="ropeSin"/> precomputed tables.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor timeEmb, int t, float[] ropeCos, float[] ropeSin)
    {
        int dim = _cfg.Dim;
        int heads = _cfg.Heads;
        int headDim = _cfg.HeadDim;
        int ffInner = _cfg.FfInner;

        // 1. AdaLN-Zero modulation: project timeEmb through SiLU + Linear → 6*dim.
        Tensor tCopy = new(timeEmb.Shape, DType.F32);
        unsafe
        {
            // SiLU on a copy of timeEmb (don't clobber the caller's tensor).
            float* src = (float*)timeEmb.DataPointer;
            float* dst = (float*)tCopy.DataPointer;
            long n = timeEmb.ElementCount;
            for (long i = 0; i < n; i++) { float vv = src[i]; dst[i] = vv / (1f + MathF.Exp(-vv)); }
        }
        Tensor mods = WhisperOps.ProjectLinear(backend, tCopy, _attnNormLinW!, _attnNormLinB, 1, 1, dim, 6 * dim);
        tCopy.Dispose();

        float* modsP = (float*)mods.DataPointer;
        float* shiftMsa = modsP;
        float* scaleMsa = modsP + dim;
        float* gateMsa = modsP + 2 * dim;
        float* shiftMlp = modsP + 3 * dim;
        float* scaleMlp = modsP + 4 * dim;
        float* gateMlp = modsP + 5 * dim;

        // 2. Pre-norm + modulate for attention: x_n = (LayerNorm(x) with no affine) * (1 + scale_msa) + shift_msa
        //    Our IBackend.LayerNorm requires weight + bias; pre-allocate ones/zeros to act
        //    as the elementwise_affine=False case.
        Tensor onesD = OnesLike(dim);
        Tensor zerosD = ZerosLike(dim);
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, onesD, zerosD, 1e-6f);
        onesD.Dispose(); zerosD.Dispose();

        F5Ops.AdaLnZeroModulate(normed, shiftMsa, scaleMsa, 1, t, dim);

        // 3. Self-attention with RoPE on Q, K.
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW!, _qB, 1, t, dim, dim);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW!, _kB, 1, t, dim, dim);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW!, _vB, 1, t, dim, dim);
        normed.Dispose();

        TensorShape mh = new(1, heads, t, headDim);
        Tensor qMh = new(mh, DType.F32);
        Tensor kMh = new(mh, DType.F32);
        Tensor vMh = new(mh, DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qMh, q, 1, t, heads, headDim);
        WhisperOps.ReshapeToMultiHead4D(kMh, k, 1, t, heads, headDim);
        WhisperOps.ReshapeToMultiHead4D(vMh, v, 1, t, heads, headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        // Full RoPE (rotary_dim == head_dim == 64) on Q and K.
        RotaryEmbedding.ApplyInPlace(qMh, heads, t, headDim, headDim, 0, ropeCos, ropeSin);
        RotaryEmbedding.ApplyInPlace(kMh, heads, t, headDim, headDim, 0, ropeCos, ropeSin);

        float scale = 1f / MathF.Sqrt(headDim);
        Tensor attnOut = new(mh, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, mask: null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor merged = new(x.Shape, DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(merged, attnOut, 1, t, heads, headDim);
        attnOut.Dispose();

        Tensor attnProj = WhisperOps.ProjectLinear(backend, merged, _oW!, _oB, 1, t, dim, dim);
        merged.Dispose();

        // 4. Gated residual: x = x + gate_msa * attn_proj
        Tensor afterAttn = new(x.Shape, DType.F32);
        F5Ops.AdaLnZeroGatedAdd(afterAttn, x, attnProj, gateMsa, 1, t, dim);
        attnProj.Dispose();

        // 5. Pre-norm + modulate for FFN.
        Tensor ones2 = OnesLike(dim);
        Tensor zeros2 = ZerosLike(dim);
        Tensor normed2 = new(afterAttn.Shape, DType.F32);
        backend.LayerNorm(normed2, afterAttn, ones2, zeros2, 1e-6f);
        ones2.Dispose(); zeros2.Dispose();

        F5Ops.AdaLnZeroModulate(normed2, shiftMlp, scaleMlp, 1, t, dim);

        // 6. FF: Linear → GELU → Linear (ff_mult=2).
        Tensor ff1 = WhisperOps.ProjectLinear(backend, normed2, _ffW1!, _ffB1, 1, t, dim, ffInner);
        normed2.Dispose();
        Tensor ffAct = new(ff1.Shape, DType.F32);
        backend.Gelu(ffAct, ff1);
        ff1.Dispose();
        Tensor ff2 = WhisperOps.ProjectLinear(backend, ffAct, _ffW2!, _ffB2, 1, t, ffInner, dim);
        ffAct.Dispose();

        // 7. Gated residual: x = x + gate_mlp * ff_out
        Tensor outX = new(x.Shape, DType.F32);
        F5Ops.AdaLnZeroGatedAdd(outX, afterAttn, ff2, gateMlp, 1, t, dim);
        afterAttn.Dispose();
        ff2.Dispose();
        mods.Dispose();
        return outX;
    }

    private static Tensor OnesLike(int dim)
    {
        Tensor t = new(new TensorShape(dim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < dim; i++) p[i] = 1f;
        return t;
    }

    private static Tensor ZerosLike(int dim)
    {
        Tensor t = new(new TensorShape(dim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < dim; i++) p[i] = 0f;
        return t;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _attnNormLinW, _attnNormLinB,
            _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB,
            _ffW1, _ffB1, _ffW2, _ffB2,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
