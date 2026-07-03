using HartsyInference.Audio.Models.Moonshine;  // RotaryEmbedding (interleaved x_transformers-style)
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.F5Tts;

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
    /// <paramref name="siluTimeEmb"/> <c>[1, dim]</c> SiLU'd timestep embedding (computed once per step by
    /// the caller), <paramref name="ropeCos"/>/<paramref name="ropeSin"/> <c>[maxPos, headDim]</c> GPU-ready
    /// tables. Fully backend-resident: no activation is touched on the host, so the whole block runs without
    /// a single GPU stall — the previous version did adaLN/RoPE/GELU/reshapes as CPU pointer loops, forcing
    /// D2H+H2D round-trips per op (the reason F5 ran ~60x slower than the reference).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor siluTimeEmb, int t, Tensor ropeCos, Tensor ropeSin)
    {
        int dim = _cfg.Dim;
        int heads = _cfg.Heads;
        int headDim = _cfg.HeadDim;
        int ffInner = _cfg.FfInner;

        // 1. AdaLN-Zero modulation params, kept on-device: Linear(silu(t_emb)) → [1,1,6*dim] → six [1,1,dim]
        //    slices; scales get the "+1" on-device.
        Tensor mods = WhisperOps.ProjectLinear(backend, siluTimeEmb, _attnNormLinW!, _attnNormLinB, 1, 1, dim, 6 * dim);
        Tensor shiftMsa = SliceMod(backend, mods, 0, dim);
        Tensor scaleMsa = SliceMod(backend, mods, 1, dim);
        Tensor gateMsa = SliceMod(backend, mods, 2, dim);
        Tensor shiftMlp = SliceMod(backend, mods, 3, dim);
        Tensor scaleMlp = SliceMod(backend, mods, 4, dim);
        Tensor gateMlp = SliceMod(backend, mods, 5, dim);
        mods.Dispose();
        backend.AddScalar(scaleMsa, scaleMsa, 1f);
        backend.AddScalar(scaleMlp, scaleMlp, 1f);

        // 2. Pre-norm + modulate for attention.
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, x, 1e-6f);
        Tensor modded = new(x.Shape, DType.F32);
        backend.AffineBroadcastLastDim(modded, normed, scaleMsa, shiftMsa);
        normed.Dispose();

        // 3. Self-attention with RoPE on Q, K. The projections write straight into [1, T, heads, headDim]
        //    tensors (shape is metadata to Linear), so RoPE applies in place with no reshape, and the
        //    head transpose for SDPA runs on-device via Permute0213.
        TensorShape packed = new(1, t, heads, headDim);
        Tensor q = new(packed, DType.F32);
        Tensor k = new(packed, DType.F32);
        Tensor v = new(packed, DType.F32);
        backend.Linear(q, modded, _qW!, _qB);
        backend.Linear(k, modded, _kW!, _kB);
        backend.Linear(v, modded, _vW!, _vB);
        modded.Dispose();

        // On-device interleaved RoPE (WanRopeInterleaved has a CUDA kernel; ApplyRopeInterleaved does not).
        // q/k are [1, t, heads, headDim] — flat layout == [t, heads, headDim], which the kernel indexes directly.
        backend.WanRopeInterleaved(q, ropeCos, ropeSin, t, heads, headDim);
        backend.WanRopeInterleaved(k, ropeCos, ropeSin, t, heads, headDim);

        TensorShape mh = new(1, heads, t, headDim);
        Tensor qMh = new(mh, DType.F32);
        Tensor kMh = new(mh, DType.F32);
        Tensor vMh = new(mh, DType.F32);
        backend.Permute0213(qMh, q, t, heads, headDim);
        backend.Permute0213(kMh, k, t, heads, headDim);
        backend.Permute0213(vMh, v, t, heads, headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        // Fused flash attention: at DiT sequence lengths (T ~2000) the naive SDPA materializes a
        // [heads, T, T] score tensor (~256MB/block) — pure DRAM-bandwidth burn; flash never does.
        float scale = 1f / MathF.Sqrt(headDim);
        Tensor attnOut = new(mh, DType.F32);
        backend.FlashAttention(attnOut, qMh, kMh, vMh, kvLen: t, kvGroup: 1, causal: false, qOffset: 0, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        // [1, heads, T, headDim] → [1, T, heads*headDim] (inverse transpose, still on-device).
        Tensor merged = new(x.Shape, DType.F32);
        backend.Permute0213(merged, attnOut, heads, t, headDim);
        attnOut.Dispose();

        Tensor attnProj = WhisperOps.ProjectLinear(backend, merged, _oW!, _oB, 1, t, dim, dim);
        merged.Dispose();

        // 4. Gated residual: x = x + gate_msa * attn_proj
        Tensor afterAttn = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(afterAttn, x, attnProj, gateMsa);
        attnProj.Dispose();

        // 5. Pre-norm + modulate for FFN.
        Tensor normed2 = new(afterAttn.Shape, DType.F32);
        backend.LayerNormNoAffine(normed2, afterAttn, 1e-6f);
        Tensor modded2 = new(afterAttn.Shape, DType.F32);
        backend.AffineBroadcastLastDim(modded2, normed2, scaleMlp, shiftMlp);
        normed2.Dispose();

        // 6. FF: Linear → tanh-GELU (the backend's Gelu IS the tanh approximation) → Linear (ff_mult=2).
        Tensor ff1 = WhisperOps.ProjectLinear(backend, modded2, _ffW1!, _ffB1, 1, t, dim, ffInner);
        modded2.Dispose();
        backend.Gelu(ff1, ff1);
        Tensor ff2 = WhisperOps.ProjectLinear(backend, ff1, _ffW2!, _ffB2, 1, t, ffInner, dim);
        ff1.Dispose();

        // 7. Gated residual: x = x + gate_mlp * ff_out
        Tensor outX = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(outX, afterAttn, ff2, gateMlp);
        afterAttn.Dispose();
        ff2.Dispose();
        shiftMsa.Dispose(); scaleMsa.Dispose(); gateMsa.Dispose();
        shiftMlp.Dispose(); scaleMlp.Dispose(); gateMlp.Dispose();
        return outX;
    }

    /// <summary>On-device slice of modulation chunk <paramref name="idx"/> (a <c>[1, 1, dim]</c> tensor) out of
    /// the packed <c>[1, 1, 6*dim]</c> projection.</summary>
    private static Tensor SliceMod(IBackend backend, Tensor mods, int idx, int dim)
    {
        Tensor slice = new(new TensorShape(1, 1, dim), DType.F32);
        backend.SliceLastDim(slice, mods, idx * dim);
        return slice;
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
