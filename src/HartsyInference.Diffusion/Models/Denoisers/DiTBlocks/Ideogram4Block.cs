using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Ideogram 4 single-stream transformer block (<c>Ideogram4TransformerBlock</c>), ported verbatim from <c>modeling_ideogram4.py</c>.
///
/// <para>Per-block modulation comes from the shared 512-d AdaLN conditioning <c>c</c>: <c>mod = adaln_modulation(c)</c> → <c>chunk(4)</c> = <c>(scale_msa, gate_msa, scale_mlp, gate_mlp)</c>, then <c>gate = tanh(gate)</c>, <c>scale = 1 + scale</c>. **There is NO shift term** — modulation is scale-only.</para>
///
/// <para>Forward (sandwich norms — a norm before AND after each sublayer):</para>
/// <code>
/// attn = attention(attention_norm1(x) * scale_msa)
/// x    = x + gate_msa * attention_norm2(attn)
/// x    = x + gate_mlp * ffn_norm2(feed_forward(ffn_norm1(x) * scale_mlp))
/// </code>
/// Attention is fused-QKV (<c>Linear(hidden → 3*hidden, bias=False)</c>), per-head QK-RMSNorm, 3D MRoPE, then <c>o</c> (bias=False). FFN is SwiGLU <c>w2(silu(w1)·w3)</c>, all bias=False.</summary>
public sealed unsafe class Ideogram4Block
{
    /// <summary>F16-mode damping for the two RmsNorm-sandwiched projections (the Z-Image recipe): the raw
    /// <c>attention.o</c> and SwiGLU outputs can exceed F16's 65504, but both feed straight into a sandwich
    /// RMSNorm and <c>RMSNorm(c·x) ≡ RMSNorm(x)</c> — so scaling the weights via <see cref="Tensor.Fp8ScaleFactor"/>
    /// (folded into the GEMM alpha, zero extra kernels) is bit-exact post-norm.</summary>
    private const float F16SandwichDamp = 1.0f / 64.0f;

    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnHidden;
    private readonly float _eps;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // RMSNorm scales (read as float* — must be F32 at load).
    private Tensor? _attnNorm1;
    private Tensor? _attnNorm2;
    private Tensor? _ffnNorm1;
    private Tensor? _ffnNorm2;

    // Attention: fused QKV + output, all bias=False.
    private Tensor? _qkvWeight;
    private Tensor? _oWeight;

    // SwiGLU FFN, bias=False.
    private Tensor? _w1; // gate
    private Tensor? _w2; // down
    private Tensor? _w3; // up
    private Tensor? _w13; // fused [gate; up] (converter HARTSY_FUSED_FFN=1 — one GEMM, INFERENCE_ACCEL_GRIND §H3.2)

    // AdaLN modulation: Linear(adalnDim → 4*hidden), bias=True.
    private Tensor? _adalnWeight;
    private Tensor? _adalnBias;

    public Ideogram4Block(int hidden, int numHeads, int ffnHidden, float eps)
    {
        if (hidden % numHeads != 0)
            throw new ArgumentException($"hidden {hidden} must be divisible by numHeads {numHeads}.", nameof(hidden));
        _hidden = hidden;
        _numHeads = numHeads;
        _headDim = hidden / numHeads;
        _ffnHidden = ffnHidden;
        _eps = eps;
        _normQ = new QkNorm(_headDim, eps);
        _normK = new QkNorm(_headDim, eps);
    }

    /// <summary>Loads weights using upstream naming. <paramref name="prefix"/> is e.g. <c>"layers.0"</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _attnNorm1 = LoadAsF32(weights, $"{prefix}.attention_norm1.weight");
        _attnNorm2 = LoadAsF32(weights, $"{prefix}.attention_norm2.weight");
        _ffnNorm1 = LoadAsF32(weights, $"{prefix}.ffn_norm1.weight");
        _ffnNorm2 = LoadAsF32(weights, $"{prefix}.ffn_norm2.weight");

        _qkvWeight = weights[$"{prefix}.attention.qkv.weight"];
        _oWeight = weights[$"{prefix}.attention.o.weight"];
        _normQ.LoadWeights(weights[$"{prefix}.attention.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attention.norm_k.weight"]);

        // Fused-FFN path (converter emits w13 = [w1; w3] under HARTSY_FUSED_FFN=1) — else the split pair.
        if (weights.TryGetValue($"{prefix}.feed_forward.w13.weight", out Tensor? w13))
        {
            _w13 = w13;
        }
        else
        {
            _w1 = weights[$"{prefix}.feed_forward.w1.weight"];
            _w3 = weights[$"{prefix}.feed_forward.w3.weight"];
        }
        _w2 = weights[$"{prefix}.feed_forward.w2.weight"];
        // F16 mode: damp the two sandwich-normed projections so their raw outputs fit F16 range (see
        // F16SandwichDamp). attention_norm2 cancels the o damp; ffn_norm2 cancels the w3 damp (it scales
        // silu(w1)·w3 and thus the w2 output linearly). F32 path untouched — baseline stays bit-identical.
        // Fused w13 has ONE scale, so the damp shrinks the gate half too — ForwardSwiGlu un-damps the gate
        // before silu (silu is non-homogeneous; the up half stays damped exactly like the split path).
        if (DitDtype.Act == DType.F16)
        {
            _oWeight.Fp8ScaleFactor *= F16SandwichDamp;
            if (_w3 is not null) _w3.Fp8ScaleFactor *= F16SandwichDamp;
            if (_w13 is not null) _w13.Fp8ScaleFactor *= F16SandwichDamp;
        }

        _adalnWeight = weights[$"{prefix}.adaln_modulation.weight"];
        weights.TryGetValue($"{prefix}.adaln_modulation.bias", out _adalnBias);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_attnNorm1 is not null) yield return _attnNorm1;
        if (_attnNorm2 is not null) yield return _attnNorm2;
        if (_ffnNorm1 is not null) yield return _ffnNorm1;
        if (_ffnNorm2 is not null) yield return _ffnNorm2;
        if (_qkvWeight is not null) yield return _qkvWeight;
        if (_oWeight is not null) yield return _oWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_w1 is not null) yield return _w1;
        if (_w2 is not null) yield return _w2;
        if (_w3 is not null) yield return _w3;
        if (_w13 is not null) yield return _w13;
        if (_adalnWeight is not null) yield return _adalnWeight;
        if (_adalnBias is not null) yield return _adalnBias;
    }

    /// <summary>Forward pass. <paramref name="x"/> is <c>[B, L, hidden]</c>; <paramref name="adalnInput"/> is the shared conditioning <c>[B, adalnDim]</c>; <paramref name="cos"/>/<paramref name="sin"/> are <c>[B, L, headDim]</c>; <paramref name="attentionMask"/> is optional (null = full attention, correct for single-prompt B=1).
    ///
    /// <para>Fully backend-op based: every step is an <see cref="IBackend"/> call so the activation stays
    /// GPU-resident across the whole block (no <c>DataPointer</c> reads, no per-op device syncs).</para></summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor adalnInput, Tensor cos, Tensor sin, Tensor? attentionMask)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hidden);
        long total = (long)batch * seqLen * _hidden;

        // ── Modulation: adaln_modulation(c) → chunk(4) → (1+scale, tanh(gate)) ──
        TensorShape projShape = new TensorShape(batch, 4 * _hidden);
        Tensor proj = new Tensor(projShape, adalnInput.DType);
        backend.Linear(proj, adalnInput, _adalnWeight!, _adalnBias);
        TensorShape modShape = new TensorShape(batch, _hidden);
        Tensor scaleMsa = new Tensor(modShape, DType.F32);
        Tensor gateMsa = new Tensor(modShape, DType.F32);
        Tensor scaleMlp = new Tensor(modShape, DType.F32);
        Tensor gateMlp = new Tensor(modShape, DType.F32);
        backend.ModulationSplit4(scaleMsa, gateMsa, scaleMlp, gateMlp, proj);
        proj.Dispose();

        // ── Attention sublayer: attn = attention(norm1(x) * scale_msa); x += gate_msa * norm2(attn) ──
        Tensor norm1 = new Tensor(shape, x.DType);
        backend.RmsNorm(norm1, x, _attnNorm1!, _eps);
        Tensor mod1 = new Tensor(shape, x.DType);
        backend.AffineBroadcastLastDim(mod1, norm1, scaleMsa, null);
        norm1.Dispose();

        System.Diagnostics.Stopwatch? attnSw = null;
        if (Ideogram4Profile.Enabled) { backend.Sync(); attnSw = System.Diagnostics.Stopwatch.StartNew(); }
        Tensor attn = Attention(backend, mod1, cos, sin, attentionMask, batch, seqLen);
        if (Ideogram4Profile.Enabled) { backend.Sync(); Ideogram4Profile.AttentionMs += attnSw!.Elapsed.TotalMilliseconds; }
        mod1.Dispose();

        Tensor attnNormed = new Tensor(shape, x.DType);
        backend.RmsNorm(attnNormed, attn, _attnNorm2!, _eps);
        attn.Dispose();

        Tensor afterAttn = new Tensor(shape, x.DType);
        backend.GatedResidualLastDim(afterAttn, x, attnNormed, gateMsa);
        attnNormed.Dispose();

        // ── MLP sublayer: x += gate_mlp * norm2(feed_forward(norm1(x) * scale_mlp)) ──
        Tensor norm2 = new Tensor(shape, x.DType);
        backend.RmsNorm(norm2, afterAttn, _ffnNorm1!, _eps);
        Tensor mod2 = new Tensor(shape, x.DType);
        backend.AffineBroadcastLastDim(mod2, norm2, scaleMlp, null);
        norm2.Dispose();

        System.Diagnostics.Stopwatch? mlpSw = null;
        if (Ideogram4Profile.Enabled) { backend.Sync(); mlpSw = System.Diagnostics.Stopwatch.StartNew(); }
        Tensor mlp = ForwardSwiGlu(backend, mod2, batch, seqLen);
        if (Ideogram4Profile.Enabled) { backend.Sync(); Ideogram4Profile.MlpMs += mlpSw!.Elapsed.TotalMilliseconds; }
        mod2.Dispose();

        Tensor mlpNormed = new Tensor(shape, x.DType);
        backend.RmsNorm(mlpNormed, mlp, _ffnNorm2!, _eps);
        mlp.Dispose();

        Tensor result = new Tensor(shape, x.DType);
        backend.GatedResidualLastDim(result, afterAttn, mlpNormed, gateMlp);
        afterAttn.Dispose();
        mlpNormed.Dispose();

        scaleMsa.Dispose();
        gateMsa.Dispose();
        scaleMlp.Dispose();
        gateMlp.Dispose();
        return result;
    }

    /// <summary>Self-attention: fused QKV → per-head QK-RMSNorm → MRoPE → SDPA → output proj. All backend
    /// ops; reshape-to-heads is free (outputs allocated with the consumer's shape, identical byte layout)
    /// and the head permutes reuse <c>Permute0213</c>.</summary>
    private Tensor Attention(IBackend backend, Tensor input, Tensor cos, Tensor sin,
        Tensor? attentionMask, int batch, int seqLen)
    {
        TensorShape flat = new TensorShape(batch, seqLen, _hidden);
        TensorShape heads = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mh = new TensorShape(batch, _numHeads, seqLen, _headDim);
        long headsTotal = (long)batch * seqLen * _hidden;

        Tensor qkv = new Tensor(new TensorShape(batch, seqLen, 3 * _hidden), input.DType);
        backend.Linear(qkv, input, _qkvWeight!, null);

        // Split fused [B, L, 3*hidden] (layout [q|k|v]) directly into [B, L, numHeads, headDim] chunks —
        // the slice is a contiguous gather and the reshape is just the declared output shape.
        Tensor q = new Tensor(heads, input.DType);
        Tensor k = new Tensor(heads, input.DType);
        Tensor v = new Tensor(heads, input.DType);
        backend.SliceLastDim(q, qkv, 0);
        backend.SliceLastDim(k, qkv, _hidden);
        backend.SliceLastDim(v, qkv, 2 * _hidden);
        qkv.Dispose();

        Tensor qN = new Tensor(heads, input.DType);
        Tensor kN = new Tensor(heads, input.DType);
        backend.RmsNorm(qN, q, _normQ.Weight, _normQ.Eps);
        backend.RmsNorm(kN, k, _normK.Weight, _normK.Eps);
        q.Dispose();
        k.Dispose();

        backend.ApplyRope(qN, kN, cos, sin);

        // Permute [B, L, numHeads, headDim] → [B, numHeads, L, headDim] for SDPA.
        Tensor qMh = new Tensor(mh, input.DType);
        Tensor kMh = new Tensor(mh, input.DType);
        Tensor vMh = new Tensor(mh, input.DType);
        backend.Permute0213(qMh, qN, seqLen, _numHeads, _headDim);
        backend.Permute0213(kMh, kN, seqLen, _numHeads, _headDim);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        qN.Dispose();
        kN.Dispose();
        v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mh, input.DType);
        // allowF16: Q/K are per-head RMSNormed above, so scores are bounded — F16 SDPA is range-safe and
        // engages the cuDNN fused flash path (native zero-cast when the block runs F16 activations).
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attentionMask, scale, allowF16: true);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // Permute back [B, numHeads, L, headDim] → [B, L, numHeads, headDim]; declared flat [B, L, hidden].
        Tensor attnFlat = new Tensor(flat, input.DType);
        backend.Permute0213(attnFlat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(flat, input.DType);
        backend.Linear(projected, attnFlat, _oWeight!, null);
        attnFlat.Dispose();
        return projected;
    }

    /// <summary>SwiGLU FFN: <c>w2(silu(w1(x)) * w3(x))</c>, all bias=False. With the fused <c>w13</c>
    /// (HARTSY_FUSED_FFN) the two projections run as ONE GEMM and split via contiguous slices; in F16 mode
    /// the shared damp on w13 is undone on the gate half before silu (see LoadWeights).</summary>
    private Tensor ForwardSwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ff = new TensorShape(batch, seqLen, _ffnHidden);
        Tensor gate;
        Tensor up;
        if (_w13 is not null)
        {
            Tensor both = new Tensor(new TensorShape(batch, seqLen, 2 * _ffnHidden), input.DType);
            backend.Linear(both, input, _w13, null);
            gate = new Tensor(ff, input.DType);
            up = new Tensor(ff, input.DType);
            backend.SliceLastDim(gate, both, 0);
            backend.SliceLastDim(up, both, _ffnHidden);
            both.Dispose();
            if (DitDtype.Act == DType.F16)
            {
                // Undo the shared w13 damp on the gate half (the split path damps only w3).
                Tensor undamped = new Tensor(ff, input.DType);
                backend.Scale(undamped, gate, 1.0f / F16SandwichDamp);
                gate.Dispose();
                gate = undamped;
            }
        }
        else
        {
            gate = new Tensor(ff, input.DType);
            backend.Linear(gate, input, _w1!, null);
            up = new Tensor(ff, input.DType);
            backend.Linear(up, input, _w3!, null);
        }
        Tensor gateAct = new Tensor(ff, input.DType);
        backend.Silu(gateAct, gate);
        gate.Dispose();

        Tensor combined = new Tensor(ff, input.DType);
        backend.Mul(combined, gateAct, up);
        gateAct.Dispose();
        up.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hidden);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, combined, _w2!, null);
        combined.Dispose();
        return output;
    }

    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
