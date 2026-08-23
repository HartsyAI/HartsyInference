using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>ERNIE-Image single-stream block (<c>ErnieImageSharedAdaLNBlock</c> in diffusers). Identical structure for every layer; the 6-vector AdaLN modulation is computed ONCE at the transformer level and broadcast into every block (the "shared AdaLN" architecture choice — saves ~300 MB of per-block linear weights vs Flux).
///
/// Sub-block order (per <c>transformer_ernie_image.py:257-275</c>):
/// <list type="number">
///   <item>RMSNorm (<c>adaLN_sa_ln</c>) → <c>x*(1 + scale_msa) + shift_msa</c>.</item>
///   <item>Self-attention with separate Q/K/V projections (NO fused QKV), QK-RMSNorm, 3D RoPE, all linears <c>bias=False</c>.</item>
///   <item>Gated residual: <c>x = residual + gate_msa * attn_out</c>.</item>
///   <item>RMSNorm (<c>adaLN_mlp_ln</c>) → <c>x*(1 + scale_mlp) + shift_mlp</c>.</item>
///   <item>GELU-gated FFN (NOT SwiGLU): <c>linear_fc2(up_proj(x) * gelu(gate_proj(x)))</c>. All linears <c>bias=False</c>.</item>
///   <item>Gated residual: <c>x = residual + gate_mlp * mlp_out</c>.</item>
/// </list>
///
/// Internal layout is <c>[B, S, H]</c> (the Python code internally permutes to <c>[S, B, H]</c> at block boundaries for legacy Megatron parity, but our backend SDPA expects <c>[B, S, H]</c> so we skip the round-trip).</summary>
public sealed unsafe class ErnieImageBlock
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnHidden;
    private readonly float _eps;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // RMSNorm scales — read as float* directly, must be F32 at load time.
    private Tensor? _adaLnSaWeight;
    private Tensor? _adaLnMlpWeight;

    // Attention linears — separate Q/K/V/Out, all bias=False.
    private Tensor? _toQWeight;
    private Tensor? _toKWeight;
    private Tensor? _toVWeight;
    private Tensor? _toOutWeight;

    // GELU-gated FFN linears — bias=False.
    private Tensor? _gateProjWeight;
    private Tensor? _upProjWeight;
    private Tensor? _linearFc2Weight;

    public ErnieImageBlock(int hidden, int numHeads, int ffnHidden, float eps = 1e-6f)
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

    /// <summary>Loads weights using diffusers naming. <paramref name="prefix"/> is e.g. <c>"layers.0"</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // RMSNorm scales must be F32 (CudaBackend.RmsNorm + CPU CastTo paths read float* directly).
        _adaLnSaWeight = TensorCasts.LoadF32(weights, $"{prefix}.adaLN_sa_ln.weight");
        _adaLnMlpWeight = TensorCasts.LoadF32(weights, $"{prefix}.adaLN_mlp_ln.weight");

        _toQWeight = weights[$"{prefix}.self_attention.to_q.weight"];
        _toKWeight = weights[$"{prefix}.self_attention.to_k.weight"];
        _toVWeight = weights[$"{prefix}.self_attention.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.self_attention.to_out.0.weight"];

        _normQ.LoadWeights(weights[$"{prefix}.self_attention.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.self_attention.norm_k.weight"]);

        _gateProjWeight = weights[$"{prefix}.mlp.gate_proj.weight"];
        _upProjWeight = weights[$"{prefix}.mlp.up_proj.weight"];
        _linearFc2Weight = weights[$"{prefix}.mlp.linear_fc2.weight"];
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_adaLnSaWeight is not null) yield return _adaLnSaWeight;
        if (_adaLnMlpWeight is not null) yield return _adaLnMlpWeight;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toOutWeight is not null) yield return _toOutWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_gateProjWeight is not null) yield return _gateProjWeight;
        if (_upProjWeight is not null) yield return _upProjWeight;
        if (_linearFc2Weight is not null) yield return _linearFc2Weight;
    }

    /// <summary>Forward pass with the shared 6-vector modulation broadcast in.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="x">Token sequence <c>[B, S, hidden]</c>.</param>
    /// <param name="shiftMsa">Attn shift, shape <c>[B, hidden]</c>.</param>
    /// <param name="scaleMsa">Attn scale, shape <c>[B, hidden]</c>.</param>
    /// <param name="gateMsa">Attn output gate, shape <c>[B, hidden]</c>.</param>
    /// <param name="shiftMlp">MLP shift.</param>
    /// <param name="scaleMlp">MLP scale.</param>
    /// <param name="gateMlp">MLP output gate.</param>
    /// <param name="ropeCos">RoPE cos table <c>[B, S, head_dim]</c> (pre-sliced once at the transformer level from <see cref="ErnieImageRope.BuildFreqs"/>'s packed output — was re-sliced per block).</param>
    /// <param name="ropeSin">RoPE sin table <c>[B, S, head_dim]</c>.</param>
    /// <param name="attentionMask">Optional attention mask <c>[B, 1, 1, S]</c> — bool-style, where 0=mask out.</param>
    public Tensor Forward(IBackend backend, Tensor x,
        Tensor shiftMsa, Tensor scaleMsa, Tensor gateMsa,
        Tensor shiftMlp, Tensor scaleMlp, Tensor gateMlp,
        Tensor ropeCos, Tensor ropeSin, Tensor? attentionMask)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hidden);
        TensorShape headsShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // ── Attention sub-block ────────────────────────────────────────────
        // GPU-residency rewrite (mirrors the verified Krea2Block / FluxSingleStreamBlock ports): the two AdaLN
        // (1+scale)·norm+shift affines, the two gated residuals, the head-split/merge reshapes and the QK-norm all
        // run as IBackend ops so the activation chain stays device-resident — no per-op DataPointer D2H sync barriers
        // around the attention/FFN GEMMs. ApplyShiftScale → DiTUtils.Modulate; ApplyGatedResidual →
        // GatedResidualLastDim; the ReshapeFlatToHeads/PermuteBshdToBhsd/PermuteBhsdToBsh host loops →
        // Permute0213 (+ declaring Q/K/V directly as [B, S, H, D], byte-identical to [B, S, hidden]). RoPE is now
        // GPU-resident too: the non-interleaved Megatron "rotate_half" (pairs (i, i+halfDim)) is applied via
        // backend.ApplyRope (which computes exactly x·cos + rotate_half(x)·sin) after slicing the packed freqs into
        // cos/sin on-device — see ErnieImageRope.ApplyRotaryEmbGpu. The block forward is fully device-resident.
        Tensor norm1 = new Tensor(shape, DType.F32);
        backend.RmsNorm(norm1, x, _adaLnSaWeight!, _eps);
        Tensor modulated = DiTUtils.Modulate(backend, norm1, shiftMsa, scaleMsa, shape); // x·(1+scale_msa) + shift_msa
        norm1.Dispose();

        // Separate Q, K, V projections; bias=False. Q/K/V declared directly as [B, S, H, D] (byte-identical to
        // [B, S, hidden]) so RmsNorm normalizes over headDim and Permute0213 runs with no explicit reshape.
        Tensor qHeads = new Tensor(headsShape, DType.F32);
        Tensor kHeads = new Tensor(headsShape, DType.F32);
        Tensor v = new Tensor(headsShape, DType.F32);
        backend.Linear(qHeads, modulated, _toQWeight!, null);
        backend.Linear(kHeads, modulated, _toKWeight!, null);
        backend.Linear(v, modulated, _toVWeight!, null);
        modulated.Dispose();

        // QK-RMSNorm over the last dim (headDim); weight already F32. Identical to the old QkNorm host loop.
        Tensor qNormed = new Tensor(headsShape, DType.F32);
        Tensor kNormed = new Tensor(headsShape, DType.F32);
        backend.RmsNorm(qNormed, qHeads, _normQ.Weight, _normQ.Eps);
        backend.RmsNorm(kNormed, kHeads, _normK.Weight, _normK.Eps);
        qHeads.Dispose();
        kHeads.Dispose();

        // 3D RoPE (in-place on qNormed/kNormed, both still [B, S, numHeads, headDim]) — GPU-resident via
        // backend.ApplyRope (rotate_half) on the pre-sliced cos/sin tables. Bit-identical to the host path.
        backend.ApplyRope(qNormed, kNormed, ropeCos, ropeSin);

        // SDPA expects [B, numHeads, S, headDim]. Permute (B, S, H, D) → (B, H, S, D).
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(qMh, qNormed, seqLen, _numHeads, _headDim);
        backend.Permute0213(kMh, kNormed, seqLen, _numHeads, _headDim);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        // allowF16: QK-RMS-norm bounds the scores, and the [B,1,Sq,Skv] F32 padding mask rides the cuDNN fused
        // engine's fp32 bias path (added to fp32 scores inside the engine, never rounded through F16) — the
        // proven Chroma/Z-Image config. Falls back to the materialized F32 path when cuDNN is unavailable.
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attentionMask, scale, allowF16: true);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // [B, H, S, D] → [B, S, hidden]
        Tensor attnFlat = new Tensor(shape, DType.F32);
        backend.Permute0213(attnFlat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, DType.F32);
        backend.Linear(projected, attnFlat, _toOutWeight!, null);
        attnFlat.Dispose();

        Tensor afterAttn = new Tensor(shape, DType.F32);
        backend.GatedResidualLastDim(afterAttn, x, projected, gateMsa);   // x + gate_msa·attn_out
        projected.Dispose();

        // ── MLP sub-block ──────────────────────────────────────────────────
        Tensor norm2 = new Tensor(shape, DType.F32);
        backend.RmsNorm(norm2, afterAttn, _adaLnMlpWeight!, _eps);
        Tensor modulated2 = DiTUtils.Modulate(backend, norm2, shiftMlp, scaleMlp, shape); // x·(1+scale_mlp) + shift_mlp
        norm2.Dispose();

        Tensor mlpOut = ForwardGeluGatedFfn(backend, modulated2, batch, seqLen);
        modulated2.Dispose();

        Tensor result = new Tensor(shape, DType.F32);
        backend.GatedResidualLastDim(result, afterAttn, mlpOut, gateMlp);  // afterAttn + gate_mlp·mlp_out
        afterAttn.Dispose();
        mlpOut.Dispose();

        return result;
    }

    /// <summary>GELU-gated FFN: <c>linear_fc2(up_proj(x) * gelu(gate_proj(x)))</c>. Note the multiplicative order — diffusers uses <c>up_proj(x) * gelu(gate_proj(x))</c>, NOT <c>gelu(gate_proj(x)) * up_proj(x)</c>. Float-mul is commutative so the result is identical, but matching layout simplifies dump-diffing.</summary>
    private Tensor ForwardGeluGatedFfn(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnHidden);

        Tensor gate = new Tensor(ffShape, input.DType);
        backend.Linear(gate, input, _gateProjWeight!, null);
        Tensor gateActivated = new Tensor(ffShape, input.DType);
        backend.Gelu(gateActivated, gate);
        gate.Dispose();

        Tensor up = new Tensor(ffShape, input.DType);
        backend.Linear(up, input, _upProjWeight!, null);

        Tensor combined = new Tensor(ffShape, input.DType);
        backend.Mul(combined, up, gateActivated);
        up.Dispose();
        gateActivated.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hidden);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, combined, _linearFc2Weight!, null);
        combined.Dispose();

        return output;
    }
}
