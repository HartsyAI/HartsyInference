using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Anima / Cosmos-Predict2 DiT transformer block. Three sub-blocks (self-attn, cross-attn, FFN), each gated by
/// its own AdaLN-LoRA modulator. All weight shapes verified against the actual Anima checkpoint
/// <c>anima-preview3-base.safetensors</c>:
///
/// <para><b>AdaLN-LoRA modulators:</b> Each block has three independent modulators
/// (<c>adaln_modulation_{self_attn,cross_attn,mlp}</c>). The forward is:</para>
/// <code>
/// y1 = Linear[256, 2048] @ silu(embedded_timestep)   # [B, 256]   — bottleneck
/// y2 = Linear[6144, 256] @ y1                        # [B, 6144]  — expand to 3*hidden
/// y = y2 + temb[:6144]                               # add main temb slice
/// shift, scale, gate = chunk(y, 3, dim=-1)           # each [B, 2048]
/// x = norm(x) * (1 + scale) + shift                  # LayerNorm-no-affine, then modulate
/// </code>
/// No biases on any AdaLN linear.
///
/// <para><b>Self-attention</b>: standard SDPA with QK-RMSNorm (per-head_dim scale length 128) and 3D RoPE on Q/K.
/// All projections (Q, K, V, output) are <c>[2048, 2048]</c>, no biases.</para>
///
/// <para><b>Cross-attention</b>: Q from hidden (<c>[2048, 2048]</c>), K/V from <see cref="AnimaLlmAdapter"/> output
/// (<c>[2048, 1024]</c>) — the LlmAdapter emits 1024-dim refined Qwen-3 features, and the DiT consumes them via a
/// rectangular K/V projection. Output projection <c>[2048, 2048]</c>. Per-head_dim QK-RMSNorm (length 128). No biases.</para>
///
/// <para><b>FFN</b>: <c>Linear[8192, 2048] → GELU → Linear[2048, 8192]</c>, named <c>mlp.layer1</c> / <c>mlp.layer2</c>.
/// No biases.</para></summary>
public sealed unsafe class AnimaBlock
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnHidden;
    private readonly int _adaLnRank;
    private readonly int _adaLnExpandWidth;  // = 3 * hidden, the post-expand width of the AdaLN-LoRA output.
    private readonly float _eps;
    private readonly float _qkEps;

    private readonly QkNorm _selfQNorm;
    private readonly QkNorm _selfKNorm;
    private readonly QkNorm _crossQNorm;
    private readonly QkNorm _crossKNorm;

    // AdaLN-LoRA modulators (3 per block: self-attn, cross-attn, mlp). Each has:
    //   .1.weight [rank=256, hidden=2048]  — bottleneck projection
    //   .2.weight [3*hidden=6144, rank=256] — expand projection
    // No biases.
    private Tensor? _adaSelfL1, _adaSelfL2;
    private Tensor? _adaCrossL1, _adaCrossL2;
    private Tensor? _adaMlpL1, _adaMlpL2;

    // Self-attention projections.
    private Tensor? _selfQ, _selfK, _selfV, _selfOut;

    // Cross-attention projections.
    private Tensor? _crossQ;          // [hidden, hidden]
    private Tensor? _crossK, _crossV; // [hidden, kvDim] where kvDim is the LlmAdapter hidden (1024).
    private Tensor? _crossOut;        // [hidden, hidden]

    // MLP (no biases).
    private Tensor? _mlp1, _mlp2;

    private int _kvDim;  // Inferred from cross_attn.k_proj weight shape at LoadWeights time.

    public AnimaBlock(int hidden, int numHeads, int headDim, int ffnHidden, int adaLnRank, float eps = 1e-6f, float qkEps = 1e-6f)
    {
        if (hidden != numHeads * headDim)
            throw new ArgumentException(
                $"hidden {hidden} != numHeads {numHeads} × headDim {headDim} ({numHeads * headDim}).", nameof(hidden));
        _hidden = hidden;
        _numHeads = numHeads;
        _headDim = headDim;
        _ffnHidden = ffnHidden;
        _adaLnRank = adaLnRank;
        _adaLnExpandWidth = 3 * hidden;
        _eps = eps;
        _qkEps = qkEps;
        _selfQNorm = new QkNorm(headDim, qkEps);
        _selfKNorm = new QkNorm(headDim, qkEps);
        _crossQNorm = new QkNorm(headDim, qkEps);
        _crossKNorm = new QkNorm(headDim, qkEps);
    }

    /// <summary>Loads block weights from the converter-stripped DiT trunk bucket. <paramref name="prefix"/>
    /// is e.g. <c>blocks.0</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // ── AdaLN-LoRA modulators (3 per block) ──
        _adaSelfL1 = weights[$"{prefix}.adaln_modulation_self_attn.1.weight"];
        _adaSelfL2 = weights[$"{prefix}.adaln_modulation_self_attn.2.weight"];
        _adaCrossL1 = weights[$"{prefix}.adaln_modulation_cross_attn.1.weight"];
        _adaCrossL2 = weights[$"{prefix}.adaln_modulation_cross_attn.2.weight"];
        _adaMlpL1 = weights[$"{prefix}.adaln_modulation_mlp.1.weight"];
        _adaMlpL2 = weights[$"{prefix}.adaln_modulation_mlp.2.weight"];

        // ── Self-attention ──
        _selfQ = weights[$"{prefix}.self_attn.q_proj.weight"];
        _selfK = weights[$"{prefix}.self_attn.k_proj.weight"];
        _selfV = weights[$"{prefix}.self_attn.v_proj.weight"];
        _selfOut = weights[$"{prefix}.self_attn.output_proj.weight"];
        _selfQNorm.LoadWeights(weights[$"{prefix}.self_attn.q_norm.weight"]);
        _selfKNorm.LoadWeights(weights[$"{prefix}.self_attn.k_norm.weight"]);

        // ── Cross-attention ──
        _crossQ = weights[$"{prefix}.cross_attn.q_proj.weight"];
        _crossK = weights[$"{prefix}.cross_attn.k_proj.weight"];
        _crossV = weights[$"{prefix}.cross_attn.v_proj.weight"];
        _crossOut = weights[$"{prefix}.cross_attn.output_proj.weight"];
        _crossQNorm.LoadWeights(weights[$"{prefix}.cross_attn.q_norm.weight"]);
        _crossKNorm.LoadWeights(weights[$"{prefix}.cross_attn.k_norm.weight"]);

        // K/V input dim is inferred from the K-projection weight: shape is [out_features=hidden, in_features=kvDim].
        _kvDim = (int)_crossK.Shape[1];

        // ── MLP ──
        _mlp1 = weights[$"{prefix}.mlp.layer1.weight"];
        _mlp2 = weights[$"{prefix}.mlp.layer2.weight"];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_adaSelfL1 is not null) yield return _adaSelfL1;
        if (_adaSelfL2 is not null) yield return _adaSelfL2;
        if (_adaCrossL1 is not null) yield return _adaCrossL1;
        if (_adaCrossL2 is not null) yield return _adaCrossL2;
        if (_adaMlpL1 is not null) yield return _adaMlpL1;
        if (_adaMlpL2 is not null) yield return _adaMlpL2;
        if (_selfQ is not null) yield return _selfQ;
        if (_selfK is not null) yield return _selfK;
        if (_selfV is not null) yield return _selfV;
        if (_selfOut is not null) yield return _selfOut;
        foreach (Tensor w in _selfQNorm.EnumerateWeights()) yield return w;
        foreach (Tensor w in _selfKNorm.EnumerateWeights()) yield return w;
        if (_crossQ is not null) yield return _crossQ;
        if (_crossK is not null) yield return _crossK;
        if (_crossV is not null) yield return _crossV;
        if (_crossOut is not null) yield return _crossOut;
        foreach (Tensor w in _crossQNorm.EnumerateWeights()) yield return w;
        foreach (Tensor w in _crossKNorm.EnumerateWeights()) yield return w;
        if (_mlp1 is not null) yield return _mlp1;
        if (_mlp2 is not null) yield return _mlp2;
    }

    /// <summary>Runs self-attention, cross-attention, then MLP, each residual-gated by its own AdaLN-LoRA modulator.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="hidden">Image token sequence <c>[B, S, hidden]</c>.</param>
    /// <param name="encoderHidden">Refined text features from <see cref="AnimaLlmAdapter"/>, <c>[B, T, kvDim=1024]</c>.</param>
    /// <param name="embeddedTimestep">RMSNorm'd sin/cos vector <c>[B, 2048]</c> — input to the AdaLN-LoRA bottleneck.</param>
    /// <param name="temb">Processed timestep <c>[B, 6144]</c> — added to the AdaLN-LoRA expand output before chunking.</param>
    /// <param name="ropeCos">RoPE cos table <c>[S, headDim]</c> for image self-attn.</param>
    /// <param name="ropeSin">RoPE sin table <c>[S, headDim]</c> for image self-attn.</param>
    /// <param name="rope">RoPE applier (re-used across blocks).</param>
    /// <param name="crossAttentionMask">Optional cross-attn mask <c>[B, 1, 1, T]</c>.</param>
    // GPU-residency rewrite (mirrors the verified Krea2Block / Krea2Attention ports): every glue op — the
    // AdaLN-LoRA temb add + 3-way chunk, the LayerNorm+modulate, the head split/merge, the per-head QK-RMSNorm,
    // the rotary, and the gated residuals — now runs as an IBackend op, so the activation chain stays
    // device-resident. The previous host `float*` loops D2H-synced (and freed) every Linear/SDPA output around
    // every GEMM: 14 pipeline drains per block × 28 blocks × 2 CFG passes = 792 D2H syncs per denoise step.
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor encoderHidden,
        Tensor embeddedTimestep, Tensor temb,
        Tensor ropeCos, Tensor ropeSin, AnimaRope rope,
        Tensor? crossAttentionMask)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        int textSeq = (int)encoderHidden.Shape[1];
        TensorShape hiddenShape = new TensorShape(batch, seqLen, _hidden);

        // ── 1. AdaLN-LoRA #1 + self-attention ──
        (Tensor norm1, Tensor gate1) = ApplyAdaLnLora(backend, hidden, embeddedTimestep, temb,
            _adaSelfL1!, _adaSelfL2!, batch, seqLen);
        Tensor attn1Out = SelfAttention(backend, norm1, ropeCos, ropeSin, rope, batch, seqLen);
        norm1.Dispose();
        Tensor afterAttn1 = new Tensor(hiddenShape, DType.F32);
        backend.GatedResidualLastDim(afterAttn1, hidden, attn1Out, gate1);
        attn1Out.Dispose();
        gate1.Dispose();

        // ── 2. AdaLN-LoRA #2 + cross-attention ──
        (Tensor norm2, Tensor gate2) = ApplyAdaLnLora(backend, afterAttn1, embeddedTimestep, temb,
            _adaCrossL1!, _adaCrossL2!, batch, seqLen);
        Tensor attn2Out = CrossAttention(backend, norm2, encoderHidden, crossAttentionMask, batch, seqLen, textSeq);
        norm2.Dispose();
        Tensor afterAttn2 = new Tensor(hiddenShape, DType.F32);
        backend.GatedResidualLastDim(afterAttn2, afterAttn1, attn2Out, gate2);
        attn2Out.Dispose();
        afterAttn1.Dispose();
        gate2.Dispose();

        // ── 3. AdaLN-LoRA #3 + MLP ──
        (Tensor norm3, Tensor gate3) = ApplyAdaLnLora(backend, afterAttn2, embeddedTimestep, temb,
            _adaMlpL1!, _adaMlpL2!, batch, seqLen);
        Tensor ffOut = FeedForward(backend, norm3, batch, seqLen);
        norm3.Dispose();
        Tensor result = new Tensor(hiddenShape, DType.F32);
        backend.GatedResidualLastDim(result, afterAttn2, ffOut, gate3);
        afterAttn2.Dispose();
        ffOut.Dispose();
        gate3.Dispose();

        return result;
    }

    /// <summary>AdaLN-LoRA modulation: silu(embedded_timestep) → Linear(rank=256) → Linear(3*hidden=6144) → + temb[:6144] →
    /// chunk(3) → (shift, scale, gate). Applies <c>norm(x) * (1 + scale) + shift</c> and returns the modulated tensor
    /// + the gate (used for the gated residual).</summary>
    private (Tensor Modulated, Tensor Gate) ApplyAdaLnLora(IBackend backend, Tensor x,
        Tensor embeddedTimestep, Tensor temb,
        Tensor linRankWeight, Tensor linExpandWeight,
        int batch, int seqLen)
    {
        TensorShape embShape = new TensorShape(batch, _hidden);
        TensorShape rankShape = new TensorShape(batch, _adaLnRank);
        TensorShape expandShape = new TensorShape(batch, _adaLnExpandWidth);

        // silu(embedded_timestep) [B, 2048]
        Tensor act = new Tensor(embShape, DType.F32);
        backend.Silu(act, embeddedTimestep);

        // bottleneck Linear[256, 2048] → [B, 256]
        Tensor rank = new Tensor(rankShape, DType.F32);
        backend.Linear(rank, act, linRankWeight, null);
        act.Dispose();

        // expand Linear[6144, 256] → [B, 6144]
        Tensor expanded = new Tensor(expandShape, DType.F32);
        backend.Linear(expanded, rank, linExpandWeight, null);
        rank.Dispose();

        // + temb[:6144] (broadcast over batch — temb is [B, 6144], so element-wise add)
        Tensor modulation = AddTembSlice(backend, expanded, temb, batch, _adaLnExpandWidth);
        expanded.Dispose();

        // chunk(3) → shift, scale, gate, each [B, hidden]
        TensorShape chunkShape = new TensorShape(batch, _hidden);
        Tensor shift = new Tensor(chunkShape, DType.F32);
        Tensor scale = new Tensor(chunkShape, DType.F32);
        Tensor gate = new Tensor(chunkShape, DType.F32);
        backend.SliceLastDim(shift, modulation, 0);
        backend.SliceLastDim(scale, modulation, _hidden);
        backend.SliceLastDim(gate, modulation, 2 * _hidden);
        modulation.Dispose();

        // LayerNorm-no-affine on x along last dim, then x * (1 + scale) + shift — one fused device op.
        TensorShape xShape = new TensorShape(batch, seqLen, _hidden);
        Tensor modulated = new Tensor(xShape, DType.F32);
        backend.LayerNormModulate(modulated, x, scale, shift, _eps);
        shift.Dispose();
        scale.Dispose();

        return (modulated, gate);
    }

    /// <summary>Self-attention with QK-RMSNorm and 3D RoPE on Q,K.</summary>
    // Q/K/V are declared directly as [B, S, H, D] — byte-identical to [B, S, H·D], so the head split costs
    // nothing, RmsNorm normalizes over headDim with no reshape, and the rotary runs on the pre-permute layout
    // (IBackend.ApplyRope indexes cos/sin at row = b·S + s, which for B=1 is exactly AnimaRope's s·headDim
    // offset and the identical rotate-half formula). Permute0213 then produces SDPA's [B, H, S, D].
    private Tensor SelfAttention(IBackend backend, Tensor x, Tensor ropeCos, Tensor ropeSin, AnimaRope rope, int batch, int seqLen)
    {
        TensorShape flatShape = new TensorShape(batch, seqLen, _hidden);
        TensorShape headShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        Tensor q = new Tensor(headShape, DType.F32);
        Tensor k = new Tensor(headShape, DType.F32);
        Tensor v = new Tensor(headShape, DType.F32);
        backend.Linear(q, x, _selfQ!, null);
        backend.Linear(k, x, _selfK!, null);
        backend.Linear(v, x, _selfV!, null);

        Tensor qNorm = new Tensor(headShape, DType.F32);
        Tensor kNorm = new Tensor(headShape, DType.F32);
        backend.RmsNorm(qNorm, q, _selfQNorm.Weight, _qkEps);
        backend.RmsNorm(kNorm, k, _selfKNorm.Weight, _qkEps);
        q.Dispose();
        k.Dispose();

        // The device rotary broadcasts one cos/sin row per (batch, position); AnimaRope's table is position-only,
        // so B > 1 keeps the host applier on the post-permute layout.
        bool gpuRope = batch == 1;
        if (gpuRope)
        {
            backend.ApplyRope(qNorm, kNorm, ropeCos, ropeSin);
        }

        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(qMh, qNorm, seqLen, _numHeads, _headDim);
        backend.Permute0213(kMh, kNorm, seqLen, _numHeads, _headDim);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        qNorm.Dispose();
        kNorm.Dispose();
        v.Dispose();

        if (!gpuRope)
        {
            rope.ApplyRotation(qMh, kMh, ropeCos, ropeSin, batch, _numHeads, seqLen);
        }

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        Tensor flat = new Tensor(flatShape, DType.F32);
        backend.Permute0213(flat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(flatShape, DType.F32);
        backend.Linear(projected, flat, _selfOut!, null);
        flat.Dispose();
        return projected;
    }

    /// <summary>Cross-attention from image tokens (Q) to text tokens (K, V from the LlmAdapter output).
    /// No RoPE. K/V projections are rectangular (<c>[hidden, kvDim]</c>) since the LlmAdapter emits a
    /// 1024-dim stream while the DiT operates at 2048-dim.</summary>
    private Tensor CrossAttention(IBackend backend, Tensor x, Tensor encoderHidden, Tensor? attnMask,
        int batch, int seqLen, int textSeq)
    {
        TensorShape qShape = new TensorShape(batch, seqLen, _hidden);
        TensorShape qHeadShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape kvHeadShape = new TensorShape(batch, textSeq, _numHeads, _headDim);
        TensorShape qMhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(batch, _numHeads, textSeq, _headDim);

        Tensor q = new Tensor(qHeadShape, DType.F32);
        Tensor k = new Tensor(kvHeadShape, DType.F32);
        Tensor v = new Tensor(kvHeadShape, DType.F32);
        backend.Linear(q, x, _crossQ!, null);
        backend.Linear(k, encoderHidden, _crossK!, null);
        backend.Linear(v, encoderHidden, _crossV!, null);

        Tensor qNorm = new Tensor(qHeadShape, DType.F32);
        Tensor kNorm = new Tensor(kvHeadShape, DType.F32);
        backend.RmsNorm(qNorm, q, _crossQNorm.Weight, _qkEps);
        backend.RmsNorm(kNorm, k, _crossKNorm.Weight, _qkEps);
        q.Dispose();
        k.Dispose();

        Tensor qMh = new Tensor(qMhShape, DType.F32);
        Tensor kMh = new Tensor(kvMhShape, DType.F32);
        Tensor vMh = new Tensor(kvMhShape, DType.F32);
        backend.Permute0213(qMh, qNorm, seqLen, _numHeads, _headDim);
        backend.Permute0213(kMh, kNorm, textSeq, _numHeads, _headDim);
        backend.Permute0213(vMh, v, textSeq, _numHeads, _headDim);
        qNorm.Dispose();
        kNorm.Dispose();
        v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attnMask, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        Tensor flat = new Tensor(qShape, DType.F32);
        backend.Permute0213(flat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(qShape, DType.F32);
        backend.Linear(projected, flat, _crossOut!, null);
        flat.Dispose();
        return projected;
    }

    /// <summary>FFN: <c>Linear[ffn, hidden] → GELU → Linear[hidden, ffn]</c>. No biases.</summary>
    private Tensor FeedForward(IBackend backend, Tensor x, int batch, int seqLen)
    {
        TensorShape inShape = new TensorShape(batch, seqLen, _hidden);
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnHidden);

        Tensor proj1 = new Tensor(ffShape, DType.F32);
        backend.Linear(proj1, x, _mlp1!, null);

        Tensor activated = new Tensor(ffShape, DType.F32);
        backend.Gelu(activated, proj1);
        proj1.Dispose();

        Tensor output = new Tensor(inShape, DType.F32);
        backend.Linear(output, activated, _mlp2!, null);
        activated.Dispose();
        return output;
    }

    /// <summary>Device <c>expand + temb[:, :sliceLen]</c>, returning a new <c>[B, sliceLen]</c> tensor. Shared with
    /// <c>AnimaTransformer</c>'s final layer, which slices 2·hidden out of the 3·hidden-wide temb.</summary>
    internal static Tensor AddTembSlice(IBackend backend, Tensor expand, Tensor temb, int batch, int sliceLen)
    {
        int tembStride = (int)temb.Shape[1];
        if (tembStride < sliceLen)
            throw new ArgumentException($"temb width {tembStride} < slice {sliceLen}.", nameof(temb));
        TensorShape shape = new TensorShape(batch, sliceLen);
        Tensor slice = temb;
        if (tembStride > sliceLen)
        {
            slice = new Tensor(shape, DType.F32);
            backend.SliceLastDim(slice, temb, 0);
        }
        Tensor sum = new Tensor(shape, DType.F32);
        backend.Add(sum, expand, slice);
        if (!ReferenceEquals(slice, temb)) slice.Dispose();
        return sum;
    }
}
