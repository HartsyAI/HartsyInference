using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

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

    /// <summary>Forward pass through one DiT block.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="hidden">Image token sequence <c>[B, S, hidden]</c>.</param>
    /// <param name="encoderHidden">Refined text features from <see cref="AnimaLlmAdapter"/>, <c>[B, T, kvDim=1024]</c>.</param>
    /// <param name="embeddedTimestep">RMSNorm'd sin/cos vector <c>[B, 2048]</c> — input to the AdaLN-LoRA bottleneck.</param>
    /// <param name="temb">Processed timestep <c>[B, 6144]</c> — added to the AdaLN-LoRA expand output before chunking.</param>
    /// <param name="ropeCos">RoPE cos table <c>[S, headDim]</c> for image self-attn.</param>
    /// <param name="ropeSin">RoPE sin table <c>[S, headDim]</c> for image self-attn.</param>
    /// <param name="rope">RoPE applier (re-used across blocks).</param>
    /// <param name="crossAttentionMask">Optional cross-attn mask <c>[B, 1, 1, T]</c>.</param>
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
        Tensor afterAttn1 = AddGated(hidden, attn1Out, gate1, batch, seqLen, _hidden);
        attn1Out.Dispose();
        gate1.Dispose();

        // ── 2. AdaLN-LoRA #2 + cross-attention ──
        (Tensor norm2, Tensor gate2) = ApplyAdaLnLora(backend, afterAttn1, embeddedTimestep, temb,
            _adaCrossL1!, _adaCrossL2!, batch, seqLen);
        Tensor attn2Out = CrossAttention(backend, norm2, encoderHidden, crossAttentionMask, batch, seqLen, textSeq);
        norm2.Dispose();
        Tensor afterAttn2 = AddGated(afterAttn1, attn2Out, gate2, batch, seqLen, _hidden);
        attn2Out.Dispose();
        afterAttn1.Dispose();
        gate2.Dispose();

        // ── 3. AdaLN-LoRA #3 + MLP ──
        (Tensor norm3, Tensor gate3) = ApplyAdaLnLora(backend, afterAttn2, embeddedTimestep, temb,
            _adaMlpL1!, _adaMlpL2!, batch, seqLen);
        Tensor ffOut = FeedForward(backend, norm3, batch, seqLen);
        norm3.Dispose();
        Tensor result = AddGated(afterAttn2, ffOut, gate3, batch, seqLen, _hidden);
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
        AddTembSlice(expanded, temb, batch, _adaLnExpandWidth);

        // chunk(3) → shift, scale, gate, each [B, hidden]
        TensorShape chunkShape = new TensorShape(batch, _hidden);
        Tensor shift = new Tensor(chunkShape, DType.F32);
        Tensor scale = new Tensor(chunkShape, DType.F32);
        Tensor gate = new Tensor(chunkShape, DType.F32);
        SplitThree(expanded, shift, scale, gate, batch, _hidden);
        expanded.Dispose();

        // LayerNorm-no-affine on x along last dim, then x * (1 + scale) + shift.
        TensorShape xShape = new TensorShape(batch, seqLen, _hidden);
        Tensor normed = new Tensor(xShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, _hidden, _eps);

        Tensor modulated = AdaLNModulation.ApplyModulation(normed, shift, scale, batch, seqLen, _hidden);
        normed.Dispose();
        shift.Dispose();
        scale.Dispose();

        return (modulated, gate);
    }

    /// <summary>Self-attention with QK-RMSNorm and 3D RoPE on Q,K.</summary>
    private Tensor SelfAttention(IBackend backend, Tensor x, Tensor ropeCos, Tensor ropeSin, AnimaRope rope, int batch, int seqLen)
    {
        TensorShape flatShape = new TensorShape(batch, seqLen, _hidden);

        Tensor q = new Tensor(flatShape, DType.F32);
        Tensor k = new Tensor(flatShape, DType.F32);
        Tensor v = new Tensor(flatShape, DType.F32);
        backend.Linear(q, x, _selfQ!, null);
        backend.Linear(k, x, _selfK!, null);
        backend.Linear(v, x, _selfV!, null);

        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, seqLen, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, seqLen, _numHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, seqLen, _numHeads, _headDim);
        q.Dispose();
        k.Dispose();
        v.Dispose();

        int totalVecs = batch * _numHeads * seqLen;
        Tensor qNorm = new Tensor(qMh.Shape, DType.F32);
        Tensor kNorm = new Tensor(kMh.Shape, DType.F32);
        _selfQNorm.Forward(qNorm, qMh, totalVecs);
        _selfKNorm.Forward(kNorm, kMh, totalVecs);
        qMh.Dispose();
        kMh.Dispose();

        rope.ApplyRotation(qNorm, kNorm, ropeCos, ropeSin, batch, _numHeads, seqLen);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMh.Shape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qNorm, kNorm, vMh, null, scale);
        qNorm.Dispose();
        kNorm.Dispose();
        vMh.Dispose();

        Tensor flat = DiTUtils.ReshapeFromMultiHead(attnOut, batch, seqLen, _numHeads, _headDim);
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
        TensorShape kvShape = new TensorShape(batch, textSeq, _hidden);

        Tensor q = new Tensor(qShape, DType.F32);
        Tensor k = new Tensor(kvShape, DType.F32);
        Tensor v = new Tensor(kvShape, DType.F32);
        backend.Linear(q, x, _crossQ!, null);
        backend.Linear(k, encoderHidden, _crossK!, null);
        backend.Linear(v, encoderHidden, _crossV!, null);

        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, seqLen, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, textSeq, _numHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, textSeq, _numHeads, _headDim);
        q.Dispose();
        k.Dispose();
        v.Dispose();

        int qVecs = batch * _numHeads * seqLen;
        int kVecs = batch * _numHeads * textSeq;
        Tensor qNorm = new Tensor(qMh.Shape, DType.F32);
        Tensor kNorm = new Tensor(kMh.Shape, DType.F32);
        _crossQNorm.Forward(qNorm, qMh, qVecs);
        _crossKNorm.Forward(kNorm, kMh, kVecs);
        qMh.Dispose();
        kMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMh.Shape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qNorm, kNorm, vMh, attnMask, scale);
        qNorm.Dispose();
        kNorm.Dispose();
        vMh.Dispose();

        Tensor flat = DiTUtils.ReshapeFromMultiHead(attnOut, batch, seqLen, _numHeads, _headDim);
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

    /// <summary><c>output = residual + gate[b] * value</c>. Gate broadcast over the sequence dim.</summary>
    private static Tensor AddGated(Tensor residual, Tensor value, Tensor gate, int batch, int seqLen, int hidden)
    {
        TensorShape shape = new TensorShape(batch, seqLen, hidden);
        Tensor output = new Tensor(shape, DType.F32);

        float* resPtr = (float*)residual.DataPointer;
        float* valPtr = (float*)value.DataPointer;
        float* gatePtr = (float*)gate.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int condBase = b * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int rowOff = (b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    outPtr[rowOff + d] = resPtr[rowOff + d] + gatePtr[condBase + d] * valPtr[rowOff + d];
                }
            }
        }
        return output;
    }

    /// <summary>Adds the first <paramref name="sliceLen"/> elements of <paramref name="temb"/> per batch into
    /// <paramref name="expand"/> in-place. Implements the <c>expand + temb[:sliceLen]</c> step of the AdaLN-LoRA.</summary>
    private static void AddTembSlice(Tensor expand, Tensor temb, int batch, int sliceLen)
    {
        int tembStride = (int)temb.Shape[1];
        if (tembStride < sliceLen)
            throw new ArgumentException($"temb width {tembStride} < slice {sliceLen}.", nameof(temb));
        float* outPtr = (float*)expand.DataPointer;
        float* tembPtr = (float*)temb.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int rowOff = b * sliceLen;
            int tRowOff = b * tembStride;
            for (int d = 0; d < sliceLen; d++)
                outPtr[rowOff + d] += tembPtr[tRowOff + d];
        }
    }

    /// <summary>Splits <c>[B, 3*hidden]</c> into three <c>[B, hidden]</c> tensors (shift, scale, gate) in Cosmos order.</summary>
    private static void SplitThree(Tensor src, Tensor a, Tensor b, Tensor c, int batch, int hidden)
    {
        float* sPtr = (float*)src.DataPointer;
        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* cPtr = (float*)c.DataPointer;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++)
        {
            long src0 = (long)batchIdx * 3 * hidden;
            long dst = (long)batchIdx * hidden;
            long bytes = (long)hidden * sizeof(float);
            Buffer.MemoryCopy(sPtr + src0, aPtr + dst, bytes, bytes);
            Buffer.MemoryCopy(sPtr + src0 + hidden, bPtr + dst, bytes, bytes);
            Buffer.MemoryCopy(sPtr + src0 + 2 * hidden, cPtr + dst, bytes, bytes);
        }
    }
}
