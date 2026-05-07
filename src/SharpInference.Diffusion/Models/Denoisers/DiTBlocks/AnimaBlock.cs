using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Anima / Cosmos-Predict2 transformer block. Three sub-blocks (self-attn, cross-attn, FFN) each gated by its own
/// <c>CosmosAdaLayerNormZero</c> module. The AdaLN-Zero produces <c>(shift, scale, gate)</c> from <c>embedded_timestep</c>
/// (with an additive contribution from <c>temb</c>):
/// <code>
///     y = linear_2(silu(linear_1(embedded_timestep)))   # [B, 3*hidden]
///     y = y + temb[..., :3*hidden]                      # broadcast add
///     shift, scale, gate = y.chunk(3, dim=-1)
///     x = norm(x) * (1 + scale) + shift                 # norm = LayerNorm-no-affine
/// </code>
/// The norm is unparameterized LayerNorm (Cosmos uses <c>nn.LayerNorm(elementwise_affine=False, ...)</c>). The block runs:
/// <list type="number">
///   <item>norm1+modulate → SDPA self-attention with QK-RMSNorm + 3D RoPE on Q,K → gated residual.</item>
///   <item>norm2+modulate → SDPA cross-attention (Q from hidden, K/V from <paramref name="encoderHidden"/>) → gated residual.</item>
///   <item>norm3+modulate → Linear(hidden→ffn)+GELU+Linear(ffn→hidden) → gated residual.</item>
/// </list>
/// All attention linears default to <c>bias=True</c> in the upstream Cosmos config; FFN linears do too. We carry biases when
/// present and skip cleanly when absent.</summary>
public sealed unsafe class AnimaBlock
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnHidden;
    private readonly float _eps;
    private readonly float _qkEps;

    private readonly QkNorm _normQ1;
    private readonly QkNorm _normK1;
    private readonly QkNorm _normQ2;
    private readonly QkNorm _normK2;

    // AdaLN_Zero #1 (self-attn): linear_1 [hidden, hidden], linear_2 [hidden, 3*hidden].
    private Tensor? _norm1Linear1Weight, _norm1Linear1Bias;
    private Tensor? _norm1Linear2Weight, _norm1Linear2Bias;

    // AdaLN_Zero #2 (cross-attn): same shape as #1.
    private Tensor? _norm2Linear1Weight, _norm2Linear1Bias;
    private Tensor? _norm2Linear2Weight, _norm2Linear2Bias;

    // AdaLN_Zero #3 (FFN): same shape.
    private Tensor? _norm3Linear1Weight, _norm3Linear1Bias;
    private Tensor? _norm3Linear2Weight, _norm3Linear2Bias;

    // Self-attention (attn1): to_q [hidden, hidden], to_k [hidden, hidden], to_v [hidden, hidden], to_out [hidden, hidden].
    private Tensor? _attn1QWeight, _attn1QBias;
    private Tensor? _attn1KWeight, _attn1KBias;
    private Tensor? _attn1VWeight, _attn1VBias;
    private Tensor? _attn1OutWeight, _attn1OutBias;

    // Cross-attention (attn2): Q from hidden [hidden, hidden], K/V from text [hidden, hidden] (post crossattn_proj),
    // to_out [hidden, hidden].
    private Tensor? _attn2QWeight, _attn2QBias;
    private Tensor? _attn2KWeight, _attn2KBias;
    private Tensor? _attn2VWeight, _attn2VBias;
    private Tensor? _attn2OutWeight, _attn2OutBias;

    // FFN (ff): net.0.proj [ffn, hidden], net.2 [hidden, ffn] in the diffusers FeedForward layout.
    private Tensor? _ffLinear1Weight, _ffLinear1Bias;
    private Tensor? _ffLinear2Weight, _ffLinear2Bias;

    public AnimaBlock(int hidden, int numHeads, int ffnHidden, float eps = 1e-6f, float qkEps = 1e-6f)
    {
        if (hidden % numHeads != 0)
            throw new ArgumentException($"hidden {hidden} must be divisible by numHeads {numHeads}.", nameof(hidden));
        _hidden = hidden;
        _numHeads = numHeads;
        _headDim = hidden / numHeads;
        _ffnHidden = ffnHidden;
        _eps = eps;
        _qkEps = qkEps;
        _normQ1 = new QkNorm(_headDim, qkEps);
        _normK1 = new QkNorm(_headDim, qkEps);
        _normQ2 = new QkNorm(_headDim, qkEps);
        _normK2 = new QkNorm(_headDim, qkEps);
    }

    /// <summary>Loads weights from a diffusers-style key dict. <paramref name="prefix"/> is e.g. <c>transformer_blocks.0</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // ── AdaLN-Zero #1 (self-attn) ──
        _norm1Linear1Weight = weights[$"{prefix}.norm1.linear_1.weight"];
        weights.TryGetValue($"{prefix}.norm1.linear_1.bias", out _norm1Linear1Bias);
        _norm1Linear2Weight = weights[$"{prefix}.norm1.linear_2.weight"];
        weights.TryGetValue($"{prefix}.norm1.linear_2.bias", out _norm1Linear2Bias);

        // ── AdaLN-Zero #2 (cross-attn) ──
        _norm2Linear1Weight = weights[$"{prefix}.norm2.linear_1.weight"];
        weights.TryGetValue($"{prefix}.norm2.linear_1.bias", out _norm2Linear1Bias);
        _norm2Linear2Weight = weights[$"{prefix}.norm2.linear_2.weight"];
        weights.TryGetValue($"{prefix}.norm2.linear_2.bias", out _norm2Linear2Bias);

        // ── AdaLN-Zero #3 (FFN) ──
        _norm3Linear1Weight = weights[$"{prefix}.norm3.linear_1.weight"];
        weights.TryGetValue($"{prefix}.norm3.linear_1.bias", out _norm3Linear1Bias);
        _norm3Linear2Weight = weights[$"{prefix}.norm3.linear_2.weight"];
        weights.TryGetValue($"{prefix}.norm3.linear_2.bias", out _norm3Linear2Bias);

        // ── Self-attn ──
        _attn1QWeight = weights[$"{prefix}.attn1.to_q.weight"];
        weights.TryGetValue($"{prefix}.attn1.to_q.bias", out _attn1QBias);
        _attn1KWeight = weights[$"{prefix}.attn1.to_k.weight"];
        weights.TryGetValue($"{prefix}.attn1.to_k.bias", out _attn1KBias);
        _attn1VWeight = weights[$"{prefix}.attn1.to_v.weight"];
        weights.TryGetValue($"{prefix}.attn1.to_v.bias", out _attn1VBias);
        _attn1OutWeight = weights[$"{prefix}.attn1.to_out.0.weight"];
        weights.TryGetValue($"{prefix}.attn1.to_out.0.bias", out _attn1OutBias);

        _normQ1.LoadWeights(weights[$"{prefix}.attn1.norm_q.weight"]);
        _normK1.LoadWeights(weights[$"{prefix}.attn1.norm_k.weight"]);

        // ── Cross-attn ──
        _attn2QWeight = weights[$"{prefix}.attn2.to_q.weight"];
        weights.TryGetValue($"{prefix}.attn2.to_q.bias", out _attn2QBias);
        _attn2KWeight = weights[$"{prefix}.attn2.to_k.weight"];
        weights.TryGetValue($"{prefix}.attn2.to_k.bias", out _attn2KBias);
        _attn2VWeight = weights[$"{prefix}.attn2.to_v.weight"];
        weights.TryGetValue($"{prefix}.attn2.to_v.bias", out _attn2VBias);
        _attn2OutWeight = weights[$"{prefix}.attn2.to_out.0.weight"];
        weights.TryGetValue($"{prefix}.attn2.to_out.0.bias", out _attn2OutBias);

        _normQ2.LoadWeights(weights[$"{prefix}.attn2.norm_q.weight"]);
        _normK2.LoadWeights(weights[$"{prefix}.attn2.norm_k.weight"]);

        // ── FFN — diffusers FeedForward stores the inner linear at `ff.net.0.proj` and the output at `ff.net.2`. ──
        _ffLinear1Weight = weights[$"{prefix}.ff.net.0.proj.weight"];
        weights.TryGetValue($"{prefix}.ff.net.0.proj.bias", out _ffLinear1Bias);
        _ffLinear2Weight = weights[$"{prefix}.ff.net.2.weight"];
        weights.TryGetValue($"{prefix}.ff.net.2.bias", out _ffLinear2Bias);
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_norm1Linear1Weight is not null) yield return _norm1Linear1Weight;
        if (_norm1Linear1Bias is not null) yield return _norm1Linear1Bias;
        if (_norm1Linear2Weight is not null) yield return _norm1Linear2Weight;
        if (_norm1Linear2Bias is not null) yield return _norm1Linear2Bias;
        if (_norm2Linear1Weight is not null) yield return _norm2Linear1Weight;
        if (_norm2Linear1Bias is not null) yield return _norm2Linear1Bias;
        if (_norm2Linear2Weight is not null) yield return _norm2Linear2Weight;
        if (_norm2Linear2Bias is not null) yield return _norm2Linear2Bias;
        if (_norm3Linear1Weight is not null) yield return _norm3Linear1Weight;
        if (_norm3Linear1Bias is not null) yield return _norm3Linear1Bias;
        if (_norm3Linear2Weight is not null) yield return _norm3Linear2Weight;
        if (_norm3Linear2Bias is not null) yield return _norm3Linear2Bias;
        if (_attn1QWeight is not null) yield return _attn1QWeight;
        if (_attn1QBias is not null) yield return _attn1QBias;
        if (_attn1KWeight is not null) yield return _attn1KWeight;
        if (_attn1KBias is not null) yield return _attn1KBias;
        if (_attn1VWeight is not null) yield return _attn1VWeight;
        if (_attn1VBias is not null) yield return _attn1VBias;
        if (_attn1OutWeight is not null) yield return _attn1OutWeight;
        if (_attn1OutBias is not null) yield return _attn1OutBias;
        foreach (Tensor w in _normQ1.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK1.EnumerateWeights()) yield return w;
        if (_attn2QWeight is not null) yield return _attn2QWeight;
        if (_attn2QBias is not null) yield return _attn2QBias;
        if (_attn2KWeight is not null) yield return _attn2KWeight;
        if (_attn2KBias is not null) yield return _attn2KBias;
        if (_attn2VWeight is not null) yield return _attn2VWeight;
        if (_attn2VBias is not null) yield return _attn2VBias;
        if (_attn2OutWeight is not null) yield return _attn2OutWeight;
        if (_attn2OutBias is not null) yield return _attn2OutBias;
        foreach (Tensor w in _normQ2.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK2.EnumerateWeights()) yield return w;
        if (_ffLinear1Weight is not null) yield return _ffLinear1Weight;
        if (_ffLinear1Bias is not null) yield return _ffLinear1Bias;
        if (_ffLinear2Weight is not null) yield return _ffLinear2Weight;
        if (_ffLinear2Bias is not null) yield return _ffLinear2Bias;
    }

    /// <summary>Forward pass.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="hidden">Image token sequence <c>[B, S, hidden]</c>.</param>
    /// <param name="encoderHidden">Text encoder hiddens projected to model hidden, <c>[B, T, hidden]</c>.</param>
    /// <param name="embeddedTimestep">Per-batch timestep features <c>[B, hidden]</c> (the RMSNorm'd sin/cos vector).</param>
    /// <param name="temb">Per-batch processed timestep <c>[B, condition_dim]</c>; <c>condition_dim ≥ 3*hidden</c>. Added to AdaLN <c>linear_2</c> output.</param>
    /// <param name="ropeCos">RoPE cos table <c>[S, headDim]</c> for image self-attn.</param>
    /// <param name="ropeSin">RoPE sin table <c>[S, headDim]</c> for image self-attn.</param>
    /// <param name="rope">RoPE applier (re-used across blocks).</param>
    /// <param name="crossAttentionMask">Optional cross-attn mask <c>[B, 1, 1, T]</c> with 0 for valid, large-negative for padded text.</param>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor encoderHidden,
        Tensor embeddedTimestep, Tensor temb,
        Tensor ropeCos, Tensor ropeSin, AnimaRope rope,
        Tensor? crossAttentionMask)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        int textSeq = (int)encoderHidden.Shape[1];
        TensorShape hiddenShape = new TensorShape(batch, seqLen, _hidden);
        TensorShape textShape = new TensorShape(batch, textSeq, _hidden);

        // ── 1. AdaLN-Zero #1 + self-attention ──
        (Tensor norm1, Tensor gate1) = ApplyAdaLnZero(backend, hidden, embeddedTimestep, temb,
            _norm1Linear1Weight!, _norm1Linear1Bias, _norm1Linear2Weight!, _norm1Linear2Bias,
            batch, seqLen);
        Tensor attn1Out = SelfAttention(backend, norm1, ropeCos, ropeSin, rope, batch, seqLen);
        norm1.Dispose();
        Tensor afterAttn1 = AddGated(hidden, attn1Out, gate1, batch, seqLen, _hidden);
        attn1Out.Dispose();
        gate1.Dispose();

        // ── 2. AdaLN-Zero #2 + cross-attention ──
        (Tensor norm2, Tensor gate2) = ApplyAdaLnZero(backend, afterAttn1, embeddedTimestep, temb,
            _norm2Linear1Weight!, _norm2Linear1Bias, _norm2Linear2Weight!, _norm2Linear2Bias,
            batch, seqLen);
        Tensor attn2Out = CrossAttention(backend, norm2, encoderHidden, crossAttentionMask, batch, seqLen, textSeq);
        norm2.Dispose();
        Tensor afterAttn2 = AddGated(afterAttn1, attn2Out, gate2, batch, seqLen, _hidden);
        attn2Out.Dispose();
        afterAttn1.Dispose();
        gate2.Dispose();

        // ── 3. AdaLN-Zero #3 + FFN ──
        (Tensor norm3, Tensor gate3) = ApplyAdaLnZero(backend, afterAttn2, embeddedTimestep, temb,
            _norm3Linear1Weight!, _norm3Linear1Bias, _norm3Linear2Weight!, _norm3Linear2Bias,
            batch, seqLen);
        Tensor ffOut = FeedForward(backend, norm3, batch, seqLen);
        norm3.Dispose();
        Tensor result = AddGated(afterAttn2, ffOut, gate3, batch, seqLen, _hidden);
        afterAttn2.Dispose();
        ffOut.Dispose();
        gate3.Dispose();

        return result;
    }

    /// <summary>Computes one AdaLN-Zero modulation pair: returns the modulated tensor and the gate. Allocates
    /// <c>norm</c> output (caller disposes) and <c>gate</c> (caller disposes).</summary>
    private (Tensor Modulated, Tensor Gate) ApplyAdaLnZero(IBackend backend, Tensor x,
        Tensor embeddedTimestep, Tensor temb,
        Tensor lin1Weight, Tensor? lin1Bias, Tensor lin2Weight, Tensor? lin2Bias,
        int batch, int seqLen)
    {
        TensorShape embShape = new TensorShape(batch, _hidden);
        Tensor act = new Tensor(embShape, DType.F32);
        backend.Silu(act, embeddedTimestep);

        Tensor lin1Out = new Tensor(embShape, DType.F32);
        backend.Linear(lin1Out, act, lin1Weight, lin1Bias);
        act.Dispose();

        TensorShape projShape = new TensorShape(batch, 3 * _hidden);
        Tensor lin2Out = new Tensor(projShape, DType.F32);
        backend.Linear(lin2Out, lin1Out, lin2Weight, lin2Bias);
        lin1Out.Dispose();

        // Add temb[..., :3*hidden] to the projected output.
        AddTembSlice(lin2Out, temb, batch, 3 * _hidden);

        // Chunk into shift, scale, gate along last dim.
        TensorShape chunkShape = new TensorShape(batch, _hidden);
        Tensor shift = new Tensor(chunkShape, DType.F32);
        Tensor scale = new Tensor(chunkShape, DType.F32);
        Tensor gate = new Tensor(chunkShape, DType.F32);
        SplitThree(lin2Out, shift, scale, gate, batch, _hidden);
        lin2Out.Dispose();

        // LayerNorm-no-affine on x along last dim, then x*(1+scale)+shift.
        TensorShape xShape = new TensorShape(batch, seqLen, _hidden);
        Tensor normed = new Tensor(xShape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, _hidden, _eps);

        Tensor modulated = AdaLNModulation.ApplyModulation(normed, shift, scale, batch, seqLen, _hidden);
        normed.Dispose();
        shift.Dispose();
        scale.Dispose();

        return (modulated, gate);
    }

    /// <summary>Self-attention with QK-RMSNorm and 3D RoPE on Q,K. Output <c>[B, S, hidden]</c>.</summary>
    private Tensor SelfAttention(IBackend backend, Tensor x, Tensor ropeCos, Tensor ropeSin, AnimaRope rope, int batch, int seqLen)
    {
        TensorShape flatShape = new TensorShape(batch, seqLen, _hidden);
        Tensor q = new Tensor(flatShape, DType.F32);
        Tensor k = new Tensor(flatShape, DType.F32);
        Tensor v = new Tensor(flatShape, DType.F32);
        backend.Linear(q, x, _attn1QWeight!, _attn1QBias);
        backend.Linear(k, x, _attn1KWeight!, _attn1KBias);
        backend.Linear(v, x, _attn1VWeight!, _attn1VBias);

        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, seqLen, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, seqLen, _numHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, seqLen, _numHeads, _headDim);
        q.Dispose();
        k.Dispose();
        v.Dispose();

        int totalVecs = batch * _numHeads * seqLen;
        Tensor qNorm = new Tensor(mhShape, DType.F32);
        Tensor kNorm = new Tensor(mhShape, DType.F32);
        _normQ1.Forward(qNorm, qMh, totalVecs);
        _normK1.Forward(kNorm, kMh, totalVecs);
        qMh.Dispose();
        kMh.Dispose();

        rope.ApplyRotation(qNorm, kNorm, ropeCos, ropeSin, batch, _numHeads, seqLen);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qNorm, kNorm, vMh, null, scale);
        qNorm.Dispose();
        kNorm.Dispose();
        vMh.Dispose();

        Tensor flat = DiTUtils.ReshapeFromMultiHead(attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(flatShape, DType.F32);
        backend.Linear(projected, flat, _attn1OutWeight!, _attn1OutBias);
        flat.Dispose();
        return projected;
    }

    /// <summary>Cross-attention from image tokens (Q) to text tokens (K, V). No RoPE, no QK-norm-on-text — the upstream
    /// only applies <c>norm_q</c>/<c>norm_k</c> to the projected Q and K, which we mirror.</summary>
    private Tensor CrossAttention(IBackend backend, Tensor x, Tensor encoderHidden, Tensor? attnMask, int batch, int seqLen, int textSeq)
    {
        TensorShape qShape = new TensorShape(batch, seqLen, _hidden);
        TensorShape kvShape = new TensorShape(batch, textSeq, _hidden);

        Tensor q = new Tensor(qShape, DType.F32);
        Tensor k = new Tensor(kvShape, DType.F32);
        Tensor v = new Tensor(kvShape, DType.F32);
        backend.Linear(q, x, _attn2QWeight!, _attn2QBias);
        backend.Linear(k, encoderHidden, _attn2KWeight!, _attn2KBias);
        backend.Linear(v, encoderHidden, _attn2VWeight!, _attn2VBias);

        TensorShape qMhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(batch, _numHeads, textSeq, _headDim);
        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, seqLen, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, textSeq, _numHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, textSeq, _numHeads, _headDim);
        q.Dispose();
        k.Dispose();
        v.Dispose();

        int qVecs = batch * _numHeads * seqLen;
        int kVecs = batch * _numHeads * textSeq;
        Tensor qNorm = new Tensor(qMhShape, DType.F32);
        Tensor kNorm = new Tensor(kvMhShape, DType.F32);
        _normQ2.Forward(qNorm, qMh, qVecs);
        _normK2.Forward(kNorm, kMh, kVecs);
        qMh.Dispose();
        kMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qNorm, kNorm, vMh, attnMask, scale);
        qNorm.Dispose();
        kNorm.Dispose();
        vMh.Dispose();

        Tensor flat = DiTUtils.ReshapeFromMultiHead(attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(qShape, DType.F32);
        backend.Linear(projected, flat, _attn2OutWeight!, _attn2OutBias);
        flat.Dispose();
        return projected;
    }

    /// <summary>Standard FFN: Linear(hidden→ffn) → GELU → Linear(ffn→hidden).</summary>
    private Tensor FeedForward(IBackend backend, Tensor x, int batch, int seqLen)
    {
        TensorShape inShape = new TensorShape(batch, seqLen, _hidden);
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnHidden);

        Tensor proj1 = new Tensor(ffShape, DType.F32);
        backend.Linear(proj1, x, _ffLinear1Weight!, _ffLinear1Bias);

        Tensor activated = new Tensor(ffShape, DType.F32);
        backend.Gelu(activated, proj1);
        proj1.Dispose();

        Tensor output = new Tensor(inShape, DType.F32);
        backend.Linear(output, activated, _ffLinear2Weight!, _ffLinear2Bias);
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
    /// <paramref name="lin2Out"/> in-place. Cosmos's AdaLN does <c>embedded_timestep + temb[..., :3*hidden]</c>.</summary>
    private static void AddTembSlice(Tensor lin2Out, Tensor temb, int batch, int sliceLen)
    {
        if (temb.Shape[1] < sliceLen)
            throw new ArgumentException($"temb width {temb.Shape[1]} < slice {sliceLen}.", nameof(temb));
        int tembStride = (int)temb.Shape[1];
        float* outPtr = (float*)lin2Out.DataPointer;
        float* tembPtr = (float*)temb.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int rowOff = b * sliceLen;
            int tRowOff = b * tembStride;
            for (int d = 0; d < sliceLen; d++)
                outPtr[rowOff + d] += tembPtr[tRowOff + d];
        }
    }

    /// <summary>Splits <c>[B, 3*hidden]</c> into three <c>[B, hidden]</c> tensors (shift, scale, gate).</summary>
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
