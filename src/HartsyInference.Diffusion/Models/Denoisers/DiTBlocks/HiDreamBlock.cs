using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>HiDream-I1 transformer block. Two flavors share the same code path:
/// <list type="bullet">
/// <item>Double-stream (joint MM-attention): image and text get separate Q/K/V projections, RMSNorm Q/K, RoPE on the
/// joint sequence (text positions are zero → identity), joint SDPA over the concatenated sequence, separate output
/// projections, separate FFNs (MoE on the image side, vanilla SwiGLU on the text side). 12-way AdaLN modulation
/// (6 per stream).</item>
/// <item>Single-stream: text is already concatenated to the image sequence by the caller; one set of Q/K/V projections,
/// one SDPA, one MoE FFN on the joint sequence. 6-way AdaLN modulation.</item>
/// </list>
/// <para><b>MoE FFN (full top-k routing):</b> the image-side FFN is HiDream's sparse MoE — a softmax gate over
/// <c>num_routed_experts</c> (4) experts, top-k (<c>num_activated_experts</c> = 2) selection per token. Per ComfyUI
/// (<c>MoEGate.norm_topk_prob = False</c>) the selected experts keep their <i>raw</i> softmax weight (no renormalization
/// to sum 1), plus an always-on shared expert added on top. Because the <see cref="IBackend"/> exposes no
/// grouped-GEMM / sort-scatter expert-dispatch primitive, this block runs the routed forward densely (every expert
/// evaluated for every token) and zeroes non-selected experts via the per-token gate weight — numerically identical to
/// a true sparse dispatch, just with extra compute. See <see cref="MoeForward"/>.</para>
/// <para><b>GPU-residency:</b> every glue op — non-affine LayerNorm, AdaLN affine (scale/shift), gated residual, Q/K
/// RMSNorm, reshape-to-heads, joint concat/split, and the MoE gate scatter — runs as an <see cref="IBackend"/> GPU op so
/// the activation stays device-resident across the whole block (no per-op <c>DataPointer</c> reads / D2H sync barriers).
/// The only CPU step is the tiny MoE gate softmax/top-k over the <c>[B,S,num_experts]</c> logits, and RoPE
/// (<see cref="HiDreamRope"/>, whose D2H/H2D for Q/K is coherent with the activation cache). Numerics match the old
/// CPU path bit-for-bit.</para></summary>
public sealed unsafe class HiDreamBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffDim;
    private readonly bool _isSingle;
    private readonly float _qkNormEps;
    private readonly int _numActivatedExperts;

    // AdaLN: 12 params (double) or 6 params (single). All produced from a single SiLU+Linear MLP.
    private Tensor? _adaLnLinearWeight, _adaLnLinearBias;

    // ── Image attention (always present) ──
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;
    private Tensor? _qRmsNormWeight, _kRmsNormWeight;

    // ── Text attention (double-stream only) ──
    private Tensor? _toQTWeight, _toQTBias;
    private Tensor? _toKTWeight, _toKTBias;
    private Tensor? _toVTWeight, _toVTBias;
    private Tensor? _toOutTWeight, _toOutTBias;
    private Tensor? _qRmsNormTWeight, _kRmsNormTWeight;

    // ── Image FFN (MoE: shared + experts[0..N-1] + gate) ──
    // SwiGLU: w1 = gate-up, w3 = up, w2 = down. shared has hidden_dim/2; routed experts have hidden_dim.
    private Tensor? _sharedW1, _sharedW2, _sharedW3;

    // Routed experts. Stored as flat arrays; all participate in the top-k routed forward (and all are
    // reported by EnumerateWeights for the full checkpoint footprint / GPU preload).
    private Tensor[]? _expertW1;
    private Tensor[]? _expertW2;
    private Tensor[]? _expertW3;
    private Tensor? _moeGateWeight;
    private readonly int _numRoutedExperts;

    // ── Text FFN (vanilla SwiGLU, double-stream only) ──
    private Tensor? _ffTW1, _ffTW2, _ffTW3;

    /// <summary>Creates a HiDream block.</summary>
    /// <param name="hiddenSize">Inner model dim (= numHeads * headDim).</param>
    /// <param name="numHeads">Attention head count.</param>
    /// <param name="headDim">Per-head dim.</param>
    /// <param name="ffDim">SwiGLU inner dim. Diffusers uses 4 * hiddenSize for HiDream.</param>
    /// <param name="isSingle">True for single-stream blocks (only image-side modules).</param>
    /// <param name="numRoutedExperts">Number of routed experts in the MoE FFN (HiDream: 4). All are loaded and participate in routing.</param>
    /// <param name="numActivatedExperts">Top-k experts selected per token during the routed forward (HiDream: 2). If &lt;= 1, routing collapses to top-1 (argmax expert).</param>
    /// <param name="qkNormEps">RMSNorm epsilon for the per-stream Q/K norms.</param>
    public HiDreamBlock(int hiddenSize, int numHeads, int headDim, int ffDim,
        bool isSingle, int numRoutedExperts, int numActivatedExperts, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _ffDim = ffDim;
        _isSingle = isSingle;
        _qkNormEps = qkNormEps;
        _numRoutedExperts = numRoutedExperts;
        _numActivatedExperts = numActivatedExperts;
    }

    /// <summary>Loads all weights for this block from the converted (diffusers-style) state dict.</summary>
    /// <param name="weights">Flat weight dict.</param>
    /// <param name="prefix">Either <c>double_stream_blocks.{i}.block</c> or <c>single_stream_blocks.{i}.block</c>.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // AdaLN modulation MLP: SiLU + Linear(dim, 12*dim) (or 6*dim for single).
        // The diffusers Sequential places the Linear at index 1, so the state-dict key is
        // "{prefix}.adaLN_modulation.1.weight/bias".
        _adaLnLinearWeight = weights[$"{prefix}.adaLN_modulation.1.weight"];
        _adaLnLinearBias = weights[$"{prefix}.adaLN_modulation.1.bias"];

        // Image attention projections
        _toQWeight = weights[$"{prefix}.attn1.to_q.weight"];
        _toQBias = weights.TryGetValue($"{prefix}.attn1.to_q.bias", out Tensor? toQBias) ? toQBias : null;
        _toKWeight = weights[$"{prefix}.attn1.to_k.weight"];
        _toKBias = weights.TryGetValue($"{prefix}.attn1.to_k.bias", out Tensor? toKBias) ? toKBias : null;
        _toVWeight = weights[$"{prefix}.attn1.to_v.weight"];
        _toVBias = weights.TryGetValue($"{prefix}.attn1.to_v.bias", out Tensor? toVBias) ? toVBias : null;
        _toOutWeight = weights[$"{prefix}.attn1.to_out.weight"];
        _toOutBias = weights.TryGetValue($"{prefix}.attn1.to_out.bias", out Tensor? toOutBias) ? toOutBias : null;
        _qRmsNormWeight = weights[$"{prefix}.attn1.q_rms_norm.weight"];
        _kRmsNormWeight = weights[$"{prefix}.attn1.k_rms_norm.weight"];

        if (!_isSingle)
        {
            _toQTWeight = weights[$"{prefix}.attn1.to_q_t.weight"];
            _toQTBias = weights.TryGetValue($"{prefix}.attn1.to_q_t.bias", out Tensor? toQTB) ? toQTB : null;
            _toKTWeight = weights[$"{prefix}.attn1.to_k_t.weight"];
            _toKTBias = weights.TryGetValue($"{prefix}.attn1.to_k_t.bias", out Tensor? toKTB) ? toKTB : null;
            _toVTWeight = weights[$"{prefix}.attn1.to_v_t.weight"];
            _toVTBias = weights.TryGetValue($"{prefix}.attn1.to_v_t.bias", out Tensor? toVTB) ? toVTB : null;
            _toOutTWeight = weights[$"{prefix}.attn1.to_out_t.weight"];
            _toOutTBias = weights.TryGetValue($"{prefix}.attn1.to_out_t.bias", out Tensor? toOutTB) ? toOutTB : null;
            _qRmsNormTWeight = weights[$"{prefix}.attn1.q_rms_norm_t.weight"];
            _kRmsNormTWeight = weights[$"{prefix}.attn1.k_rms_norm_t.weight"];
        }

        // Image FFN (MoE): shared experts + N routed experts + gate
        _sharedW1 = weights[$"{prefix}.ff_i.shared_experts.w1.weight"];
        _sharedW2 = weights[$"{prefix}.ff_i.shared_experts.w2.weight"];
        _sharedW3 = weights[$"{prefix}.ff_i.shared_experts.w3.weight"];

        _expertW1 = new Tensor[_numRoutedExperts];
        _expertW2 = new Tensor[_numRoutedExperts];
        _expertW3 = new Tensor[_numRoutedExperts];
        for (int e = 0; e < _numRoutedExperts; e++)
        {
            _expertW1[e] = weights[$"{prefix}.ff_i.experts.{e}.w1.weight"];
            _expertW2[e] = weights[$"{prefix}.ff_i.experts.{e}.w2.weight"];
            _expertW3[e] = weights[$"{prefix}.ff_i.experts.{e}.w3.weight"];
        }
        _moeGateWeight = weights[$"{prefix}.ff_i.gate.weight"];

        // Text FFN (vanilla SwiGLU, double-stream only)
        if (!_isSingle)
        {
            _ffTW1 = weights[$"{prefix}.ff_t.w1.weight"];
            _ffTW2 = weights[$"{prefix}.ff_t.w2.weight"];
            _ffTW3 = weights[$"{prefix}.ff_t.w3.weight"];
        }
    }

    /// <summary>Yields every weight tensor in this block, including all routed experts (all of which
    /// participate in the top-k routed forward).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_adaLnLinearWeight is not null) yield return _adaLnLinearWeight;
        if (_adaLnLinearBias is not null) yield return _adaLnLinearBias;

        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_toOutWeight is not null) yield return _toOutWeight;
        if (_toOutBias is not null) yield return _toOutBias;
        if (_qRmsNormWeight is not null) yield return _qRmsNormWeight;
        if (_kRmsNormWeight is not null) yield return _kRmsNormWeight;

        if (_toQTWeight is not null) yield return _toQTWeight;
        if (_toQTBias is not null) yield return _toQTBias;
        if (_toKTWeight is not null) yield return _toKTWeight;
        if (_toKTBias is not null) yield return _toKTBias;
        if (_toVTWeight is not null) yield return _toVTWeight;
        if (_toVTBias is not null) yield return _toVTBias;
        if (_toOutTWeight is not null) yield return _toOutTWeight;
        if (_toOutTBias is not null) yield return _toOutTBias;
        if (_qRmsNormTWeight is not null) yield return _qRmsNormTWeight;
        if (_kRmsNormTWeight is not null) yield return _kRmsNormTWeight;

        if (_sharedW1 is not null) yield return _sharedW1;
        if (_sharedW2 is not null) yield return _sharedW2;
        if (_sharedW3 is not null) yield return _sharedW3;

        if (_expertW1 is not null)
        {
            for (int e = 0; e < _expertW1.Length; e++)
            {
                yield return _expertW1[e];
                yield return _expertW2![e];
                yield return _expertW3![e];
            }
        }
        if (_moeGateWeight is not null) yield return _moeGateWeight;

        if (_ffTW1 is not null) yield return _ffTW1;
        if (_ffTW2 is not null) yield return _ffTW2;
        if (_ffTW3 is not null) yield return _ffTW3;
    }

    /// <summary>Forward pass for double-stream blocks. Returns updated (image, text) pair. The text input here is the per-block "encoder hidden states" — the caller is responsible for splicing the active Llama hidden state back to the initial-encoder slice between blocks.</summary>
    public (Tensor image, Tensor text) ForwardDouble(IBackend backend, Tensor image, Tensor text,
        Tensor temb, HiDreamRope rope, int totalRopeSeqLen)
    {
        if (_isSingle)
            throw new InvalidOperationException("ForwardDouble called on a single-stream block.");

        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];

        // ── 1. AdaLN modulation: 12 params (image: 6, text: 6) ──
        // Order matches diffusers/ComfyUI: shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i,
        // gate_mlp_i, shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t.
        Tensor[] mods = ComputeAdaLnParams(backend, temb, batch, 12);

        // ── 2. Pre-attention norms + AdaLN modulation ──
        Tensor imgNormed = LayerNorm(backend, image, batch, imgSeqLen);
        Tensor imgMod = ApplyMod(backend, imgNormed, mods[0], mods[1], batch, imgSeqLen);
        imgNormed.Dispose();

        Tensor txtNormed = LayerNorm(backend, text, batch, txtSeqLen);
        Tensor txtMod = ApplyMod(backend, txtNormed, mods[6], mods[7], batch, txtSeqLen);
        txtNormed.Dispose();

        // ── 3. Joint MM-attention ──
        (Tensor imgAttnOut, Tensor txtAttnOut) = JointAttention(backend, imgMod, txtMod, rope, batch, imgSeqLen, txtSeqLen);
        imgMod.Dispose();
        txtMod.Dispose();

        // ── 4. Gated residuals: hidden = hidden + gate * attn_out ──
        Tensor imgAfterAttn = GatedRes(backend, image, imgAttnOut, mods[2], batch, imgSeqLen);
        imgAttnOut.Dispose();
        Tensor txtAfterAttn = GatedRes(backend, text, txtAttnOut, mods[8], batch, txtSeqLen);
        txtAttnOut.Dispose();

        // ── 5. Image FFN (MoE) ──
        Tensor imgPreFfn = LayerNorm(backend, imgAfterAttn, batch, imgSeqLen);
        Tensor imgFfnIn = ApplyMod(backend, imgPreFfn, mods[3], mods[4], batch, imgSeqLen);
        imgPreFfn.Dispose();

        Tensor imgFfnOut = MoeForward(backend, imgFfnIn, batch, imgSeqLen);
        imgFfnIn.Dispose();

        Tensor imgFinal = GatedRes(backend, imgAfterAttn, imgFfnOut, mods[5], batch, imgSeqLen);
        imgFfnOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 6. Text FFN (vanilla SwiGLU) ──
        Tensor txtPreFfn = LayerNorm(backend, txtAfterAttn, batch, txtSeqLen);
        Tensor txtFfnIn = ApplyMod(backend, txtPreFfn, mods[9], mods[10], batch, txtSeqLen);
        txtPreFfn.Dispose();

        Tensor txtFfnOut = SwiGluForward(backend, txtFfnIn, _ffTW1!, _ffTW3!, _ffTW2!, batch, txtSeqLen, _ffDim);
        txtFfnIn.Dispose();

        Tensor txtFinal = GatedRes(backend, txtAfterAttn, txtFfnOut, mods[11], batch, txtSeqLen);
        txtFfnOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < mods.Length; i++) mods[i].Dispose();
        return (imgFinal, txtFinal);
    }

    /// <summary>Forward pass for single-stream blocks. The hidden states already contain
    /// image tokens followed by all relevant text tokens; one set of Q/K/V projections is run
    /// over the joint sequence.</summary>
    public Tensor ForwardSingle(IBackend backend, Tensor hidden, Tensor temb, HiDreamRope rope, int imgSeqLen, int totalRopeSeqLen)
    {
        if (!_isSingle)
            throw new InvalidOperationException("ForwardSingle called on a double-stream block.");

        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];

        // ── 1. AdaLN: 6 params (shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i) ──
        Tensor[] mods = ComputeAdaLnParams(backend, temb, batch, 6);

        // ── 2. Pre-attention norm + modulate ──
        Tensor preAttn = LayerNorm(backend, hidden, batch, seqLen);
        Tensor preAttnMod = ApplyMod(backend, preAttn, mods[0], mods[1], batch, seqLen);
        preAttn.Dispose();

        // ── 3. Self-attention over the joint sequence (text positions rotate by 0 = identity) ──
        Tensor attnOut = SingleStreamSelfAttention(backend, preAttnMod, rope, batch, seqLen);
        preAttnMod.Dispose();

        Tensor afterAttn = GatedRes(backend, hidden, attnOut, mods[2], batch, seqLen);
        attnOut.Dispose();

        // ── 4. MoE FFN over the joint sequence ──
        Tensor preFfn = LayerNorm(backend, afterAttn, batch, seqLen);
        Tensor ffnIn = ApplyMod(backend, preFfn, mods[3], mods[4], batch, seqLen);
        preFfn.Dispose();

        Tensor ffnOut = MoeForward(backend, ffnIn, batch, seqLen);
        ffnIn.Dispose();

        Tensor result = GatedRes(backend, afterAttn, ffnOut, mods[5], batch, seqLen);
        ffnOut.Dispose();
        afterAttn.Dispose();

        for (int i = 0; i < mods.Length; i++) mods[i].Dispose();
        return result;
    }

    /// <summary>Computes <paramref name="numParams"/> AdaLN modulation tensors (each [B, hiddenSize]) from a
    /// [B, hiddenSize] timestep embedding using SiLU + Linear, then splits the [B, numParams*hiddenSize]
    /// projection into per-param [B, hiddenSize] GPU slices (<see cref="IBackend.SliceLastDim"/>).</summary>
    private Tensor[] ComputeAdaLnParams(IBackend backend, Tensor temb, int batch, int numParams)
    {
        int outDim = numParams * _hiddenSize;
        Tensor activated = new Tensor(new TensorShape(batch, _hiddenSize), DType.F32);
        backend.Silu(activated, temb);

        Tensor projected = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.Linear(projected, activated, _adaLnLinearWeight!, _adaLnLinearBias);
        activated.Dispose();

        Tensor[] results = new Tensor[numParams];
        for (int p = 0; p < numParams; p++)
        {
            Tensor param = new Tensor(new TensorShape(batch, _hiddenSize), DType.F32);
            backend.SliceLastDim(param, projected, p * _hiddenSize);
            results[p] = param;
        }
        projected.Dispose();
        return results;
    }

    /// <summary>Non-affine LayerNorm over the hidden dim (ComfyUI <c>norm1_i/norm3_i</c>, eps 1e-6).</summary>
    private Tensor LayerNorm(IBackend backend, Tensor input, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hiddenSize), DType.F32);
        backend.LayerNormNoAffine(output, input, 1e-6f);
        return output;
    }

    /// <summary>AdaLN affine modulation <c>out = input * (1 + scale) + shift</c>, GPU-resident
    /// (<c>AddScalar</c> + <c>AffineBroadcastLastDim</c>). <paramref name="shift"/>/<paramref name="scale"/> are
    /// [B, hidden] broadcast over the sequence axis.</summary>
    private Tensor ApplyMod(IBackend backend, Tensor input, Tensor shift, Tensor scale, int batch, int seqLen)
    {
        Tensor scalePlus1 = new Tensor(new TensorShape(batch, _hiddenSize), DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hiddenSize), DType.F32);
        backend.AffineBroadcastLastDim(output, input, scalePlus1, shift);
        scalePlus1.Dispose();
        return output;
    }

    /// <summary>Gated residual <c>out = residual + gate * value</c>, GPU-resident
    /// (<see cref="IBackend.GatedResidualLastDim"/>). <paramref name="gate"/> is [B, hidden] broadcast over seq.</summary>
    private Tensor GatedRes(IBackend backend, Tensor residual, Tensor value, Tensor gate, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hiddenSize), DType.F32);
        backend.GatedResidualLastDim(output, residual, value, gate);
        return output;
    }

    /// <summary>Joint MM-attention shared between image and text in double-stream blocks. Image and text each get
    /// their own Q/K/V; both are RMS-normed over the full inner dim (ComfyUI <c>q_rms_norm</c> = RMSNorm(inner_dim)),
    /// concatenated [image, text] on the sequence axis, reshaped to heads, RoPE applied over the joint sequence (text
    /// positions are zero → identity), SDPA, then split back and projected per stream.</summary>
    private (Tensor img, Tensor txt) JointAttention(IBackend backend, Tensor img, Tensor txt,
        HiDreamRope rope, int batch, int imgSeqLen, int txtSeqLen)
    {
        int total = imgSeqLen + txtSeqLen;
        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        TensorShape jointShape = new TensorShape(batch, total, _hiddenSize);

        // Q/K/V projections + RMSNorm Q/K over the full inner dim (image side).
        Tensor imgQ = LinearAlloc(backend, img, _toQWeight!, _toQBias, imgShape);
        Tensor imgQn = RmsNormAlloc(backend, imgQ, _qRmsNormWeight!, imgShape);
        imgQ.Dispose();
        Tensor imgK = LinearAlloc(backend, img, _toKWeight!, _toKBias, imgShape);
        Tensor imgKn = RmsNormAlloc(backend, imgK, _kRmsNormWeight!, imgShape);
        imgK.Dispose();
        Tensor imgV = LinearAlloc(backend, img, _toVWeight!, _toVBias, imgShape);

        // Q/K/V projections + RMSNorm Q/K (text side).
        Tensor txtQ = LinearAlloc(backend, txt, _toQTWeight!, _toQTBias, txtShape);
        Tensor txtQn = RmsNormAlloc(backend, txtQ, _qRmsNormTWeight!, txtShape);
        txtQ.Dispose();
        Tensor txtK = LinearAlloc(backend, txt, _toKTWeight!, _toKTBias, txtShape);
        Tensor txtKn = RmsNormAlloc(backend, txtK, _kRmsNormTWeight!, txtShape);
        txtK.Dispose();
        Tensor txtV = LinearAlloc(backend, txt, _toVTWeight!, _toVTBias, txtShape);

        // Concat [image, text] along the sequence dim (matches ComfyUI cat([q_i, q_t], dim=1)).
        Tensor jointQ = new Tensor(jointShape, DType.F32);
        backend.Concat(jointQ, new Tensor[] { imgQn, txtQn }, 1);
        Tensor jointK = new Tensor(jointShape, DType.F32);
        backend.Concat(jointK, new Tensor[] { imgKn, txtKn }, 1);
        Tensor jointV = new Tensor(jointShape, DType.F32);
        backend.Concat(jointV, new Tensor[] { imgV, txtV }, 1);
        imgQn.Dispose(); txtQn.Dispose(); imgKn.Dispose(); txtKn.Dispose(); imgV.Dispose(); txtV.Dispose();

        Tensor qMh = ToHeads(backend, jointQ, batch, total);
        jointQ.Dispose();
        Tensor kMh = ToHeads(backend, jointK, batch, total);
        jointK.Dispose();
        Tensor vMh = ToHeads(backend, jointV, batch, total);
        jointV.Dispose();

        // RoPE over the joint sequence; text positions are zero in the rope table → identity.
        rope.Forward(qMh, kMh, batch, _numHeads, total);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(new TensorShape(batch, _numHeads, total, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = FromHeads(backend, attnMh, batch, total);
        attnMh.Dispose();

        // Split [image, text] and project per stream.
        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        backend.SliceRows(imgAttn, attnFlat, 0);
        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        backend.SliceRows(txtAttn, attnFlat, imgSeqLen);
        attnFlat.Dispose();

        Tensor imgOut = LinearAlloc(backend, imgAttn, _toOutWeight!, _toOutBias, imgShape);
        Tensor txtOut = LinearAlloc(backend, txtAttn, _toOutTWeight!, _toOutTBias, txtShape);
        imgAttn.Dispose();
        txtAttn.Dispose();

        return (imgOut, txtOut);
    }

    /// <summary>Self-attention over the concatenated single-stream sequence. The image-side weights are reused for
    /// the entire sequence; RoPE rotates every position (text positions are zero → identity).</summary>
    private Tensor SingleStreamSelfAttention(IBackend backend, Tensor hidden, HiDreamRope rope, int batch, int seqLen)
    {
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);

        Tensor q = LinearAlloc(backend, hidden, _toQWeight!, _toQBias, shape);
        Tensor qn = RmsNormAlloc(backend, q, _qRmsNormWeight!, shape);
        q.Dispose();
        Tensor k = LinearAlloc(backend, hidden, _toKWeight!, _toKBias, shape);
        Tensor kn = RmsNormAlloc(backend, k, _kRmsNormWeight!, shape);
        k.Dispose();
        Tensor v = LinearAlloc(backend, hidden, _toVWeight!, _toVBias, shape);

        Tensor qMh = ToHeads(backend, qn, batch, seqLen);
        qn.Dispose();
        Tensor kMh = ToHeads(backend, kn, batch, seqLen);
        kn.Dispose();
        Tensor vMh = ToHeads(backend, v, batch, seqLen);
        v.Dispose();

        rope.Forward(qMh, kMh, batch, _numHeads, seqLen);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(new TensorShape(batch, _numHeads, seqLen, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = FromHeads(backend, attnMh, batch, seqLen);
        attnMh.Dispose();

        Tensor outProj = LinearAlloc(backend, attnFlat, _toOutWeight!, _toOutBias, shape);
        attnFlat.Dispose();
        return outProj;
    }

    /// <summary>Reshapes a flat [B, S, hidden] tensor to head-major [B, numHeads, S, headDim]
    /// (<see cref="IBackend.Permute0213"/> on the byte-identical [B, S, numHeads, headDim] view).</summary>
    private Tensor ToHeads(IBackend backend, Tensor flat, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, _numHeads, seqLen, _headDim), DType.F32);
        backend.Permute0213(output, flat, seqLen, _numHeads, _headDim);
        return output;
    }

    /// <summary>Inverse of <see cref="ToHeads"/>: [B, numHeads, S, headDim] → [B, S, hidden].</summary>
    private Tensor FromHeads(IBackend backend, Tensor mh, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hiddenSize), DType.F32);
        backend.Permute0213(output, mh, _numHeads, seqLen, _headDim);
        return output;
    }

    /// <summary>Allocates an output of the given shape and runs <c>backend.RmsNorm</c> over the last dim.</summary>
    private Tensor RmsNormAlloc(IBackend backend, Tensor input, Tensor weight, TensorShape shape)
    {
        Tensor output = new Tensor(shape, DType.F32);
        backend.RmsNorm(output, input, weight, _qkNormEps);
        return output;
    }

    /// <summary>MoE SwiGLU FFN: <c>shared_experts(x) + sum_k(topk_weight_k * experts[idx_k](x))</c>.
    /// <para>Mirrors ComfyUI HiDream <c>MoEGate</c> + <c>MOEFeedForwardSwiGLU</c>: the gate Linear produces
    /// <c>[B, S, num_routed_experts]</c> logits; softmax over the expert axis; the top-k (= num_activated_experts)
    /// experts per token are selected and kept at their <b>raw</b> softmax weight (ComfyUI sets
    /// <c>norm_topk_prob = False</c>, so there is no renormalization to sum 1). Each routed expert is evaluated
    /// densely and scaled by its (possibly zero) per-token gate weight via <see cref="IBackend.MaskRows"/>, then
    /// accumulated; the always-on shared expert is added on top. GPU-resident except the tiny gate softmax/top-k.</para></summary>
    private Tensor MoeForward(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape inShape = new TensorShape(batch, seqLen, _hiddenSize);

        // Shared (always-on) expert: SwiGLU at hidden_dim/2. Seeds the accumulator.
        int sharedDim = (_ffDim + 1) / 2; // ff_dim // 2 (Python integer division of 4*hidden / 2)
        Tensor combined = SwiGluForward(backend, input, _sharedW1!, _sharedW3!, _sharedW2!, batch, seqLen, sharedDim);

        // Gate logits → per-token, per-expert renorm-free top-k weights (0 for non-selected). [B, S, E].
        Tensor gateLogits = new Tensor(new TensorShape(batch, seqLen, _numRoutedExperts), DType.F32);
        backend.Linear(gateLogits, input, _moeGateWeight!, null);
        Tensor gateWeights = new Tensor(new TensorShape(batch, seqLen, _numRoutedExperts), DType.F32);
        int activeK = _numActivatedExperts <= 1 ? 1 : _numActivatedExperts;
        ComputeTopKGateWeights(gateLogits, gateWeights, batch, seqLen, _numRoutedExperts, activeK);
        gateLogits.Dispose();

        for (int e = 0; e < _numRoutedExperts; e++)
        {
            Tensor expertOut = SwiGluForward(backend, input, _expertW1![e], _expertW3![e], _expertW2![e], batch, seqLen, _ffDim);

            // Per-token gate column [B, S, 1] → per-row scale of the expert output, then accumulate.
            Tensor gateColE = new Tensor(new TensorShape(batch, seqLen, 1), DType.F32);
            backend.SliceLastDim(gateColE, gateWeights, e);
            Tensor scaled = new Tensor(inShape, DType.F32);
            backend.MaskRows(scaled, expertOut, gateColE);
            expertOut.Dispose();
            gateColE.Dispose();

            Tensor newCombined = new Tensor(inShape, DType.F32);
            backend.Add(newCombined, combined, scaled);
            combined.Dispose();
            scaled.Dispose();
            combined = newCombined;
        }
        gateWeights.Dispose();
        return combined;
    }

    /// <summary>SwiGLU forward: <c>w2(silu(w1(x)) * w3(x))</c>. Used by the text FFN, the shared MoE expert, and
    /// each routed expert.</summary>
    private Tensor SwiGluForward(IBackend backend, Tensor input, Tensor w1, Tensor w3, Tensor w2,
        int batch, int seqLen, int hiddenInner)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, hiddenInner);
        Tensor gate = new Tensor(ffShape, DType.F32);
        backend.Linear(gate, input, w1, null);

        Tensor gateAct = new Tensor(ffShape, DType.F32);
        backend.Silu(gateAct, gate);
        gate.Dispose();

        Tensor up = new Tensor(ffShape, DType.F32);
        backend.Linear(up, input, w3, null);

        Tensor gated = new Tensor(ffShape, DType.F32);
        backend.Mul(gated, gateAct, up);
        gateAct.Dispose();
        up.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, gated, w2, null);
        gated.Dispose();
        return output;
    }

    /// <summary>Allocates an output tensor of shape <paramref name="outShape"/> and runs <c>backend.Linear</c>
    /// from <paramref name="input"/> through <paramref name="weight"/> + optional <paramref name="bias"/>.</summary>
    private static Tensor LinearAlloc(IBackend backend, Tensor input, Tensor weight, Tensor? bias, TensorShape outShape)
    {
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    /// <summary>Computes the per-token, per-expert top-k gate weights from routing logits.
    /// <para>Matches ComfyUI HiDream <c>MoEGate</c>: <c>scores = softmax(logits, dim=-1)</c>; pick the top-k experts
    /// by score and keep their raw softmax weight (<c>norm_topk_prob = False</c> → no renormalization). The output
    /// <paramref name="weights"/> is dense <c>[B, S, E]</c> with the selected weight at each chosen expert slot and 0
    /// elsewhere, so a dense per-expert masked accumulation reproduces the sparse top-k dispatch exactly.</para>
    /// This is the only CPU step in the block — a tiny [B, S, E] tensor (E = 4).</summary>
    private static void ComputeTopKGateWeights(Tensor logits, Tensor weights, int batch, int seqLen, int numExperts, int topK)
    {
        float* lp = (float*)logits.DataPointer;
        float* wp = (float*)weights.DataPointer;
        Span<float> probs = stackalloc float[numExperts];
        int k = Math.Min(topK, numExperts);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int off = (b * seqLen + s) * numExperts;

                // softmax over the expert axis.
                float maxLogit = lp[off];
                for (int e = 1; e < numExperts; e++)
                    if (lp[off + e] > maxLogit) maxLogit = lp[off + e];
                float sum = 0f;
                for (int e = 0; e < numExperts; e++)
                {
                    float ex = MathF.Exp(lp[off + e] - maxLogit);
                    probs[e] = ex;
                    sum += ex;
                }
                for (int e = 0; e < numExperts; e++)
                {
                    probs[e] /= sum;
                    wp[off + e] = 0f;
                }

                // top-k selection (k small — simple repeated argmax). Selected experts keep their raw softmax
                // weight; ComfyUI's norm_topk_prob is False so there is no renormalization.
                for (int kk = 0; kk < k; kk++)
                {
                    int best = -1;
                    float bestVal = float.NegativeInfinity;
                    for (int e = 0; e < numExperts; e++)
                    {
                        if (wp[off + e] != 0f) continue; // already selected
                        if (probs[e] > bestVal)
                        {
                            bestVal = probs[e];
                            best = e;
                        }
                    }
                    wp[off + best] = bestVal;
                }
            }
        }
    }
}

/// <summary>Tiny helper to read a tensor shape dim by index without a managed cast for every use site.</summary>
internal static class HiDreamTensorExtensions
{
    public static int ShapeValue(this Tensor t, int axis) => (int)t.Shape[axis];
}
