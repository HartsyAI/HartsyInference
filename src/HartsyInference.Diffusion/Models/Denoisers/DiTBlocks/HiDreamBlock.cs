using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>HiDream-I1 transformer block. Two flavors share the same code path:
/// <list type="bullet">
/// <item>Double-stream (joint MM-attention): image and text get separate Q/K/V projections, RMSNorm Q/K, RoPE on the image side, joint SDPA over the concatenated sequence, separate output projections, separate FFNs (MoE on the image side, vanilla SwiGLU on the text side). 12-way AdaLN modulation (6 per stream).</item>
/// <item>Single-stream: text is already concatenated to the image sequence by the caller; one set of Q/K/V projections, one SDPA, one MoE FFN on the joint sequence. 6-way AdaLN modulation.</item>
/// </list>
/// <para><b>MoE design decision (single-expert fallback):</b> the MoE FFN in HiDream uses softmax gating, top-k=2 selection out of 4 routed experts plus a shared expert. HartsyInference's <see cref="IBackend"/> doesn't yet expose grouped-GEMM / expert-dispatch / sort-scatter primitives needed for an efficient routed MoE forward, and per-token expert routing on the CPU would require allocations on the inference hot path. To keep the model loadable and forward-shape-correct without expanding the backend surface, this block uses a <i>single-expert routed path</i>: it loads <c>experts[0]</c>'s SwiGLU weights as the only routed expert and adds the shared expert. All four expert tensors are still loaded into <see cref="EnumerateWeights"/> so the GPU preload byte count and the checkpoint-conversion path stay honest, but only <c>experts[0]</c> participates in the forward pass. The shape of the output is correct; the values diverge from a true MoE forward in proportion to how often the trained gate routed to experts[1..3] vs experts[0]. See the <c>// TODO: full MoE routing</c> markers below for the call sites a future routed implementation must replace.</para></summary>
public sealed unsafe class HiDreamBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffDim;
    private readonly bool _isSingle;
    private readonly float _qkNormEps;

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

    // Routed experts. Stored as flat arrays so EnumerateWeights() reports every expert tensor (full
    // checkpoint footprint), even though the forward pass only uses _experts[0] under the single-expert
    // fallback. Slot 0 is the active expert; the rest are "weights-only" placeholders.
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
    /// <param name="numRoutedExperts">Number of routed experts in the MoE FFN. Used only for weight enumeration; the single-expert fallback always runs experts[0].</param>
    /// <param name="qkNormEps">RMSNorm epsilon for the per-stream Q/K norms.</param>
    public HiDreamBlock(int hiddenSize, int numHeads, int headDim, int ffDim,
        bool isSingle, int numRoutedExperts, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _ffDim = ffDim;
        _isSingle = isSingle;
        _qkNormEps = qkNormEps;
        _numRoutedExperts = numRoutedExperts;
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

    /// <summary>Yields every weight tensor in this block — including all routed experts even though the
    /// single-expert fallback only consumes <c>experts[0]</c>. This keeps the GPU preload footprint and
    /// checkpoint-conversion path honest, and matches what a future full-MoE forward will need anyway.</summary>
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
        int totalSeqLen = imgSeqLen + txtSeqLen;
        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);

        // ── 1. AdaLN modulation: 12 params (image: 6, text: 6) ──
        Tensor[] mods = ComputeAdaLnParams(backend, temb, batch, 12);
        // mods order matches diffusers: shift_msa_i, scale_msa_i, gate_msa_i,
        //                                shift_mlp_i, scale_mlp_i, gate_mlp_i,
        //                                shift_msa_t, scale_msa_t, gate_msa_t,
        //                                shift_mlp_t, scale_mlp_t, gate_mlp_t

        // ── 2. Pre-attention norms + AdaLN modulation ──
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        DiTUtils.LayerNormNoAffine(imgNormed, image, batch, imgSeqLen, _hiddenSize);
        Tensor imgMod = AdaLNModulation.ApplyModulation(imgNormed, mods[0], mods[1], batch, imgSeqLen, _hiddenSize);
        imgNormed.Dispose();

        Tensor txtNormed = new Tensor(txtShape, DType.F32);
        DiTUtils.LayerNormNoAffine(txtNormed, text, batch, txtSeqLen, _hiddenSize);
        Tensor txtMod = AdaLNModulation.ApplyModulation(txtNormed, mods[6], mods[7], batch, txtSeqLen, _hiddenSize);
        txtNormed.Dispose();

        // ── 3. Joint MM-attention ──
        Tensor imgAttnOut, txtAttnOut;
        (imgAttnOut, txtAttnOut) = JointAttention(backend, imgMod, txtMod, rope, totalRopeSeqLen, batch, imgSeqLen, txtSeqLen);
        imgMod.Dispose();
        txtMod.Dispose();

        // ── 4. Gated residuals: hidden = hidden + gate * attn_out ──
        Tensor imgAfterAttn = AdaLNModulation.ApplyGatedResidual(image, imgAttnOut, mods[2], batch, imgSeqLen, _hiddenSize);
        imgAttnOut.Dispose();
        Tensor txtAfterAttn = AdaLNModulation.ApplyGatedResidual(text, txtAttnOut, mods[8], batch, txtSeqLen, _hiddenSize);
        txtAttnOut.Dispose();

        // ── 5. Image FFN (MoE single-expert fallback) ──
        Tensor imgPreFfn = new Tensor(imgShape, DType.F32);
        DiTUtils.LayerNormNoAffine(imgPreFfn, imgAfterAttn, batch, imgSeqLen, _hiddenSize);
        Tensor imgFfnIn = AdaLNModulation.ApplyModulation(imgPreFfn, mods[3], mods[4], batch, imgSeqLen, _hiddenSize);
        imgPreFfn.Dispose();

        // TODO: full MoE routing — currently using single-expert path (shared + experts[0]).
        Tensor imgFfnOut = MoeForwardSingleExpert(backend, imgFfnIn, batch, imgSeqLen);
        imgFfnIn.Dispose();

        Tensor imgFinal = AdaLNModulation.ApplyGatedResidual(imgAfterAttn, imgFfnOut, mods[5], batch, imgSeqLen, _hiddenSize);
        imgFfnOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 6. Text FFN (vanilla SwiGLU) ──
        Tensor txtPreFfn = new Tensor(txtShape, DType.F32);
        DiTUtils.LayerNormNoAffine(txtPreFfn, txtAfterAttn, batch, txtSeqLen, _hiddenSize);
        Tensor txtFfnIn = AdaLNModulation.ApplyModulation(txtPreFfn, mods[9], mods[10], batch, txtSeqLen, _hiddenSize);
        txtPreFfn.Dispose();

        Tensor txtFfnOut = SwiGluForward(backend, txtFfnIn, _ffTW1!, _ffTW3!, _ffTW2!, batch, txtSeqLen, _ffDim);
        txtFfnIn.Dispose();

        Tensor txtFinal = AdaLNModulation.ApplyGatedResidual(txtAfterAttn, txtFfnOut, mods[11], batch, txtSeqLen, _hiddenSize);
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
        int txtSeqLen = seqLen - imgSeqLen;
        TensorShape hiddenShape = new TensorShape(batch, seqLen, _hiddenSize);

        // ── 1. AdaLN: 6 params ──
        Tensor[] mods = ComputeAdaLnParams(backend, temb, batch, 6);
        // shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i

        // ── 2. Pre-attention norm + modulate ──
        Tensor preAttn = new Tensor(hiddenShape, DType.F32);
        DiTUtils.LayerNormNoAffine(preAttn, hidden, batch, seqLen, _hiddenSize);
        Tensor preAttnMod = AdaLNModulation.ApplyModulation(preAttn, mods[0], mods[1], batch, seqLen, _hiddenSize);
        preAttn.Dispose();

        // ── 3. Self-attention over the joint sequence (only image side has RoPE; text positions are zero) ──
        Tensor attnOut = SingleStreamSelfAttention(backend, preAttnMod, rope, totalRopeSeqLen, batch, seqLen, imgSeqLen, txtSeqLen);
        preAttnMod.Dispose();

        Tensor afterAttn = AdaLNModulation.ApplyGatedResidual(hidden, attnOut, mods[2], batch, seqLen, _hiddenSize);
        attnOut.Dispose();

        // ── 4. MoE FFN over the joint sequence ──
        Tensor preFfn = new Tensor(hiddenShape, DType.F32);
        DiTUtils.LayerNormNoAffine(preFfn, afterAttn, batch, seqLen, _hiddenSize);
        Tensor ffnIn = AdaLNModulation.ApplyModulation(preFfn, mods[3], mods[4], batch, seqLen, _hiddenSize);
        preFfn.Dispose();

        // TODO: full MoE routing — currently using single-expert path (shared + experts[0]).
        Tensor ffnOut = MoeForwardSingleExpert(backend, ffnIn, batch, seqLen);
        ffnIn.Dispose();

        Tensor result = AdaLNModulation.ApplyGatedResidual(afterAttn, ffnOut, mods[5], batch, seqLen, _hiddenSize);
        ffnOut.Dispose();
        afterAttn.Dispose();

        for (int i = 0; i < mods.Length; i++) mods[i].Dispose();
        return result;
    }

    /// <summary>Computes <paramref name="numParams"/> AdaLN modulation tensors (each [B, hiddenSize])
    /// from a [B, hiddenSize] timestep embedding using SiLU + Linear. <paramref name="numParams"/> is
    /// 12 for double blocks, 6 for single blocks, 2 for the final layer.</summary>
    private Tensor[] ComputeAdaLnParams(IBackend backend, Tensor temb, int batch, int numParams)
    {
        int outDim = numParams * _hiddenSize;
        TensorShape inShape = new TensorShape(batch, _hiddenSize);
        Tensor activated = new Tensor(inShape, DType.F32);
        backend.Silu(activated, temb);

        TensorShape outShape = new TensorShape(batch, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, activated, _adaLnLinearWeight!, _adaLnLinearBias);
        activated.Dispose();

        Tensor[] results = new Tensor[numParams];
        float* projPtr = (float*)projected.DataPointer;

        for (int p = 0; p < numParams; p++)
        {
            TensorShape paramShape = new TensorShape(batch, _hiddenSize);
            Tensor param = new Tensor(paramShape, DType.F32);
            float* pPtr = (float*)param.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int srcOff = b * outDim + p * _hiddenSize;
                int dstOff = b * _hiddenSize;
                for (int d = 0; d < _hiddenSize; d++)
                    pPtr[dstOff + d] = projPtr[srcOff + d];
            }
            results[p] = param;
        }

        projected.Dispose();
        return results;
    }

    /// <summary>Joint MM-attention shared between image and text in double-stream blocks. Image and text
    /// each get their own Q/K/V (image via <c>to_q/k/v</c>, text via <c>to_q_t/k_t/v_t</c>); both are RMS-normed
    /// per stream, RoPE is applied to the image-side Q/K (text positions are zero so RoPE is identity), then
    /// SDPA is run over the concatenated [img_tokens, text_tokens] sequence and the result is split back.</summary>
    private (Tensor img, Tensor txt) JointAttention(IBackend backend, Tensor img, Tensor txt,
        HiDreamRope rope, int totalRopeSeqLen, int batch, int imgSeqLen, int txtSeqLen)
    {
        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        int totalSeqLen = imgSeqLen + txtSeqLen;
        TensorShape totalMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        // Q/K/V projections (image side)
        Tensor imgQ = LinearAlloc(backend, img, _toQWeight!, _toQBias, imgShape);
        Tensor imgK = LinearAlloc(backend, img, _toKWeight!, _toKBias, imgShape);
        Tensor imgV = LinearAlloc(backend, img, _toVWeight!, _toVBias, imgShape);

        // RMSNorm Q and K over the flat last dim (= inner_dim).
        Tensor imgQNormed = new Tensor(imgShape, DType.F32);
        Tensor imgKNormed = new Tensor(imgShape, DType.F32);
        backend.RmsNorm(imgQNormed, imgQ, _qRmsNormWeight!, _qkNormEps);
        backend.RmsNorm(imgKNormed, imgK, _kRmsNormWeight!, _qkNormEps);
        imgQ.Dispose(); imgK.Dispose();

        // Q/K/V projections (text side)
        Tensor txtQ = LinearAlloc(backend, txt, _toQTWeight!, _toQTBias, txtShape);
        Tensor txtK = LinearAlloc(backend, txt, _toKTWeight!, _toKTBias, txtShape);
        Tensor txtV = LinearAlloc(backend, txt, _toVTWeight!, _toVTBias, txtShape);

        Tensor txtQNormed = new Tensor(txtShape, DType.F32);
        Tensor txtKNormed = new Tensor(txtShape, DType.F32);
        backend.RmsNorm(txtQNormed, txtQ, _qRmsNormTWeight!, _qkNormEps);
        backend.RmsNorm(txtKNormed, txtK, _kRmsNormTWeight!, _qkNormEps);
        txtQ.Dispose(); txtK.Dispose();

        // Reshape to multi-head [B, H, S, D]
        TensorShape imgMhShape = new TensorShape(batch, _numHeads, imgSeqLen, _headDim);
        TensorShape txtMhShape = new TensorShape(batch, _numHeads, txtSeqLen, _headDim);

        Tensor imgQMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgKMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgVMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(imgQMh, imgQNormed, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgKMh, imgKNormed, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgVMh, imgV, batch, imgSeqLen, _numHeads, _headDim);
        imgQNormed.Dispose(); imgKNormed.Dispose(); imgV.Dispose();

        Tensor txtQMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtKMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtVMh = new Tensor(txtMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(txtQMh, txtQNormed, batch, txtSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(txtKMh, txtKNormed, batch, txtSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(txtVMh, txtV, batch, txtSeqLen, _numHeads, _headDim);
        txtQNormed.Dispose(); txtKNormed.Dispose(); txtV.Dispose();

        // RoPE on image-side Q/K only — RoPE expects [B, H, totalRopeSeqLen, D] tables, but image
        // tokens occupy the first imgSeqLen positions in the rope table by convention.
        ApplyRopeImageOnly(imgQMh, imgKMh, rope, batch, imgSeqLen);

        // Concatenate image then text along the seq axis (multi-head layout).
        Tensor jointQ = new Tensor(totalMhShape, DType.F32);
        Tensor jointK = new Tensor(totalMhShape, DType.F32);
        Tensor jointV = new Tensor(totalMhShape, DType.F32);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointQ, imgQMh, txtQMh, batch, _numHeads, imgSeqLen, txtSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointK, imgKMh, txtKMh, batch, _numHeads, imgSeqLen, txtSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointV, imgVMh, txtVMh, batch, _numHeads, imgSeqLen, txtSeqLen, _headDim);
        imgQMh.Dispose(); imgKMh.Dispose(); imgVMh.Dispose();
        txtQMh.Dispose(); txtKMh.Dispose(); txtVMh.Dispose();

        // SDPA
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttn = new Tensor(totalMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttn, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose(); jointK.Dispose(); jointV.Dispose();

        // Split back to image / text and reshape to [B, S, hidden]
        Tensor imgAttnMh = new Tensor(imgMhShape, DType.F32);
        Tensor txtAttnMh = new Tensor(txtMhShape, DType.F32);
        DiTUtils.SplitAlongSeqDimMultiHead(imgAttnMh, txtAttnMh, jointAttn, batch, _numHeads, imgSeqLen, txtSeqLen, _headDim);
        jointAttn.Dispose();

        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(imgAttn, imgAttnMh, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeFromMultiHead(txtAttn, txtAttnMh, batch, txtSeqLen, _numHeads, _headDim);
        imgAttnMh.Dispose(); txtAttnMh.Dispose();

        // Output projections
        Tensor imgOut = LinearAlloc(backend, imgAttn, _toOutWeight!, _toOutBias, imgShape);
        Tensor txtOut = LinearAlloc(backend, txtAttn, _toOutTWeight!, _toOutTBias, txtShape);
        imgAttn.Dispose();
        txtAttn.Dispose();

        return (imgOut, txtOut);
    }

    /// <summary>Self-attention over the concatenated single-stream sequence. The image-side weights
    /// (<c>to_q/k/v</c>) are reused for the entire sequence; only the image-token Q/K positions get RoPE
    /// (text positions in the rope table are zeros and are passed through unchanged by the rotation).</summary>
    private Tensor SingleStreamSelfAttention(IBackend backend, Tensor hidden, HiDreamRope rope, int totalRopeSeqLen,
        int batch, int seqLen, int imgSeqLen, int txtSeqLen)
    {
        TensorShape hiddenShape = new TensorShape(batch, seqLen, _hiddenSize);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        Tensor q = LinearAlloc(backend, hidden, _toQWeight!, _toQBias, hiddenShape);
        Tensor k = LinearAlloc(backend, hidden, _toKWeight!, _toKBias, hiddenShape);
        Tensor v = LinearAlloc(backend, hidden, _toVWeight!, _toVBias, hiddenShape);

        Tensor qNormed = new Tensor(hiddenShape, DType.F32);
        Tensor kNormed = new Tensor(hiddenShape, DType.F32);
        backend.RmsNorm(qNormed, q, _qRmsNormWeight!, _qkNormEps);
        backend.RmsNorm(kNormed, k, _kRmsNormWeight!, _qkNormEps);
        q.Dispose(); k.Dispose();

        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(qMh, qNormed, batch, seqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(kMh, kNormed, batch, seqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(vMh, v, batch, seqLen, _numHeads, _headDim);
        qNormed.Dispose(); kNormed.Dispose(); v.Dispose();

        // RoPE applies position-by-position; zero-position text tokens just rotate by 0 (identity).
        // We rotate over min(seqLen, totalRopeSeqLen) positions — the text segment past imgSeqLen
        // uses the all-zero rope entries from BuildPositionIds.
        ApplyRopeFullSeq(qMh, kMh, rope, batch, seqLen);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attn = new Tensor(hiddenShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(attn, attnMh, batch, seqLen, _numHeads, _headDim);
        attnMh.Dispose();

        Tensor outProj = LinearAlloc(backend, attn, _toOutWeight!, _toOutBias, hiddenShape);
        attn.Dispose();
        return outProj;
    }

    /// <summary>Single-expert MoE forward (fallback). Computes `shared_expert(x) + gate_weight_for_e0 *
    /// experts[0](x)` for every token, where the per-token gate weight comes from the softmax-and-pick
    /// of the routing logits restricted to experts[0]. Shape is correct; values diverge from a true
    /// top-k=2 MoE forward in proportion to how often the trained gate selects experts[1..3].
    /// TODO: full MoE routing.</summary>
    private Tensor MoeForwardSingleExpert(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape inShape = new TensorShape(batch, seqLen, _hiddenSize);

        // Shared expert: SwiGLU at hidden_dim/2
        int sharedDim = (_ffDim + 1) / 2; // ff_dim // 2 in Python (integer division of 4*hidden / 2)
        Tensor shared = SwiGluForward(backend, input, _sharedW1!, _sharedW3!, _sharedW2!, batch, seqLen, sharedDim);

        // Routed expert 0 only (single-expert fallback)
        Tensor expert0 = SwiGluForward(backend, input, _expertW1![0], _expertW3![0], _expertW2![0], batch, seqLen, _ffDim);

        // Per-token gate weight for expert 0: softmax(input @ gate_weight^T)[..., 0]
        Tensor gateLogits = new Tensor(new TensorShape(batch, seqLen, _numRoutedExperts), DType.F32);
        backend.Linear(gateLogits, input, _moeGateWeight!, null);
        Tensor gateWeights = new Tensor(new TensorShape(batch, seqLen), DType.F32);
        ComputeExpert0SoftmaxWeight(gateLogits, gateWeights, batch, seqLen, _numRoutedExperts);
        gateLogits.Dispose();

        // Combine: out = shared + gate0 * expert0
        Tensor combined = new Tensor(inShape, DType.F32);
        ApplyGatedExpert(combined, shared, expert0, gateWeights, batch, seqLen, _hiddenSize);
        shared.Dispose();
        expert0.Dispose();
        gateWeights.Dispose();
        return combined;
    }

    /// <summary>SwiGLU forward: <c>w2(silu(w1(x)) * w3(x))</c>. Used by the text FFN, the shared MoE
    /// expert, and (under the fallback) the single routed expert.</summary>
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

    /// <summary>Applies the image-side rope to the first imgSeqLen positions of Q and K; the text segment
    /// (positions imgSeqLen..total) is left untouched. The rope was precomputed in the caller for the full
    /// image+text sequence with text positions = 0, so a single Forward call over <paramref name="imgSeqLen"/>
    /// rotates exactly the image tokens.</summary>
    private static void ApplyRopeImageOnly(Tensor q, Tensor k, HiDreamRope rope, int batch, int imgSeqLen)
    {
        // The rope cache is sized by the latest Precompute call. Forward rotates the first
        // imgSeqLen positions of each [B, H, S, D] tensor — anything beyond is ignored by Forward.
        rope.Forward(q, k, batch, q.ShapeValue(1), imgSeqLen);
    }

    /// <summary>Applies rope across the full single-stream sequence. Text tokens have all-zero positions
    /// in the rope table (set by HiDreamRope.BuildPositionIds), so their rotation is the identity.</summary>
    private static void ApplyRopeFullSeq(Tensor q, Tensor k, HiDreamRope rope, int batch, int seqLen)
    {
        rope.Forward(q, k, batch, q.ShapeValue(1), seqLen);
    }

    /// <summary>Reads the softmax of routing logits and emits the per-token weight for expert 0.</summary>
    private static void ComputeExpert0SoftmaxWeight(Tensor logits, Tensor weights, int batch, int seqLen, int numExperts)
    {
        float* lp = (float*)logits.DataPointer;
        float* wp = (float*)weights.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int off = (b * seqLen + s) * numExperts;
                float maxLogit = lp[off];
                for (int e = 1; e < numExperts; e++)
                    if (lp[off + e] > maxLogit) maxLogit = lp[off + e];
                float sum = 0f;
                for (int e = 0; e < numExperts; e++)
                    sum += MathF.Exp(lp[off + e] - maxLogit);
                wp[b * seqLen + s] = MathF.Exp(lp[off + 0] - maxLogit) / sum;
            }
        }
    }

    /// <summary>output = shared + gate0 * expert0, broadcasting gate0 ([B, S]) over the hidden dim.</summary>
    private static void ApplyGatedExpert(Tensor output, Tensor shared, Tensor expert0, Tensor gate0,
        int batch, int seqLen, int hiddenSize)
    {
        float* op = (float*)output.DataPointer;
        float* sp = (float*)shared.DataPointer;
        float* ep = (float*)expert0.DataPointer;
        float* gp = (float*)gate0.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                float g = gp[b * seqLen + s];
                int off = (b * seqLen + s) * hiddenSize;
                for (int d = 0; d < hiddenSize; d++)
                    op[off + d] = sp[off + d] + g * ep[off + d];
            }
        }
    }
}

/// <summary>Tiny helper to read a tensor shape dim by index without a managed cast for every use site.</summary>
internal static class HiDreamTensorExtensions
{
    public static int ShapeValue(this Tensor t, int axis) => (int)t.Shape[axis];
}
