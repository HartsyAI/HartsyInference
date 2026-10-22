using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>HiDream-I1 transformer block. Two flavors share the same code path:
/// <list type="bullet">
/// <item>Double-stream (joint MM-attention): image and text get separate Q/K/V projections, RMSNorm Q/K, RoPE on the image side, joint SDPA over the concatenated sequence, separate output projections, separate FFNs (MoE on the image side, vanilla SwiGLU on the text side). 12-way AdaLN modulation (6 per stream).</item>
/// <item>Single-stream: text is already concatenated to the image sequence by the caller; one set of Q/K/V projections, one SDPA, one MoE FFN on the joint sequence. 6-way AdaLN modulation.</item>
/// </list>
/// <para><b>MoE FFN (full top-k routing):</b> the image-side FFN is HiDream's sparse MoE — a softmax gate over
/// <c>num_routed_experts</c> (4) experts, top-k (<c>num_activated_experts</c> = 2) selection per token with the
/// selected gate weights renormalized to sum to 1, plus an always-on shared expert added on top. Because the
/// <see cref="IBackend"/> exposes no grouped-GEMM / sort-scatter expert-dispatch primitive, this block runs the
/// routed forward densely (every expert evaluated for every token) and zeroes non-selected experts via the
/// per-token gate weight — numerically identical to a true sparse dispatch, just with extra compute. The
/// single-expert path (shared + experts[0]) is kept only as a fallback when <c>num_activated_experts &lt;= 1</c>.
/// See <see cref="MoeForward"/> for the routing/renorm formula.</para></summary>
public sealed unsafe class HiDreamBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly bool _isSingle;
    private readonly float _qkNormEps;
    private readonly int _numActivatedExperts;

    private JointAttention? _jointAttention;
    private AttentionLayer? _selfAttention, _selfAttentionBatched;

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

        // Q/K are RMS-normed over the full width, so one batch item may narrow to F16; a batch keeps F32 attention.
        AttentionLayer image = Stream(_toQWeight!, _toQBias, _toKWeight!, _toKBias, _toVWeight!, _toVBias, _toOutWeight!, _toOutBias,
            _qRmsNormWeight!, _kRmsNormWeight!, allowF16: true, tokenMajor: _isSingle);
        if (_isSingle)
        {
            _selfAttention = image;
            _selfAttentionBatched = Stream(_toQWeight!, _toQBias, _toKWeight!, _toKBias, _toVWeight!, _toVBias, _toOutWeight!,
                _toOutBias, _qRmsNormWeight!, _kRmsNormWeight!, allowF16: false);
        }
        else
        {
            _jointAttention = new JointAttention(
                new JointAttentionConfig
                {
                    Heads = _numHeads, HeadDim = _headDim, ConcatOrder = [0, 1], AllowF16 = true, AllowF16Batched = false,
                    StageOrder = JointStageOrder.ProjectNormPerStream, TokenMajorRoute = AttentionRoute.TokenMajor,
                    HeadMajorForSage = true,
                },
                [image, Stream(_toQTWeight!, _toQTBias, _toKTWeight!, _toKTBias, _toVTWeight!, _toVTBias, _toOutTWeight!, _toOutTBias,
                    _qRmsNormTWeight!, _kRmsNormTWeight!, allowF16: true)]);
        }
    }

    private AttentionLayer Stream(Tensor q, Tensor? qBias, Tensor k, Tensor? kBias, Tensor v, Tensor? vBias, Tensor output,
        Tensor? outputBias, Tensor qNorm, Tensor kNorm, bool allowF16, bool tokenMajor = false) => new(
        new AttentionLayerConfig
        {
            Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.FullWidthRms,
            ComputeDType = DType.F32, AllowF16 = allowF16,
            TokenMajorRoute = tokenMajor ? AttentionRoute.TokenMajor : AttentionRoute.Dense, HeadMajorForSage = tokenMajor,
        },
        new AttentionLayerWeights
        {
            Query = q, QueryBias = qBias, Key = k, KeyBias = kBias, Value = v, ValueBias = vBias, Output = output,
            OutputBias = outputBias, QueryNorm = qNorm, KeyNorm = kNorm, QueryNormEps = _qkNormEps, KeyNormEps = _qkNormEps,
        });

    /// <summary>Yields every weight tensor in this block, including all routed experts (all of which participate in the top-k routed forward).</summary>
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
        backend.LayerNormNoAffine(imgNormed, image, 1e-6f);
        Tensor imgMod = DiTUtils.Modulate(backend, imgNormed, mods[0], mods[1], imgShape);
        imgNormed.Dispose();

        Tensor txtNormed = new Tensor(txtShape, DType.F32);
        backend.LayerNormNoAffine(txtNormed, text, 1e-6f);
        Tensor txtMod = DiTUtils.Modulate(backend, txtNormed, mods[6], mods[7], txtShape);
        txtNormed.Dispose();

        // ── 3. Joint MM-attention ──
        Tensor imgAttnOut, txtAttnOut;
        (imgAttnOut, txtAttnOut) = JointStreamAttention(backend, imgMod, txtMod, rope, batch, imgSeqLen);
        imgMod.Dispose();
        txtMod.Dispose();

        // ── 4. Gated residuals: hidden = hidden + gate * attn_out ──
        Tensor imgAfterAttn = GpuGatedResidual(backend, image, imgAttnOut, mods[2], batch, imgSeqLen, _hiddenSize);
        imgAttnOut.Dispose();
        Tensor txtAfterAttn = GpuGatedResidual(backend, text, txtAttnOut, mods[8], batch, txtSeqLen, _hiddenSize);
        txtAttnOut.Dispose();

        // ── 5. Image FFN (MoE single-expert fallback) ──
        Tensor imgPreFfn = new Tensor(imgShape, DType.F32);
        backend.LayerNormNoAffine(imgPreFfn, imgAfterAttn, 1e-6f);
        Tensor imgFfnIn = DiTUtils.Modulate(backend, imgPreFfn, mods[3], mods[4], imgShape);
        imgPreFfn.Dispose();

        // VALIDATION-PENDING: verify vs diffusers HiDreamImageTransformer2DModel
        Tensor imgFfnOut = MoeForward(backend, imgFfnIn, batch, imgSeqLen);
        imgFfnIn.Dispose();

        Tensor imgFinal = GpuGatedResidual(backend, imgAfterAttn, imgFfnOut, mods[5], batch, imgSeqLen, _hiddenSize);
        imgFfnOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 6. Text FFN (vanilla SwiGLU) ──
        Tensor txtPreFfn = new Tensor(txtShape, DType.F32);
        backend.LayerNormNoAffine(txtPreFfn, txtAfterAttn, 1e-6f);
        Tensor txtFfnIn = DiTUtils.Modulate(backend, txtPreFfn, mods[9], mods[10], txtShape);
        txtPreFfn.Dispose();

        Tensor txtFfnOut = SwiGluForward(backend, txtFfnIn, _ffTW1!, _ffTW3!, _ffTW2!, batch, txtSeqLen);
        txtFfnIn.Dispose();

        Tensor txtFinal = GpuGatedResidual(backend, txtAfterAttn, txtFfnOut, mods[11], batch, txtSeqLen, _hiddenSize);
        txtFfnOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < mods.Length; i++) mods[i].Dispose();
        return (imgFinal, txtFinal);
    }

    /// <summary>Forward pass for single-stream blocks. The hidden states already contain image tokens followed by all relevant text tokens; one set of Q/K/V projections is run over the joint sequence.</summary>
    public Tensor ForwardSingle(IBackend backend, Tensor hidden, Tensor temb, HiDreamRope rope, int imgSeqLen, int totalRopeSeqLen)
    {
        if (!_isSingle)
            throw new InvalidOperationException("ForwardSingle called on a double-stream block.");

        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        TensorShape hiddenShape = new TensorShape(batch, seqLen, _hiddenSize);

        // ── 1. AdaLN: 6 params ──
        Tensor[] mods = ComputeAdaLnParams(backend, temb, batch, 6);
        // shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i

        // ── 2. Pre-attention norm + modulate ──
        Tensor preAttn = new Tensor(hiddenShape, DType.F32);
        backend.LayerNormNoAffine(preAttn, hidden, 1e-6f);
        Tensor preAttnMod = DiTUtils.Modulate(backend, preAttn, mods[0], mods[1], hiddenShape);
        preAttn.Dispose();

        // ── 3. Self-attention over the joint sequence (only image side has RoPE; text positions are zero) ──
        Tensor attnOut = SingleStreamSelfAttention(backend, preAttnMod, rope, batch);
        preAttnMod.Dispose();

        Tensor afterAttn = GpuGatedResidual(backend, hidden, attnOut, mods[2], batch, seqLen, _hiddenSize);
        attnOut.Dispose();

        // ── 4. MoE FFN over the joint sequence ──
        Tensor preFfn = new Tensor(hiddenShape, DType.F32);
        backend.LayerNormNoAffine(preFfn, afterAttn, 1e-6f);
        Tensor ffnIn = DiTUtils.Modulate(backend, preFfn, mods[3], mods[4], hiddenShape);
        preFfn.Dispose();

        // VALIDATION-PENDING: verify vs diffusers HiDreamImageTransformer2DModel
        Tensor ffnOut = MoeForward(backend, ffnIn, batch, seqLen);
        ffnIn.Dispose();

        Tensor result = GpuGatedResidual(backend, afterAttn, ffnOut, mods[5], batch, seqLen, _hiddenSize);
        ffnOut.Dispose();
        afterAttn.Dispose();

        for (int i = 0; i < mods.Length; i++) mods[i].Dispose();
        return result;
    }

    /// <summary>Computes <paramref name="numParams"/> AdaLN modulation tensors (each [B, hiddenSize]) from a [B, hiddenSize] timestep embedding using SiLU + Linear. <paramref name="numParams"/> is 12 for double blocks, 6 for single blocks, 2 for the final layer.</summary>
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

        // GPU-resident split: SliceLastDim extracts each [B, hiddenSize] chunk at offset p*hiddenSize.
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

    /// <summary>GPU-resident gated residual: output = input + gate * value, gate [B, hidden] broadcast over the sequence.</summary>
    private static Tensor GpuGatedResidual(IBackend backend, Tensor input, Tensor value, Tensor gate, int batch, int seqLen, int hiddenSize)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, hiddenSize), DType.F32);
        backend.GatedResidualLastDim(output, input, value, gate);
        return output;
    }

    /// <summary>Joint MM-attention: image and text project and RMS-norm over the full width with their own weights, then one
    /// attention over <c>[img, txt]</c> and the split back. One batch item rotates the whole joint sequence on the device (text
    /// positions are zero, an identity); a batch rotates the image rows only.</summary>
    private (Tensor img, Tensor txt) JointStreamAttention(IBackend backend, Tensor img, Tensor txt, HiDreamRope rope, int batch,
        int imgSeqLen)
    {
        AttentionCallContext joint = batch == 1
            ? new AttentionCallContext { Rotation = rope }
            : new AttentionCallContext { HeadMajorRotation = new LeadingRowsRotation(imgSeqLen, rope) };
        Span<Tensor?> attended = [null, null];
        _jointAttention!.Attend(backend, [img, txt], attended, new JointCallContext { Joint = joint });
        Tensor imgOut = _jointAttention.ProjectOutput(backend, 0, attended[0]!);
        Tensor txtOut = _jointAttention.ProjectOutput(backend, 1, attended[1]!);
        return (imgOut, txtOut);
    }

    /// <summary>Self-attention over the single-stream sequence with the image-side weights; text positions rotate by identity.</summary>
    private Tensor SingleStreamSelfAttention(IBackend backend, Tensor hidden, HiDreamRope rope, int batch)
    {
        AttentionCallContext context = batch == 1
            ? new AttentionCallContext { Rotation = rope }
            : new AttentionCallContext { HeadMajorRotation = rope };
        return (batch == 1 ? _selfAttention! : _selfAttentionBatched!).Forward(backend, hidden, hidden, in context);
    }

    /// <summary>MoE SwiGLU FFN: <c>shared_experts(x) + sum_k(topk_weight_k * experts[idx_k](x))</c>.
    /// <para>Mirrors diffusers HiDream <c>MoEGate</c> + <c>MOEFeedForwardSwiGLU</c>:
    /// the gate Linear produces <c>[B, S, num_routed_experts]</c> logits; softmax is taken over the
    /// expert axis (dim=-1); the top-k (= <c>num_activated_experts</c>) experts per token are selected and
    /// their gate weights are renormalized to sum to 1 (<c>topk_weight / (topk_weight.sum(-1) + 1e-20)</c>,
    /// the <c>norm_topk_prob=True</c> path HiDream uses since top_k &gt; 1). Each token is then run through its
    /// selected experts and the outputs are weighted by the renormalized gate weights and summed. The shared
    /// (always-on) expert output is added on top.</para>
    /// <para>VALIDATION-PENDING: verify vs diffusers HiDreamImageTransformer2DModel. This eager per-token-mask
    /// implementation runs <i>all</i> routed experts densely (no grouped-GEMM dispatch — the backend has no
    /// scatter/gather expert primitive) and zeroes the non-selected experts' contributions via the per-token
    /// gate weight, which is numerically identical to a true top-k dispatch but does extra compute. If
    /// <c>num_activated_experts &lt;= 1</c> the routing collapses to top-1 (each token routes to its single
    /// argmax expert with renormalized weight 1) plus the shared expert.</para></summary>
    private Tensor MoeForward(IBackend backend, Tensor input, int batch, int seqLen)
    {
        // Shared (always-on) expert: SwiGLU. Inner dim is taken from the shared weight (3584 for HiDream V1),
        // which is NOT ff_dim/2 — so it must come from the weight, not a computed value.
        Tensor shared = SwiGluForward(backend, input, _sharedW1!, _sharedW3!, _sharedW2!, batch, seqLen);

        // Gate logits: input @ gate_weight^T → [B, S, num_routed_experts].
        Tensor gateLogits = new Tensor(new TensorShape(batch, seqLen, _numRoutedExperts), DType.F32);
        backend.Linear(gateLogits, input, _moeGateWeight!, null);

        // Per-token, per-expert renormalized top-k gate weights (0 for non-selected experts). [B, S, E].
        // Device op (was a host loop reading the device-produced logits' DataPointer — a full pipeline
        // drain per block per forward; with the host per-expert accumulate below it dominated the
        // ~29s/step wall). With top-k=1 (or fewer) this collapses to the single-expert fallback.
        Tensor gateWeights = new Tensor(new TensorShape(batch, seqLen, _numRoutedExperts), DType.F32);
        int activeK = _numActivatedExperts <= 1 ? 1 : _numActivatedExperts;
        backend.MoeTopKGate(gateWeights, gateLogits, activeK);
        gateLogits.Dispose();

        // The shared (always-on) expert output is the accumulator; each routed expert adds in place,
        // weighted by its (possibly zero) gate — all device-resident, no D2H.
        for (int e = 0; e < _numRoutedExperts; e++)
        {
            Tensor expertOut = SwiGluForward(backend, input, _expertW1![e], _expertW3![e], _expertW2![e], batch, seqLen);
            backend.RowGatedAccumulate(shared, expertOut, gateWeights, _numRoutedExperts, e);
            expertOut.Dispose();
        }
        gateWeights.Dispose();
        return shared;
    }

    /// <summary>SwiGLU forward: <c>w2(silu(w1(x)) * w3(x))</c>. Used by the text FFN, the shared MoE expert, and (under the fallback) the single routed expert.</summary>
    private Tensor SwiGluForward(IBackend backend, Tensor input, Tensor w1, Tensor w3, Tensor w2, int batch, int seqLen)
    {
        // The SwiGLU inner dim MUST come from the weight (w1.Shape[0] = the GEMM's N), not a computed ffDim.
        // HiDream's checkpoint uses the Llama SwiGLU width round_up(8/3·dim, 256) = 6912 (routed experts / text
        // FFN) and 3584 (shared expert), NOT 4·dim = 10240 / ff_dim//2 = 5120. Allocating with a computed dim that
        // disagrees with the weight makes backend.Linear (which takes N from the weight) write into a mis-sized
        // buffer, corrupting the FFN and the whole image. Deriving from the weight keeps buffer and GEMM in lockstep.
        int hiddenInner = (int)w1.Shape[0];
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
}
