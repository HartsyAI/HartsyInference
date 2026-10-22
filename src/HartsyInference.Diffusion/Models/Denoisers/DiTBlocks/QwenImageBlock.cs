using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Qwen-Image dual-stream MMDiT block (<c>QwenImageTransformerBlock</c>). Maintains separate image and text streams with independent AdaLN-Zero modulation (12 params each: <c>shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp</c> for both <c>mod1</c> and <c>mod2</c>), QK-norm, and GELU(approximate) FFNs. Joint attention concatenates <c>[txt, img]</c> along the sequence dim and applies precomputed RoPE separately to image and text (image-only RoPE for image tokens; zero-position RoPE rows for text tokens). Mirrors <c>diffusers/models/transformers/transformer_qwenimage.py:QwenImageTransformerBlock.forward</c> 1:1.</summary>
public sealed unsafe class QwenImageBlock : IStreamingBlock
{
    /// <inheritdoc/>
    /// <remarks>Via <see cref="DType.ComputeByteCount"/>, not <c>ElementCount * SizeInBytes</c>: Qwen-Image ships as a
    /// Q4_K GGUF, and block-quantized dtypes report <c>SizeInBytes == 0</c> — the naive form would total this 20B
    /// model's blocks to zero bytes and make the streaming budget believe the DiT is weightless.</remarks>
    public long EstimatedWeightBytes
    {
        get
        {
            long total = 0;
            foreach (Tensor w in EnumerateWeights()) total += w.DType.ComputeByteCount(w.ElementCount);
            return total;
        }
    }

    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

    private readonly AdaLNModulation _imgModulation;
    private readonly AdaLNModulation _txtModulation;

    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;
    private float _streamLnEps = 1e-6f;

    private Tensor? _addQWeight, _addQBias;
    private Tensor? _addKWeight, _addKBias;
    private Tensor? _addVWeight, _addVBias;
    private Tensor? _toAddOutWeight, _toAddOutBias;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;
    private readonly QkNorm _normAddedQ;
    private readonly QkNorm _normAddedK;

    private readonly SwiGluFfn _imgFfn;
    private readonly SwiGluFfn _txtFfn;

    private JointAttention? _attention;

    /// <summary>Creates a Qwen-Image dual-stream block.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for V1).</param>
    /// <param name="numHeads">Number of attention heads (24 for V1).</param>
    /// <param name="headDim">Per-head dimension (128 for V1; <c>numHeads * headDim == hiddenSize</c>).</param>
    /// <param name="mlpDim">MLP inner dimension (4 * hiddenSize = 12288 for V1).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon. Default 1e-6.</param>
    public QwenImageBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-6f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = mlpDim;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _txtModulation = new AdaLNModulation(hiddenSize, 6);

        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);
        _normAddedQ = new QkNorm(headDim, qkNormEps);
        _normAddedK = new QkNorm(headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights using diffusers naming under <c>transformer_blocks.{i}.*</c>. Modulation linears project temb to <c>2 * 6 * hidden</c> (mod1 + mod2 each with 6 params); attention is bias=True; QK-norm is RMSNorm; FFNs are GELU(approximate) with <c>ff.net.0.proj</c> + <c>ff.net.2</c> naming.</summary>
    /// <param name="branchDamp">Residual-stream damp for the F16 activation path (the exact Chroma/Flux
    /// recipe — see <see cref="ChromaF16"/>): damps every branch-output projection so the residual stream
    /// rides at 1/32 scale; the no-affine LayerNorms make it exact. 1.0 = off.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, float branchDamp = 1.0f)
    {
        // LayerNorm scale-invariance is only exact with eps scaled alongside: LN_eps(c·x) = c·LN_{eps/c²}(x).
        // At Qwen's deep damp (1/2048) the block-0 stream variance (~4e-6) is COMPARABLE to the raw 1e-6 eps,
        // which silently distorted the normalization (velocity std +37%, corr 0.995 — the damp "leak").
        _streamLnEps = 1e-6f * branchDamp * branchDamp;
        _imgModulation.LoadWeights(
            weights[$"{prefix}.img_mod.1.weight"],
            weights[$"{prefix}.img_mod.1.bias"]);
        _txtModulation.LoadWeights(
            weights[$"{prefix}.txt_mod.1.weight"],
            weights[$"{prefix}.txt_mod.1.bias"]);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];
        _toQBias = weights[$"{prefix}.attn.to_q.bias"];
        _toKBias = weights[$"{prefix}.attn.to_k.bias"];
        _toVBias = weights[$"{prefix}.attn.to_v.bias"];

        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];
        _addQBias = weights[$"{prefix}.attn.add_q_proj.bias"];
        _addKBias = weights[$"{prefix}.attn.add_k_proj.bias"];
        _addVBias = weights[$"{prefix}.attn.add_v_proj.bias"];

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
        _normAddedQ.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
        _normAddedK.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);

        Tensor imgFfnOutWeight = weights[$"{prefix}.img_mlp.net.2.weight"];
        Tensor imgFfnOutBias = weights[$"{prefix}.img_mlp.net.2.bias"];
        Tensor txtFfnOutWeight = weights[$"{prefix}.txt_mlp.net.2.weight"];
        Tensor txtFfnOutBias = weights[$"{prefix}.txt_mlp.net.2.bias"];
        if (branchDamp != 1.0f)
        {
            // Weight damp rides the GEMM alpha (honored on the dequantized-GGUF cuBLAS path too — every
            // damped weight sees token-count-sized GEMMs, never the alpha-less M=1 mat-vec kernels);
            // biases get value-scaled F32 copies (the epilogue adds bias after alpha). Once per load.
            _toOutWeight!.Fp8ScaleFactor *= branchDamp;
            _toOutBias = ChromaF16.DampBias(_toOutBias!, branchDamp);
            _toAddOutWeight!.Fp8ScaleFactor *= branchDamp;
            _toAddOutBias = ChromaF16.DampBias(_toAddOutBias!, branchDamp);
            imgFfnOutWeight.Fp8ScaleFactor *= branchDamp;
            imgFfnOutBias = ChromaF16.DampBias(imgFfnOutBias, branchDamp);
            txtFfnOutWeight.Fp8ScaleFactor *= branchDamp;
            txtFfnOutBias = ChromaF16.DampBias(txtFfnOutBias, branchDamp);
        }

        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.img_mlp.net.0.proj.weight"],
            weights[$"{prefix}.img_mlp.net.0.proj.bias"],
            imgFfnOutWeight,
            imgFfnOutBias);

        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.txt_mlp.net.0.proj.weight"],
            weights[$"{prefix}.txt_mlp.net.0.proj.bias"],
            txtFfnOutWeight,
            txtFfnOutBias);

        // Built after the damp, which replaces the output biases. QK RMS-norm bounds the scores, so the call may narrow to F16.
        _attention = new JointAttention(
            new JointAttentionConfig
            {
                Heads = _numHeads, HeadDim = _headDim, ConcatOrder = [1, 0], AllowF16 = true,
                TokenMajorRoute = AttentionRoute.TokenMajor, HeadMajorForSage = true,
            },
            [Stream(_toQWeight!, _toQBias, _toKWeight!, _toKBias, _toVWeight!, _toVBias, _toOutWeight!, _toOutBias, _normQ, _normK),
             Stream(_addQWeight!, _addQBias, _addKWeight!, _addKBias, _addVWeight!, _addVBias, _toAddOutWeight!, _toAddOutBias,
                 _normAddedQ, _normAddedK)]);
    }

    private AttentionLayer Stream(Tensor q, Tensor? qBias, Tensor k, Tensor? kBias, Tensor v, Tensor? vBias, Tensor output,
        Tensor? outputBias, QkNorm normQ, QkNorm normK) => new(
        new AttentionLayerConfig { Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead },
        new AttentionLayerWeights
        {
            Query = q, QueryBias = qBias, Key = k, KeyBias = kBias, Value = v, ValueBias = vBias, Output = output,
            OutputBias = outputBias, QueryNorm = normQ.Weight, KeyNorm = normK.Weight, QueryNormEps = normQ.Eps,
            KeyNormEps = normK.Eps,
        });

    /// <summary>The batched fallback: QwenImageRope's host rotation of the joint head-major Q and K.</summary>
    private sealed class JointRowsRotation(QwenImageRope rope, int packedH, int packedW, int txtSeqLen, int txtPositionStart,
        (int H, int W)[] refGrids) : IHeadMajorQkRotation
    {
        public void RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens) =>
            rope.ApplyJoint(q, k, batch, heads, packedH, packedW, txtSeqLen, txtPositionStart, refGrids);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _imgModulation.EnumerateWeights()) yield return w;
        foreach (Tensor w in _txtModulation.EnumerateWeights()) yield return w;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_toOutWeight is not null) yield return _toOutWeight;
        if (_toOutBias is not null) yield return _toOutBias;
        if (_addQWeight is not null) yield return _addQWeight;
        if (_addQBias is not null) yield return _addQBias;
        if (_addKWeight is not null) yield return _addKWeight;
        if (_addKBias is not null) yield return _addKBias;
        if (_addVWeight is not null) yield return _addVWeight;
        if (_addVBias is not null) yield return _addVBias;
        if (_toAddOutWeight is not null) yield return _toAddOutWeight;
        if (_toAddOutBias is not null) yield return _toAddOutBias;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normAddedQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normAddedK.EnumerateWeights()) yield return w;
        foreach (Tensor w in _imgFfn.EnumerateWeights()) yield return w;
        foreach (Tensor w in _txtFfn.EnumerateWeights()) yield return w;
    }

    /// <summary>Each modulation produces <c>[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]</c>. Joint attention concats <c>[txt, img]</c>; QK rotated by <see cref="QwenImageRope"/> separately for image (per-token <c>[0, row, col]</c>) and text (offset by <paramref name="txtPositionStart"/>) before concat. Returns <c>(image, text)</c>.</summary>
    // QwenImagePipeline runs CFG as two batch-1 passes; a batch above one rotates on the host after the head permute.
    /// <param name="refPackedH">Packed-grid height of the trailing Qwen-Image-Edit reference-latent token section
    /// (0 = no ref tokens). Ref tokens ride at the END of the image stream and get their own RoPE positions
    /// (frame axis 1, centered on the ref grid) — everything else in the block treats them as ordinary image tokens.</param>
    /// <param name="refPackedW">Packed-grid width of the reference-latent section (0 = no ref tokens).</param>
    /// <param name="tembZero">Qwen-Image-Edit-2511 (<c>index_timestep_zero</c> ref method): the timestep-0
    /// embedding. When set, the reference-latent rows of the IMAGE stream (rows ≥ <paramref name="timestepZeroIndex"/>)
    /// are modulated and gated with the t=0 modulation while the noise rows keep the real-timestep modulation —
    /// ComfyUI <c>qwen_image/model.py</c> <c>_modulate/_apply_gate</c> with <c>timestep_zero_index</c>. The text
    /// stream always uses the real-timestep modulation (upstream chunks temb back to row 0 for txt_mod).</param>
    /// <param name="timestepZeroIndex">First image-stream row of the reference tokens (= main noise token count).</param>
    /// <param name="cp">Context-parallel rank context: <paramref name="image"/> carries only this rank's rows of
    /// the img stream (the txt stream is replicated), the rope table is the rank's global-position slice, and the
    /// post-RoPE joint K/V is traded through <see cref="CpKvExchange"/> (txt prefix replicated, img rows exchanged)
    /// so the local queries attend over the FULL sequence. Null = the byte-identical single-backend path.</param>
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb,
        QwenImageRope rope, int imgPackedH, int imgPackedW, int txtPositionStart, (int H, int W)[]? refGrids = null,
        Tensor? tembZero = null, int timestepZeroIndex = 0, CpForwardContext? cp = null)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;
        if (cp is not null && (batch != 1 || refGrids is { Length: > 0 } || tembZero is not null || image.DType != DType.F32))
            throw new InvalidOperationException(
                "Context parallelism requires batch 1, F32 activations, and no edit-reference conditioning — the pipeline must gate these off.");

        // Modulation params: img_mod / txt_mod each → [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp].
        // .Forward runs Silu+Linear on the GPU; the per-row split it returns is tiny ([B, hidden] each) — negligible.
        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);
        Tensor[]? imgModZero = tembZero is not null ? _imgModulation.Forward(backend, tembZero) : null;

        // F16 activation path: every activation tensor follows the incoming stream dtype (the transformer
        // decides by casting the streams once before the block loop; F32 when the F16 mode is off).
        DType act = image.DType;
        ChromaF16.Probe("qwen.blk.imgIn", image);
        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);

        // ── 1. LayerNorm (no affine, eps 1e-6) + AdaLN modulate: x*(1+scale)+shift ──
        Tensor imgModulated = imgModZero is null
            ? NormModulate(backend, image, imgMod[0], imgMod[1], imgShape, _streamLnEps)
            : NormModulateSplit(backend, image, imgMod[0], imgMod[1], imgModZero[0], imgModZero[1], imgShape, timestepZeroIndex, _streamLnEps);
        Tensor txtModulated = NormModulate(backend, text, txtMod[0], txtMod[1], txtShape, _streamLnEps);

        // ── 2-9. Joint attention over [txt, img(, ref)]: per-head QK-norm, RoPE on the joint rows (a CP rank's table slice
        //         carries the global positions of its rows), the CP K/V rendezvous (queries stay local, keys cover the whole
        //         sequence), attention (QK RMS-norm bounds the scores), split, output projections, gated residuals ──
        AttentionCallContext joint;
        if (batch == 1)
        {
            (Tensor ropeCos, Tensor ropeSin) = cp is null
                ? rope.GetOrBuildJointTables(backend, imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, refGrids ?? [])
                : rope.GetOrBuildJointTables(backend, imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, [],
                    cp.Plan.Ranks[cp.Rank].TokenStart, imgSeqLen);
            QkRotation pairs = new(QkRotationKind.InterleavedPairs, ropeCos, ropeSin, totalSeqLen);
            joint = new AttentionCallContext
            {
                QueryRotation = pairs, KeyRotation = pairs,
                KvRewrite = cp is null ? null : new CpExchangeRewrite(cp.Exchange, cp.Rank, txtSeqLen),
            };
        }
        else
        {
            joint = new AttentionCallContext
            {
                HeadMajorRotation = new JointRowsRotation(rope, imgPackedH, imgPackedW, txtSeqLen, txtPositionStart, refGrids ?? []),
            };
        }
        Span<Tensor?> attended = [null, null];
        _attention!.Attend(backend, [imgModulated, txtModulated], attended, new JointCallContext { Joint = joint, ReleaseInputs = true });
        Tensor imgAttnProj = _attention.ProjectOutput(backend, 0, attended[0]!);
        Tensor imgAfterAttn = imgModZero is null ? GatedResidual(backend, image, imgAttnProj, imgMod[2], imgShape)
            : GatedResidualSplit(backend, image, imgAttnProj, imgMod[2], imgModZero[2], imgShape, timestepZeroIndex);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = _attention.ProjectOutput(backend, 1, attended[1]!);
        Tensor txtAfterAttn = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtAfterAttn, text, txtAttnProj, txtMod[2]);
        txtAttnProj.Dispose();

        // ── 10. Image MLP path ──
        Tensor imgMlpModulated = imgModZero is null
            ? NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape, _streamLnEps)
            : NormModulateSplit(backend, imgAfterAttn, imgMod[3], imgMod[4], imgModZero[3], imgModZero[4], imgShape, timestepZeroIndex, _streamLnEps);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = imgModZero is null ? GatedResidual(backend, imgAfterAttn, imgMlpOut, imgMod[5], imgShape)
            : GatedResidualSplit(backend, imgAfterAttn, imgMlpOut, imgMod[5], imgModZero[5], imgShape, timestepZeroIndex);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 11. Text MLP path ──
        Tensor txtMlpModulated = NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape, _streamLnEps);
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();
        if (imgModZero is not null)
            for (int i = 0; i < imgModZero.Length; i++) imgModZero[i].Dispose();

        return (imgFinal, txtFinal);
    }

    /// <summary>LayerNorm (no affine, eps 1e-6) followed by AdaLN modulation <c>out = x*(1+scale)+shift</c>, all on the
    /// GPU. <c>AffineBroadcastLastDim</c> computes <c>x*scale+shift</c>, so the scale tensor is pre-incremented by 1
    /// (<c>AddScalar</c>) to reproduce the <c>(1+scale)</c> factor — bit-identical to the old CPU
    /// <c>AdaLNModulation.ApplyModulation</c>. Mirrors ChromaDoubleStreamBlock.NormModulate.</summary>
    private static Tensor NormModulate(IBackend backend, Tensor x, Tensor shift, Tensor scale, TensorShape shape, float eps)
    {
        Tensor normed = new Tensor(shape, x.DType);
        backend.LayerNormNoAffine(normed, x, eps);
        Tensor scalePlus1 = new Tensor(scale.Shape, DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        Tensor output = new Tensor(shape, x.DType);
        backend.AffineBroadcastLastDim(output, normed, scalePlus1, shift);
        normed.Dispose();
        scalePlus1.Dispose();
        return output;
    }

    private static Tensor GatedResidual(IBackend backend, Tensor residual, Tensor value, Tensor gate, TensorShape shape)
    {
        Tensor output = new Tensor(shape, residual.DType);
        backend.GatedResidualLastDim(output, residual, value, gate);
        return output;
    }

    /// <summary>Row-split <see cref="NormModulate"/> for the 2511 timestep-zero ref method: rows
    /// <c>[0, mainRows)</c> get (shiftA, scaleA), rows <c>[mainRows, S)</c> get (shiftB, scaleB) — the
    /// LayerNorm is per-row so it runs once on the whole tensor, then each contiguous segment is affined
    /// with its own modulation and the segments are row-concatenated back. B=1 only (ref tokens require it).</summary>
    private static Tensor NormModulateSplit(IBackend backend, Tensor x, Tensor shiftA, Tensor scaleA,
        Tensor shiftB, Tensor scaleB, TensorShape shape, int mainRows, float eps)
    {
        int total = (int)shape[1];
        int dim = (int)shape[2];
        DType actT = x.DType;
        Tensor normed = new Tensor(shape, actT);
        backend.LayerNormNoAffine(normed, x, eps);

        Tensor main = new Tensor(new TensorShape(1, mainRows, dim), actT);
        backend.SliceRows(main, normed, 0);
        Tensor refPart = new Tensor(new TensorShape(1, total - mainRows, dim), actT);
        backend.SliceRows(refPart, normed, mainRows);
        normed.Dispose();

        Tensor scaleAPlus1 = new Tensor(scaleA.Shape, DType.F32);
        backend.AddScalar(scaleAPlus1, scaleA, 1.0f);
        Tensor mainOut = new Tensor(main.Shape, actT);
        backend.AffineBroadcastLastDim(mainOut, main, scaleAPlus1, shiftA);
        main.Dispose();
        scaleAPlus1.Dispose();

        Tensor scaleBPlus1 = new Tensor(scaleB.Shape, DType.F32);
        backend.AddScalar(scaleBPlus1, scaleB, 1.0f);
        Tensor refOut = new Tensor(refPart.Shape, actT);
        backend.AffineBroadcastLastDim(refOut, refPart, scaleBPlus1, shiftB);
        refPart.Dispose();
        scaleBPlus1.Dispose();

        Tensor output = new Tensor(shape, actT);
        backend.Concat(output, new Tensor[] { mainOut, refOut }, 1);
        mainOut.Dispose();
        refOut.Dispose();
        return output;
    }

    /// <summary>Row-split gated residual (2511 timestep-zero): <c>out = residual + [gateA·value_main ; gateB·value_ref]</c>.</summary>
    private static Tensor GatedResidualSplit(IBackend backend, Tensor residual, Tensor value,
        Tensor gateA, Tensor gateB, TensorShape shape, int mainRows)
    {
        int total = (int)shape[1];
        int dim = (int)shape[2];
        TensorShape mainShape = new TensorShape(1, mainRows, dim);
        TensorShape refShape = new TensorShape(1, total - mainRows, dim);
        DType actT = residual.DType;

        Tensor resMain = new Tensor(mainShape, actT);
        backend.SliceRows(resMain, residual, 0);
        Tensor valMain = new Tensor(mainShape, actT);
        backend.SliceRows(valMain, value, 0);
        Tensor outMain = new Tensor(mainShape, actT);
        backend.GatedResidualLastDim(outMain, resMain, valMain, gateA);
        resMain.Dispose();
        valMain.Dispose();

        Tensor resRef = new Tensor(refShape, actT);
        backend.SliceRows(resRef, residual, mainRows);
        Tensor valRef = new Tensor(refShape, actT);
        backend.SliceRows(valRef, value, mainRows);
        Tensor outRef = new Tensor(refShape, actT);
        backend.GatedResidualLastDim(outRef, resRef, valRef, gateB);
        resRef.Dispose();
        valRef.Dispose();

        Tensor output = new Tensor(shape, actT);
        backend.Concat(output, new Tensor[] { outMain, outRef }, 1);
        outMain.Dispose();
        outRef.Dispose();
        return output;
    }
}
