using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Microsoft Lens dual-stream MMDiT block (<c>LensTransformerBlock</c>). Mirrors upstream's per-stream modulation: <c>mod1, mod2 = Linear(SiLU(temb)).chunk(2)</c>, then <c>_modulate(x, mod) = (x*(1+scale)+shift, gate)</c> with <c>mod.chunk(3) = (shift, scale, gate)</c> — net effect is the standard <c>(shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)</c> 6-output order produced by <see cref="AdaLNModulation"/> with <c>numParams=6</c>. Joint attention concats <c>[img, txt]</c> per stream, applies complex-polar RoPE separately before concat, runs joint SDPA, splits back. <b>Returns <c>(text, image)</c></b> (encoder first, image second — matches upstream's <c>return encoder_hidden_states, hidden_states</c> in <c>LensTransformerBlock.forward</c>). Stream norms are RMSNorm with learned scale (vs QwenImageBlock's LayerNormNoAffine); FFN is SwiGLU with <c>w1/w2/w3</c> naming (vs QwenImageBlock's GELU); QKV is bias=True (upstream <c>img_qkv</c>/<c>txt_qkv</c> is split into <c>to_q/k/v</c> at checkpoint conversion).</summary>
public sealed unsafe class LensTransformerBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;
    private readonly float _streamNormEps;

    private readonly AdaLNModulation _imgModulation;
    private readonly AdaLNModulation _txtModulation;

    private Tensor? _imgNorm1Weight, _imgNorm2Weight;
    private Tensor? _txtNorm1Weight, _txtNorm2Weight;

    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;

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

    /// <summary>Creates a Lens dual-stream block. <paramref name="hiddenSize"/> = 1536, <paramref name="numHeads"/> = 24, <paramref name="headDim"/> = 64, <paramref name="mlpDim"/> = 4096 (SwiGLU 8/3 ratio) for the released weights.</summary>
    public LensTransformerBlock(int hiddenSize, int numHeads, int headDim, int mlpDim,
        float qkNormEps = 1e-5f, float streamNormEps = 1e-6f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = mlpDim;
        _streamNormEps = streamNormEps;

        _imgModulation = new AdaLNModulation(hiddenSize, 6);
        _txtModulation = new AdaLNModulation(hiddenSize, 6);

        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);
        _normAddedQ = new QkNorm(headDim, qkNormEps);
        _normAddedK = new QkNorm(headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, mlpDim);
    }

    /// <summary>Loads weights under <c>transformer_blocks.{i}.*</c>. Expects fused <c>img_qkv</c>/<c>txt_qkv</c> to have been pre-split into <c>to_q/to_k/to_v</c> and <c>add_q_proj/add_k_proj/add_v_proj</c> by <see cref="LensCheckpointConverter"/> (same pattern Sd3 uses). FFN is SwiGLU mode (no biases); stream norms are learned-scale RMSNorm.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(
            weights[$"{prefix}.img_mod.1.weight"],
            weights.TryGetValue($"{prefix}.img_mod.1.bias", out Tensor? imgModBias) ? imgModBias : null);
        _txtModulation.LoadWeights(
            weights[$"{prefix}.txt_mod.1.weight"],
            weights.TryGetValue($"{prefix}.txt_mod.1.bias", out Tensor? txtModBias) ? txtModBias : null);

        _imgNorm1Weight = CastToF32IfNeeded(weights[$"{prefix}.img_norm1.weight"]);
        _imgNorm2Weight = CastToF32IfNeeded(weights[$"{prefix}.img_norm2.weight"]);
        _txtNorm1Weight = CastToF32IfNeeded(weights[$"{prefix}.txt_norm1.weight"]);
        _txtNorm2Weight = CastToF32IfNeeded(weights[$"{prefix}.txt_norm2.weight"]);

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

        _imgFfn.LoadSwiGluWeights(
            weights[$"{prefix}.img_mlp.w1.weight"], null,
            weights[$"{prefix}.img_mlp.w3.weight"], null,
            weights[$"{prefix}.img_mlp.w2.weight"], null);

        _txtFfn.LoadSwiGluWeights(
            weights[$"{prefix}.txt_mlp.w1.weight"], null,
            weights[$"{prefix}.txt_mlp.w3.weight"], null,
            weights[$"{prefix}.txt_mlp.w2.weight"], null);
    }

    /// <summary>Yields every block weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _imgModulation.EnumerateWeights()) yield return w;
        foreach (Tensor w in _txtModulation.EnumerateWeights()) yield return w;
        if (_imgNorm1Weight is not null) yield return _imgNorm1Weight;
        if (_imgNorm2Weight is not null) yield return _imgNorm2Weight;
        if (_txtNorm1Weight is not null) yield return _txtNorm1Weight;
        if (_txtNorm2Weight is not null) yield return _txtNorm2Weight;
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

    /// <summary>Forward pass. Returns <c>(text, image)</c> matching upstream's return order (encoder first, image second). RoPE applied per-stream BEFORE the joint concat.</summary>
    public (Tensor text, Tensor image) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb,
        LensRope rope, int imgPackedH, int imgPackedW, int txtPositionStart)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;

        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        backend.RmsNorm(imgNormed, image, _imgNorm1Weight!, _streamNormEps);
        Tensor imgModulated = AdaLNModulation.ApplyModulation(imgNormed, imgMod[0], imgMod[1], batch, imgSeqLen, _hiddenSize);
        imgNormed.Dispose();

        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        Tensor txtNormed = new Tensor(txtShape, DType.F32);
        backend.RmsNorm(txtNormed, text, _txtNorm1Weight!, _streamNormEps);
        Tensor txtModulated = AdaLNModulation.ApplyModulation(txtNormed, txtMod[0], txtMod[1], batch, txtSeqLen, _hiddenSize);
        txtNormed.Dispose();

        Tensor imgQ = new Tensor(imgShape, DType.F32);
        backend.Linear(imgQ, imgModulated, _toQWeight!, _toQBias);
        Tensor imgK = new Tensor(imgShape, DType.F32);
        backend.Linear(imgK, imgModulated, _toKWeight!, _toKBias);
        Tensor imgV = new Tensor(imgShape, DType.F32);
        backend.Linear(imgV, imgModulated, _toVWeight!, _toVBias);
        imgModulated.Dispose();

        Tensor txtQ = new Tensor(txtShape, DType.F32);
        backend.Linear(txtQ, txtModulated, _addQWeight!, _addQBias);
        Tensor txtK = new Tensor(txtShape, DType.F32);
        backend.Linear(txtK, txtModulated, _addKWeight!, _addKBias);
        Tensor txtV = new Tensor(txtShape, DType.F32);
        backend.Linear(txtV, txtModulated, _addVWeight!, _addVBias);
        txtModulated.Dispose();

        int imgVectors = batch * imgSeqLen * _numHeads;
        int txtVectors = batch * txtSeqLen * _numHeads;

        Tensor imgQNormed = new Tensor(imgQ.Shape, DType.F32);
        Tensor imgKNormed = new Tensor(imgK.Shape, DType.F32);
        _normQ.Forward(imgQNormed, imgQ, imgVectors);
        _normK.Forward(imgKNormed, imgK, imgVectors);
        imgQ.Dispose();
        imgK.Dispose();

        Tensor txtQNormed = new Tensor(txtQ.Shape, DType.F32);
        Tensor txtKNormed = new Tensor(txtK.Shape, DType.F32);
        _normAddedQ.Forward(txtQNormed, txtQ, txtVectors);
        _normAddedK.Forward(txtKNormed, txtK, txtVectors);
        txtQ.Dispose();
        txtK.Dispose();

        TensorShape imgMhShape = new TensorShape(batch, _numHeads, imgSeqLen, _headDim);
        TensorShape txtMhShape = new TensorShape(batch, _numHeads, txtSeqLen, _headDim);
        TensorShape jointMhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        Tensor imgQMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgKMh = new Tensor(imgMhShape, DType.F32);
        Tensor imgVMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(imgQMh, imgQNormed, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgKMh, imgKNormed, batch, imgSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(imgVMh, imgV, batch, imgSeqLen, _numHeads, _headDim);
        imgQNormed.Dispose();
        imgKNormed.Dispose();
        imgV.Dispose();

        Tensor txtQMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtKMh = new Tensor(txtMhShape, DType.F32);
        Tensor txtVMh = new Tensor(txtMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(txtQMh, txtQNormed, batch, txtSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(txtKMh, txtKNormed, batch, txtSeqLen, _numHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(txtVMh, txtV, batch, txtSeqLen, _numHeads, _headDim);
        txtQNormed.Dispose();
        txtKNormed.Dispose();
        txtV.Dispose();

        rope.ApplyImage(imgQMh, imgKMh, batch, _numHeads, imgPackedH, imgPackedW);
        rope.ApplyText(txtQMh, txtKMh, batch, _numHeads, txtSeqLen, txtPositionStart);

        // Upstream concat order is [img, txt] (not [txt, img] like Qwen-Image): see LensJointAttention.forward
        //   q = cat([img_q, txt_q], dim=1).transpose(1, 2)
        // The split-after-attn must match — image slice first, text slice second.
        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointQ, imgQMh, txtQMh, batch, _numHeads, imgSeqLen, txtSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointK, imgKMh, txtKMh, batch, _numHeads, imgSeqLen, txtSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointV, imgVMh, txtVMh, batch, _numHeads, imgSeqLen, txtSeqLen, _headDim);
        txtQMh.Dispose(); imgQMh.Dispose();
        txtKMh.Dispose(); imgKMh.Dispose();
        txtVMh.Dispose(); imgVMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();

        Tensor imgAttnMh = new Tensor(imgMhShape, DType.F32);
        Tensor txtAttnMh = new Tensor(txtMhShape, DType.F32);
        DiTUtils.SplitAlongSeqDimMultiHead(imgAttnMh, txtAttnMh, jointAttnOut, batch, _numHeads, imgSeqLen, txtSeqLen, _headDim);
        jointAttnOut.Dispose();

        Tensor imgAttn = new Tensor(imgShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(imgAttn, imgAttnMh, batch, imgSeqLen, _numHeads, _headDim);
        imgAttnMh.Dispose();

        Tensor txtAttn = new Tensor(txtShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(txtAttn, txtAttnMh, batch, txtSeqLen, _numHeads, _headDim);
        txtAttnMh.Dispose();

        Tensor imgAttnProj = new Tensor(imgShape, DType.F32);
        backend.Linear(imgAttnProj, imgAttn, _toOutWeight!, _toOutBias);
        imgAttn.Dispose();
        Tensor imgAfterAttn = AdaLNModulation.ApplyGatedResidual(image, imgAttnProj, imgMod[2], batch, imgSeqLen, _hiddenSize);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = new Tensor(txtShape, DType.F32);
        backend.Linear(txtAttnProj, txtAttn, _toAddOutWeight!, _toAddOutBias);
        txtAttn.Dispose();
        Tensor txtAfterAttn = AdaLNModulation.ApplyGatedResidual(text, txtAttnProj, txtMod[2], batch, txtSeqLen, _hiddenSize);
        txtAttnProj.Dispose();

        Tensor imgMlpNormed = new Tensor(imgShape, DType.F32);
        backend.RmsNorm(imgMlpNormed, imgAfterAttn, _imgNorm2Weight!, _streamNormEps);
        Tensor imgMlpModulated = AdaLNModulation.ApplyModulation(imgMlpNormed, imgMod[3], imgMod[4], batch, imgSeqLen, _hiddenSize);
        imgMlpNormed.Dispose();
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = AdaLNModulation.ApplyGatedResidual(imgAfterAttn, imgMlpOut, imgMod[5], batch, imgSeqLen, _hiddenSize);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        Tensor txtMlpNormed = new Tensor(txtShape, DType.F32);
        backend.RmsNorm(txtMlpNormed, txtAfterAttn, _txtNorm2Weight!, _streamNormEps);
        Tensor txtMlpModulated = AdaLNModulation.ApplyModulation(txtMlpNormed, txtMod[3], txtMod[4], batch, txtSeqLen, _hiddenSize);
        txtMlpNormed.Dispose();
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = AdaLNModulation.ApplyGatedResidual(txtAfterAttn, txtMlpOut, txtMod[5], batch, txtSeqLen, _hiddenSize);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (txtFinal, imgFinal);
    }

    private static Tensor CastToF32IfNeeded(Tensor t) =>
        t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
