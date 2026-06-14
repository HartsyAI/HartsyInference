using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Qwen-Image dual-stream MMDiT block (<c>QwenImageTransformerBlock</c>). Maintains separate image and text streams with independent AdaLN-Zero modulation (12 params each: <c>shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp</c> for both <c>mod1</c> and <c>mod2</c>), QK-norm, and GELU(approximate) FFNs. Joint attention concatenates <c>[txt, img]</c> along the sequence dim and applies precomputed RoPE separately to image and text (image-only RoPE for image tokens; zero-position RoPE rows for text tokens). Mirrors <c>diffusers/models/transformers/transformer_qwenimage.py:QwenImageTransformerBlock.forward</c> 1:1.</summary>
public sealed unsafe class QwenImageBlock
{
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
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
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

        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.img_mlp.net.0.proj.weight"],
            weights[$"{prefix}.img_mlp.net.0.proj.bias"],
            weights[$"{prefix}.img_mlp.net.2.weight"],
            weights[$"{prefix}.img_mlp.net.2.bias"]);

        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.txt_mlp.net.0.proj.weight"],
            weights[$"{prefix}.txt_mlp.net.0.proj.bias"],
            weights[$"{prefix}.txt_mlp.net.2.weight"],
            weights[$"{prefix}.txt_mlp.net.2.bias"]);
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

    /// <summary>Forward pass. Each modulation produces <c>[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]</c>. Joint attention concats <c>[txt, img]</c>; QK rotated by <see cref="QwenImageRope"/> separately for image (per-token <c>[0, row, col]</c>) and text (offset by <paramref name="txtPositionStart"/>) before concat. Returns <c>(image, text)</c>.</summary>
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb,
        QwenImageRope rope, int imgPackedH, int imgPackedW, int txtPositionStart)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;

        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        Tensor imgNormed = new Tensor(imgShape, DType.F32);
        DiTUtils.LayerNormNoAffine(imgNormed, image, batch, imgSeqLen, _hiddenSize);
        Tensor imgModulated = AdaLNModulation.ApplyModulation(imgNormed, imgMod[0], imgMod[1], batch, imgSeqLen, _hiddenSize);
        imgNormed.Dispose();

        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        Tensor txtNormed = new Tensor(txtShape, DType.F32);
        DiTUtils.LayerNormNoAffine(txtNormed, text, batch, txtSeqLen, _hiddenSize);
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

        Tensor jointQ = new Tensor(jointMhShape, DType.F32);
        Tensor jointK = new Tensor(jointMhShape, DType.F32);
        Tensor jointV = new Tensor(jointMhShape, DType.F32);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointQ, txtQMh, imgQMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointK, txtKMh, imgKMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        DiTUtils.ConcatAlongSeqDimMultiHead(jointV, txtVMh, imgVMh, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
        txtQMh.Dispose(); imgQMh.Dispose();
        txtKMh.Dispose(); imgKMh.Dispose();
        txtVMh.Dispose(); imgVMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor jointAttnOut = new Tensor(jointMhShape, DType.F32);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, null, scale);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();

        Tensor txtAttnMh = new Tensor(txtMhShape, DType.F32);
        Tensor imgAttnMh = new Tensor(imgMhShape, DType.F32);
        DiTUtils.SplitAlongSeqDimMultiHead(txtAttnMh, imgAttnMh, jointAttnOut, batch, _numHeads, txtSeqLen, imgSeqLen, _headDim);
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
        DiTUtils.LayerNormNoAffine(imgMlpNormed, imgAfterAttn, batch, imgSeqLen, _hiddenSize);
        Tensor imgMlpModulated = AdaLNModulation.ApplyModulation(imgMlpNormed, imgMod[3], imgMod[4], batch, imgSeqLen, _hiddenSize);
        imgMlpNormed.Dispose();
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = AdaLNModulation.ApplyGatedResidual(imgAfterAttn, imgMlpOut, imgMod[5], batch, imgSeqLen, _hiddenSize);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        Tensor txtMlpNormed = new Tensor(txtShape, DType.F32);
        DiTUtils.LayerNormNoAffine(txtMlpNormed, txtAfterAttn, batch, txtSeqLen, _hiddenSize);
        Tensor txtMlpModulated = AdaLNModulation.ApplyModulation(txtMlpNormed, txtMod[3], txtMod[4], batch, txtSeqLen, _hiddenSize);
        txtMlpNormed.Dispose();
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = AdaLNModulation.ApplyGatedResidual(txtAfterAttn, txtMlpOut, txtMod[5], batch, txtSeqLen, _hiddenSize);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }
}
