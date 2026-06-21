using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Boogu-Image dual-stream (double-stream) transformer block
/// (<c>BooguImageDoubleStreamTransformerBlock</c> from <c>transformer_boogu.py</c>). Processes an image stream and an
/// instruction stream in parallel. The image stream gets three sub-layers (joint cross-attention with the instruction
/// stream, image self-attention, MLP); the instruction stream gets two (joint cross-attention, MLP).
///
/// <para>Both streams are modulated by the shared timestep embedding via Lumina <c>RMSNormZero</c> heads. The image
/// stream has three modulation heads (<c>img_norm1/2/3</c>): <c>norm1</c> gives <c>(normed, gate_msa, scale_mlp,
/// gate_mlp)</c>, <c>norm2</c> gives <c>(normed, shift_mlp, …)</c>, <c>norm3</c> gives <c>(normed, gate_self, …)</c>.
/// The instruction stream has two (<c>instruct_norm1/2</c>) mirroring the image's norm1/norm2.</para>
///
/// <para>The joint cross-attention (<c>img_instruct_attn</c>) owns separate Q/K/V projections per stream
/// (<c>processor.{img,instruct}_to_{q,k,v}</c>): Q/K/V are computed per stream, concatenated <c>[instruct, image]</c>,
/// GQA + per-head RMSNorm (the parent attention's <c>norm_q/norm_k</c>) + joint RoPE applied, one SDPA over the joint
/// sequence, then split and projected by per-stream <c>{img,instruct}_out</c> and the parent <c>to_out.0</c>. Image
/// self-attention (<c>img_self_attn</c>) is an ordinary GQA block attention over the image tokens with image RoPE.</para>
///
/// <para>Assumes a single valid (unpadded) sequence per batch element — inference runs each guidance condition as its
/// own forward, so all tokens are valid and the joint/self attention masks are all-true (passed as null to SDPA).</para></summary>
public sealed unsafe class BooguImageDoubleBlock
{
    private readonly int _hidden;
    private readonly int _numQHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _kvGroup;
    private readonly int _ffnInner;
    private readonly int _conditioningDim;
    private readonly float _normEps;

    // Joint cross-attention (img_instruct_attn).
    private Tensor? _jiImgToQ, _jiImgToK, _jiImgToV;
    private Tensor? _jiInsToQ, _jiInsToK, _jiInsToV;
    private Tensor? _jiImgOut, _jiInsOut, _jiToOut;
    private readonly QkNorm _jiNormQ, _jiNormK;

    // Image self-attention (img_self_attn).
    private Tensor? _saToQ, _saToK, _saToV, _saToOut;
    private readonly QkNorm _saNormQ, _saNormK;

    // Feed-forwards.
    private Tensor? _imgFf1, _imgFf2, _imgFf3;
    private Tensor? _insFf1, _insFf2, _insFf3;

    // Modulation heads (linear + bias) and their RMSNorm weights.
    private Tensor? _imgN1Lin, _imgN1Bias, _imgN1Norm;
    private Tensor? _imgN2Lin, _imgN2Bias, _imgN2Norm;
    private Tensor? _imgN3Lin, _imgN3Bias, _imgN3Norm;
    private Tensor? _insN1Lin, _insN1Bias, _insN1Norm;
    private Tensor? _insN2Lin, _insN2Bias, _insN2Norm;

    // Sandwich norms.
    private Tensor? _imgAttnNorm, _imgSelfAttnNorm, _imgFfnNorm1, _imgFfnNorm2;
    private Tensor? _insAttnNorm, _insFfnNorm1, _insFfnNorm2;

    /// <summary>Creates a dual-stream block.</summary>
    public BooguImageDoubleBlock(int hidden, int numQHeads, int numKvHeads, int headDim, int ffnInner,
        int conditioningDim, float normEps = 1e-5f, float qkNormEps = 1e-5f)
    {
        if (numQHeads * headDim != hidden)
            throw new ArgumentException($"numQHeads * headDim ({numQHeads} * {headDim}) must equal hidden ({hidden}).");
        if (numQHeads % numKvHeads != 0)
            throw new ArgumentException($"numQHeads ({numQHeads}) must be divisible by numKvHeads ({numKvHeads}).");

        _hidden = hidden;
        _numQHeads = numQHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _kvGroup = numQHeads / numKvHeads;
        _ffnInner = ffnInner;
        _conditioningDim = conditioningDim;
        _normEps = normEps;

        _jiNormQ = new QkNorm(headDim, qkNormEps);
        _jiNormK = new QkNorm(headDim, qkNormEps);
        _saNormQ = new QkNorm(headDim, qkNormEps);
        _saNormK = new QkNorm(headDim, qkNormEps);
    }

    /// <summary>Loads weights for <c>double_stream_layers.{i}</c> using the upstream key names.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        string ji = $"{p}.img_instruct_attn";
        _jiImgToQ = w[$"{ji}.processor.img_to_q.weight"];
        _jiImgToK = w[$"{ji}.processor.img_to_k.weight"];
        _jiImgToV = w[$"{ji}.processor.img_to_v.weight"];
        _jiInsToQ = w[$"{ji}.processor.instruct_to_q.weight"];
        _jiInsToK = w[$"{ji}.processor.instruct_to_k.weight"];
        _jiInsToV = w[$"{ji}.processor.instruct_to_v.weight"];
        _jiImgOut = w[$"{ji}.processor.img_out.weight"];
        _jiInsOut = w[$"{ji}.processor.instruct_out.weight"];
        _jiToOut = w[$"{ji}.to_out.0.weight"];
        _jiNormQ.LoadWeights(w[$"{ji}.norm_q.weight"]);
        _jiNormK.LoadWeights(w[$"{ji}.norm_k.weight"]);

        string sa = $"{p}.img_self_attn";
        _saToQ = w[$"{sa}.to_q.weight"];
        _saToK = w[$"{sa}.to_k.weight"];
        _saToV = w[$"{sa}.to_v.weight"];
        _saToOut = w[$"{sa}.to_out.0.weight"];
        _saNormQ.LoadWeights(w[$"{sa}.norm_q.weight"]);
        _saNormK.LoadWeights(w[$"{sa}.norm_k.weight"]);

        _imgFf1 = w[$"{p}.img_feed_forward.linear_1.weight"];
        _imgFf2 = w[$"{p}.img_feed_forward.linear_2.weight"];
        _imgFf3 = w[$"{p}.img_feed_forward.linear_3.weight"];
        _insFf1 = w[$"{p}.instruct_feed_forward.linear_1.weight"];
        _insFf2 = w[$"{p}.instruct_feed_forward.linear_2.weight"];
        _insFf3 = w[$"{p}.instruct_feed_forward.linear_3.weight"];

        _imgN1Lin = w[$"{p}.img_norm1.linear.weight"]; _imgN1Bias = w[$"{p}.img_norm1.linear.bias"]; _imgN1Norm = F32(w[$"{p}.img_norm1.norm.weight"]);
        _imgN2Lin = w[$"{p}.img_norm2.linear.weight"]; _imgN2Bias = w[$"{p}.img_norm2.linear.bias"]; _imgN2Norm = F32(w[$"{p}.img_norm2.norm.weight"]);
        _imgN3Lin = w[$"{p}.img_norm3.linear.weight"]; _imgN3Bias = w[$"{p}.img_norm3.linear.bias"]; _imgN3Norm = F32(w[$"{p}.img_norm3.norm.weight"]);
        _insN1Lin = w[$"{p}.instruct_norm1.linear.weight"]; _insN1Bias = w[$"{p}.instruct_norm1.linear.bias"]; _insN1Norm = F32(w[$"{p}.instruct_norm1.norm.weight"]);
        _insN2Lin = w[$"{p}.instruct_norm2.linear.weight"]; _insN2Bias = w[$"{p}.instruct_norm2.linear.bias"]; _insN2Norm = F32(w[$"{p}.instruct_norm2.norm.weight"]);

        _imgAttnNorm = F32(w[$"{p}.img_attn_norm.weight"]);
        _imgSelfAttnNorm = F32(w[$"{p}.img_self_attn_norm.weight"]);
        _imgFfnNorm1 = F32(w[$"{p}.img_ffn_norm1.weight"]);
        _imgFfnNorm2 = F32(w[$"{p}.img_ffn_norm2.weight"]);
        _insAttnNorm = F32(w[$"{p}.instruct_attn_norm.weight"]);
        _insFfnNorm1 = F32(w[$"{p}.instruct_ffn_norm1.weight"]);
        _insFfnNorm2 = F32(w[$"{p}.instruct_ffn_norm2.weight"]);
    }

    /// <summary>Enumerates all weight tensors for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _jiImgToQ, _jiImgToK, _jiImgToV, _jiInsToQ, _jiInsToK, _jiInsToV, _jiImgOut, _jiInsOut, _jiToOut,
            _saToQ, _saToK, _saToV, _saToOut,
            _imgFf1, _imgFf2, _imgFf3, _insFf1, _insFf2, _insFf3,
            _imgN1Lin, _imgN1Bias, _imgN1Norm, _imgN2Lin, _imgN2Bias, _imgN2Norm, _imgN3Lin, _imgN3Bias, _imgN3Norm,
            _insN1Lin, _insN1Bias, _insN1Norm, _insN2Lin, _insN2Bias, _insN2Norm,
            _imgAttnNorm, _imgSelfAttnNorm, _imgFfnNorm1, _imgFfnNorm2, _insAttnNorm, _insFfnNorm1, _insFfnNorm2,
        ];
        foreach (Tensor? t in all)
            if (t is not null) yield return t;
        foreach (Tensor t in _jiNormQ.EnumerateWeights()) yield return t;
        foreach (Tensor t in _jiNormK.EnumerateWeights()) yield return t;
        foreach (Tensor t in _saNormQ.EnumerateWeights()) yield return t;
        foreach (Tensor t in _saNormK.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs one dual-stream block. Updates and returns <c>(image, instruct)</c>.</summary>
    /// <param name="image">Image stream <c>[B, L_img, hidden]</c> (ref + noise image tokens). Caller owns lifetime.</param>
    /// <param name="instruct">Instruction stream <c>[B, L_ins, hidden]</c>. Caller owns lifetime.</param>
    /// <param name="rope">Shared 3-axis RoPE (used only to apply the precomputed tables).</param>
    /// <param name="jointCos">Joint-sequence <c>[instruct || image]</c> cos table, sized <c>(L_ins+L_img)·headDim/2</c>.</param>
    /// <param name="jointSin">Joint-sequence sin table.</param>
    /// <param name="imageCos">Image-only cos table for the self-attention, sized <c>L_img·headDim/2</c>.</param>
    /// <param name="imageSin">Image-only sin table.</param>
    /// <param name="capLen">Instruction token count = <c>instruct.Shape[1]</c>; first segment of the joint sequence.</param>
    /// <param name="temb">Conditioning <c>[B, conditioningDim]</c>.</param>
    public (Tensor image, Tensor instruct) Forward(IBackend backend, Tensor image, Tensor instruct, OmniGen2Rope rope,
        ReadOnlySpan<float> jointCos, ReadOnlySpan<float> jointSin, ReadOnlySpan<float> imageCos, ReadOnlySpan<float> imageSin,
        int capLen, Tensor temb)
    {
        int batch = (int)image.Shape[0];
        int imgLen = (int)image.Shape[1];
        int insLen = (int)instruct.Shape[1];

        // ── modulation heads ──
        (Tensor imgN1, Tensor imgGateMsa, Tensor imgScaleMlp, Tensor imgGateMlp) = LuminaRmsNormZero(backend, image, temb, _imgN1Lin!, _imgN1Bias!, _imgN1Norm!, batch, imgLen);
        (Tensor imgN2, Tensor imgShiftMlp, Tensor imgN2c2, Tensor imgN2c3) = LuminaRmsNormZero(backend, image, temb, _imgN2Lin!, _imgN2Bias!, _imgN2Norm!, batch, imgLen);
        (Tensor imgN3, Tensor imgGateSelf, Tensor imgN3c2, Tensor imgN3c3) = LuminaRmsNormZero(backend, image, temb, _imgN3Lin!, _imgN3Bias!, _imgN3Norm!, batch, imgLen);
        imgN2c2.Dispose(); imgN2c3.Dispose(); imgN3c2.Dispose(); imgN3c3.Dispose();

        (Tensor insN1, Tensor insGateMsa, Tensor insScaleMlp, Tensor insGateMlp) = LuminaRmsNormZero(backend, instruct, temb, _insN1Lin!, _insN1Bias!, _insN1Norm!, batch, insLen);
        (Tensor insN2, Tensor insShiftMlp, Tensor insN2c2, Tensor insN2c3) = LuminaRmsNormZero(backend, instruct, temb, _insN2Lin!, _insN2Bias!, _insN2Norm!, batch, insLen);
        insN2c2.Dispose(); insN2c3.Dispose();

        // ── joint cross-attention over [instruct, image] ──
        (Tensor insAttn, Tensor imgAttn) = JointAttention(backend, imgN1, insN1, rope, jointCos, jointSin, capLen, batch, imgLen, insLen);
        imgN1.Dispose(); insN1.Dispose();

        // ── image self-attention ──
        Tensor imgSelf = SelfAttention(backend, imgN3, rope, imageCos, imageSin, batch, imgLen);
        imgN3.Dispose();

        // ── image residual updates ──
        Tensor imgAttnNormed = RmsNorm(backend, imgAttn, _imgAttnNorm!, batch, imgLen);
        imgAttn.Dispose();
        Tensor image1 = TanhGatedResidual(image, imgAttnNormed, imgGateMsa, batch, imgLen);
        imgAttnNormed.Dispose();

        Tensor imgSelfNormed = RmsNorm(backend, imgSelf, _imgSelfAttnNorm!, batch, imgLen);
        imgSelf.Dispose();
        Tensor image2 = TanhGatedResidual(image1, imgSelfNormed, imgGateSelf, batch, imgLen);
        imgSelfNormed.Dispose(); image1.Dispose();

        Tensor imgMlpIn = AffineScaleShift(imgN2, imgScaleMlp, imgShiftMlp, batch, imgLen);
        imgN2.Dispose();
        Tensor imgFfNorm1 = RmsNorm(backend, imgMlpIn, _imgFfnNorm1!, batch, imgLen);
        imgMlpIn.Dispose();
        Tensor imgMlp = SwiGlu(backend, imgFfNorm1, _imgFf1!, _imgFf3!, _imgFf2!, batch, imgLen);
        imgFfNorm1.Dispose();
        Tensor imgMlpNormed = RmsNorm(backend, imgMlp, _imgFfnNorm2!, batch, imgLen);
        imgMlp.Dispose();
        Tensor imageOut = TanhGatedResidual(image2, imgMlpNormed, imgGateMlp, batch, imgLen);
        imgMlpNormed.Dispose(); image2.Dispose();

        // ── instruction residual updates ──
        Tensor insAttnNormed = RmsNorm(backend, insAttn, _insAttnNorm!, batch, insLen);
        insAttn.Dispose();
        Tensor instruct1 = TanhGatedResidual(instruct, insAttnNormed, insGateMsa, batch, insLen);
        insAttnNormed.Dispose();

        Tensor insMlpIn = AffineScaleShift(insN2, insScaleMlp, insShiftMlp, batch, insLen);
        insN2.Dispose();
        Tensor insFfNorm1 = RmsNorm(backend, insMlpIn, _insFfnNorm1!, batch, insLen);
        insMlpIn.Dispose();
        Tensor insMlp = SwiGlu(backend, insFfNorm1, _insFf1!, _insFf3!, _insFf2!, batch, insLen);
        insFfNorm1.Dispose();
        Tensor insMlpNormed = RmsNorm(backend, insMlp, _insFfnNorm2!, batch, insLen);
        insMlp.Dispose();
        Tensor instructOut = TanhGatedResidual(instruct1, insMlpNormed, insGateMlp, batch, insLen);
        insMlpNormed.Dispose(); instruct1.Dispose();

        imgGateMsa.Dispose(); imgScaleMlp.Dispose(); imgGateMlp.Dispose(); imgShiftMlp.Dispose(); imgGateSelf.Dispose();
        insGateMsa.Dispose(); insScaleMlp.Dispose(); insGateMlp.Dispose(); insShiftMlp.Dispose();

        return (imageOut, instructOut);
    }

    /// <summary>Joint cross-attention. Returns the attention output split into <c>(instruct[B,capLen,H], image[B,imgLen,H])</c>
    /// after all projections (per-stream out + parent to_out).</summary>
    private (Tensor instruct, Tensor image) JointAttention(IBackend backend, Tensor imgN1, Tensor insN1, OmniGen2Rope rope,
        ReadOnlySpan<float> ropeCos, ReadOnlySpan<float> ropeSin, int capLen, int batch, int imgLen, int insLen)
    {
        int qDim = _numQHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int jointLen = insLen + imgLen;

        // Per-stream Q/K/V.
        Tensor imgQ = Linear(backend, imgN1, _jiImgToQ!, batch, imgLen, qDim);
        Tensor imgK = Linear(backend, imgN1, _jiImgToK!, batch, imgLen, kvDim);
        Tensor imgV = Linear(backend, imgN1, _jiImgToV!, batch, imgLen, kvDim);
        Tensor insQ = Linear(backend, insN1, _jiInsToQ!, batch, insLen, qDim);
        Tensor insK = Linear(backend, insN1, _jiInsToK!, batch, insLen, kvDim);
        Tensor insV = Linear(backend, insN1, _jiInsToV!, batch, insLen, kvDim);

        // Concat [instruct, image] along sequence.
        Tensor q = DiTUtils.ConcatAlongSeqDim(insQ, imgQ);
        Tensor k = DiTUtils.ConcatAlongSeqDim(insK, imgK);
        Tensor v = DiTUtils.ConcatAlongSeqDim(insV, imgV);
        imgQ.Dispose(); imgK.Dispose(); imgV.Dispose(); insQ.Dispose(); insK.Dispose(); insV.Dispose();

        // qk-norm.
        Tensor qN = new Tensor(new TensorShape(batch, jointLen, qDim), DType.F32);
        Tensor kN = new Tensor(new TensorShape(batch, jointLen, kvDim), DType.F32);
        _jiNormQ.Forward(qN, q, batch * jointLen * _numQHeads);
        _jiNormK.Forward(kN, k, batch * jointLen * _numKvHeads);
        q.Dispose(); k.Dispose();

        Tensor attnFlat = GqaAttention(backend, qN, kN, v, rope, ropeCos, ropeSin, batch, jointLen);
        qN.Dispose(); kN.Dispose(); v.Dispose();

        // Split, per-stream output projections, parent to_out, then return split halves.
        (Tensor insPart, Tensor imgPart) = DiTUtils.SplitAlongSeqDim(attnFlat, capLen);
        attnFlat.Dispose();
        Tensor insProj = Linear(backend, insPart, _jiInsOut!, batch, insLen, _hidden);
        Tensor imgProj = Linear(backend, imgPart, _jiImgOut!, batch, imgLen, _hidden);
        insPart.Dispose(); imgPart.Dispose();

        Tensor merged = DiTUtils.ConcatAlongSeqDim(insProj, imgProj);
        insProj.Dispose(); imgProj.Dispose();
        Tensor outProj = Linear(backend, merged, _jiToOut!, batch, jointLen, _hidden);
        merged.Dispose();

        (Tensor insOut, Tensor imgOut) = DiTUtils.SplitAlongSeqDim(outProj, capLen);
        outProj.Dispose();
        return (insOut, imgOut);
    }

    /// <summary>Image self-attention (ordinary GQA block attention with image RoPE).</summary>
    private Tensor SelfAttention(IBackend backend, Tensor x, OmniGen2Rope rope,
        ReadOnlySpan<float> ropeCos, ReadOnlySpan<float> ropeSin, int batch, int seqLen)
    {
        int qDim = _numQHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;

        Tensor q = Linear(backend, x, _saToQ!, batch, seqLen, qDim);
        Tensor k = Linear(backend, x, _saToK!, batch, seqLen, kvDim);
        Tensor v = Linear(backend, x, _saToV!, batch, seqLen, kvDim);

        Tensor qN = new Tensor(new TensorShape(batch, seqLen, qDim), DType.F32);
        Tensor kN = new Tensor(new TensorShape(batch, seqLen, kvDim), DType.F32);
        _saNormQ.Forward(qN, q, batch * seqLen * _numQHeads);
        _saNormK.Forward(kN, k, batch * seqLen * _numKvHeads);
        q.Dispose(); k.Dispose();

        Tensor attnFlat = GqaAttention(backend, qN, kN, v, rope, ropeCos, ropeSin, batch, seqLen);
        qN.Dispose(); kN.Dispose(); v.Dispose();

        Tensor outProj = Linear(backend, attnFlat, _saToOut!, batch, seqLen, _hidden);
        attnFlat.Dispose();
        return outProj;
    }

    /// <summary>Reshape to multi-head, apply the precomputed RoPE table, GQA repeat-interleave K/V, SDPA, reshape back
    /// to <c>[B, S, hidden]</c>. Inputs are flat <c>[B, S, qDim]</c> / <c>[B, S, kvDim]</c> already qk-normed.</summary>
    private Tensor GqaAttention(IBackend backend, Tensor qFlat, Tensor kFlat, Tensor vFlat, OmniGen2Rope rope,
        ReadOnlySpan<float> ropeCos, ReadOnlySpan<float> ropeSin, int batch, int seqLen)
    {
        Tensor qMh = DiTUtils.ReshapeToMultiHead(qFlat, batch, seqLen, _numQHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(kFlat, batch, seqLen, _numKvHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(vFlat, batch, seqLen, _numKvHeads, _headDim);

        rope.Apply(qMh, ropeCos, ropeSin, batch, _numQHeads, seqLen);
        rope.Apply(kMh, ropeCos, ropeSin, batch, _numKvHeads, seqLen);

        Tensor kRep = RepeatKvHeads(kMh, batch, _numKvHeads, _kvGroup, seqLen, _headDim);
        Tensor vRep = RepeatKvHeads(vMh, batch, _numKvHeads, _kvGroup, seqLen, _headDim);
        kMh.Dispose(); vMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(new TensorShape(batch, _numQHeads, seqLen, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kRep, vRep, null, scale);
        qMh.Dispose(); kRep.Dispose(); vRep.Dispose();

        Tensor attnFlat = DiTUtils.ReshapeFromMultiHead(attnMh, batch, seqLen, _numQHeads, _headDim);
        attnMh.Dispose();
        return attnFlat;
    }

    private (Tensor normed, Tensor c1, Tensor c2, Tensor c3) LuminaRmsNormZero(IBackend backend, Tensor x, Tensor temb,
        Tensor modLin, Tensor modBias, Tensor normW, int batch, int seqLen)
    {
        Tensor activated = new Tensor(new TensorShape(batch, _conditioningDim), DType.F32);
        backend.Silu(activated, temb);
        Tensor projected = new Tensor(new TensorShape(batch, 4 * _hidden), DType.F32);
        backend.Linear(projected, activated, modLin, modBias);
        activated.Dispose();

        TensorShape paramShape = new TensorShape(batch, _hidden);
        Tensor c0 = new Tensor(paramShape, DType.F32);
        Tensor c1 = new Tensor(paramShape, DType.F32);
        Tensor c2 = new Tensor(paramShape, DType.F32);
        Tensor c3 = new Tensor(paramShape, DType.F32);
        float* pp = (float*)projected.DataPointer;
        float* p0 = (float*)c0.DataPointer, p1 = (float*)c1.DataPointer, p2 = (float*)c2.DataPointer, p3 = (float*)c3.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int src = b * 4 * _hidden;
            int dst = b * _hidden;
            for (int d = 0; d < _hidden; d++)
            {
                p0[dst + d] = pp[src + d];
                p1[dst + d] = pp[src + _hidden + d];
                p2[dst + d] = pp[src + 2 * _hidden + d];
                p3[dst + d] = pp[src + 3 * _hidden + d];
            }
        }
        projected.Dispose();

        TensorShape hShape = new TensorShape(batch, seqLen, _hidden);
        Tensor rms = new Tensor(hShape, DType.F32);
        backend.RmsNorm(rms, x, normW, _normEps);
        Tensor normed = new Tensor(hShape, DType.F32);
        float* rp = (float*)rms.DataPointer, np = (float*)normed.DataPointer, sp = (float*)c0.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int cb = b * _hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int vb = (b * seqLen + s) * _hidden;
                for (int d = 0; d < _hidden; d++)
                    np[vb + d] = rp[vb + d] * (1.0f + sp[cb + d]);
            }
        }
        rms.Dispose(); c0.Dispose();
        return (normed, c1, c2, c3);
    }

    private Tensor AffineScaleShift(Tensor input, Tensor scale, Tensor shift, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        float* ip = (float*)input.DataPointer, scp = (float*)scale.DataPointer, shp = (float*)shift.DataPointer, op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int cb = b * _hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int vb = (b * seqLen + s) * _hidden;
                for (int d = 0; d < _hidden; d++)
                    op[vb + d] = (1.0f + scp[cb + d]) * ip[vb + d] + shp[cb + d];
            }
        }
        return output;
    }

    private Tensor TanhGatedResidual(Tensor residual, Tensor value, Tensor gate, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        float* rp = (float*)residual.DataPointer, vp = (float*)value.DataPointer, gp = (float*)gate.DataPointer, op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int gb = b * _hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int vb = (b * seqLen + s) * _hidden;
                for (int d = 0; d < _hidden; d++)
                    op[vb + d] = rp[vb + d] + MathF.Tanh(gp[gb + d]) * vp[vb + d];
            }
        }
        return output;
    }

    private Tensor SwiGlu(IBackend backend, Tensor input, Tensor ff1, Tensor ff3, Tensor ff2, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnInner);
        Tensor h1 = new Tensor(ffShape, DType.F32);
        Tensor h3 = new Tensor(ffShape, DType.F32);
        backend.Linear(h1, input, ff1, null);
        backend.Linear(h3, input, ff3, null);
        Tensor act = new Tensor(ffShape, DType.F32);
        backend.Silu(act, h1);
        h1.Dispose();
        Tensor gated = new Tensor(ffShape, DType.F32);
        backend.Mul(gated, act, h3);
        act.Dispose(); h3.Dispose();
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Linear(output, gated, ff2, null);
        gated.Dispose();
        return output;
    }

    private Tensor RmsNorm(IBackend backend, Tensor input, Tensor weight, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.RmsNorm(output, input, weight, _normEps);
        return output;
    }

    private static Tensor Linear(IBackend backend, Tensor input, Tensor weight, int batch, int seqLen, int outDim)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, outDim), DType.F32);
        backend.Linear(output, input, weight, null);
        return output;
    }

    private static Tensor RepeatKvHeads(Tensor input, int batch, int kvHeads, int groupSize, int seqLen, int headDim)
    {
        if (groupSize == 1)
        {
            Tensor copy = new Tensor(new TensorShape(batch, kvHeads, seqLen, headDim), DType.F32);
            long bytes = (long)batch * kvHeads * seqLen * headDim * sizeof(float);
            Buffer.MemoryCopy((void*)input.DataPointer, (void*)copy.DataPointer, bytes, bytes);
            return copy;
        }
        int qHeads = kvHeads * groupSize;
        Tensor output = new Tensor(new TensorShape(batch, qHeads, seqLen, headDim), DType.F32);
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long perHead = (long)seqLen * headDim * sizeof(float);
        for (int b = 0; b < batch; b++)
            for (int kv = 0; kv < kvHeads; kv++)
            {
                long src = ((long)b * kvHeads + kv) * seqLen * headDim;
                for (int g = 0; g < groupSize; g++)
                {
                    long dst = ((long)b * qHeads + kv * groupSize + g) * seqLen * headDim;
                    Buffer.MemoryCopy(inPtr + src, outPtr + dst, perHead, perHead);
                }
            }
        return output;
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
