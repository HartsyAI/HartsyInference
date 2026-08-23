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
/// own forward, so all tokens are valid and the joint/self attention masks are all-true (passed as null to SDPA).</para>
///
/// <para>GPU-residency rewrite (mirrors the verified <see cref="QwenImageBlock"/> / <see cref="ChromaDoubleStreamBlock"/>):
/// every glue op — Lumina RMSNorm-zero, affine scale/shift, tanh-gated residual, per-head QK-norm, reshape-to-heads,
/// joint concat/split, GQA K/V repeat — runs as an <see cref="IBackend"/> GPU op so the activation stays device-resident
/// across the whole block (no per-op <c>DataPointer</c> reads / D2H sync barriers, which dominated the old
/// ~27 s/forward). The chunk-splits of the modulation projection are GPU last-dim slices (<c>SliceLastDim</c>); the
/// Lumina <c>(1+scale)</c> factor is reproduced by <c>AddScalar(scale, 1)</c> + <c>AffineBroadcastLastDim</c>, and the
/// tanh gate by <c>Tanh</c> + <c>GatedResidualLastDim</c> — bit-for-bit the old CPU math. The only op left on the CPU is
/// RoPE: <see cref="OmniGen2Rope"/> rotates interleaved pairs <c>(2i, 2i+1)</c> from precomputed per-token tables in
/// head-major <c>[B, H, S, D]</c> layout, which the CUDA rotate-half (NEOX) kernel does not match — so the two
/// per-stream rotations stay on <c>rope.Apply</c> (its D2H/H2D for Q,K is coherent with the activation cache).</para></summary>
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

    /// <summary>Creates a dual-stream block; requires <c>numQHeads * headDim == hidden</c> and <c>numQHeads % numKvHeads == 0</c>.</summary>
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

        _imgN1Lin = w[$"{p}.img_norm1.linear.weight"]; _imgN1Bias = w[$"{p}.img_norm1.linear.bias"]; _imgN1Norm = TensorCasts.EnsureF32(w[$"{p}.img_norm1.norm.weight"]);
        _imgN2Lin = w[$"{p}.img_norm2.linear.weight"]; _imgN2Bias = w[$"{p}.img_norm2.linear.bias"]; _imgN2Norm = TensorCasts.EnsureF32(w[$"{p}.img_norm2.norm.weight"]);
        _imgN3Lin = w[$"{p}.img_norm3.linear.weight"]; _imgN3Bias = w[$"{p}.img_norm3.linear.bias"]; _imgN3Norm = TensorCasts.EnsureF32(w[$"{p}.img_norm3.norm.weight"]);
        _insN1Lin = w[$"{p}.instruct_norm1.linear.weight"]; _insN1Bias = w[$"{p}.instruct_norm1.linear.bias"]; _insN1Norm = TensorCasts.EnsureF32(w[$"{p}.instruct_norm1.norm.weight"]);
        _insN2Lin = w[$"{p}.instruct_norm2.linear.weight"]; _insN2Bias = w[$"{p}.instruct_norm2.linear.bias"]; _insN2Norm = TensorCasts.EnsureF32(w[$"{p}.instruct_norm2.norm.weight"]);

        _imgAttnNorm = TensorCasts.EnsureF32(w[$"{p}.img_attn_norm.weight"]);
        _imgSelfAttnNorm = TensorCasts.EnsureF32(w[$"{p}.img_self_attn_norm.weight"]);
        _imgFfnNorm1 = TensorCasts.EnsureF32(w[$"{p}.img_ffn_norm1.weight"]);
        _imgFfnNorm2 = TensorCasts.EnsureF32(w[$"{p}.img_ffn_norm2.weight"]);
        _insAttnNorm = TensorCasts.EnsureF32(w[$"{p}.instruct_attn_norm.weight"]);
        _insFfnNorm1 = TensorCasts.EnsureF32(w[$"{p}.instruct_ffn_norm1.weight"]);
        _insFfnNorm2 = TensorCasts.EnsureF32(w[$"{p}.instruct_ffn_norm2.weight"]);
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
    /// <param name="jointCos">Joint-sequence <c>[instruct || image]</c> device cos table, <c>[L_ins+L_img, headDim]</c>
    /// duplicated-pair layout (<see cref="OmniGen2Rope.ExpandToDeviceTables"/>).</param>
    /// <param name="jointSin">Joint-sequence device sin table.</param>
    /// <param name="imageCos">Image-only device cos table for the self-attention, <c>[L_img, headDim]</c>.</param>
    /// <param name="imageSin">Image-only device sin table.</param>
    /// <param name="capLen">Instruction token count = <c>instruct.Shape[1]</c>; first segment of the joint sequence.</param>
    /// <param name="temb">Conditioning <c>[B, conditioningDim]</c>.</param>
    public (Tensor image, Tensor instruct) Forward(IBackend backend, Tensor image, Tensor instruct,
        Tensor jointCos, Tensor jointSin, Tensor imageCos, Tensor imageSin,
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
        (Tensor insAttn, Tensor imgAttn) = JointAttention(backend, imgN1, insN1, jointCos, jointSin, capLen, batch, imgLen, insLen);
        imgN1.Dispose(); insN1.Dispose();

        // ── image self-attention ──
        Tensor imgSelf = SelfAttention(backend, imgN3, imageCos, imageSin, batch, imgLen);
        imgN3.Dispose();

        // ── image residual updates ──
        Tensor imgAttnNormed = RmsNorm(backend, imgAttn, _imgAttnNorm!, batch, imgLen);
        imgAttn.Dispose();
        Tensor image1 = TanhGatedResidual(backend, image, imgAttnNormed, imgGateMsa, batch, imgLen);
        imgAttnNormed.Dispose();

        Tensor imgSelfNormed = RmsNorm(backend, imgSelf, _imgSelfAttnNorm!, batch, imgLen);
        imgSelf.Dispose();
        Tensor image2 = TanhGatedResidual(backend, image1, imgSelfNormed, imgGateSelf, batch, imgLen);
        imgSelfNormed.Dispose(); image1.Dispose();

        Tensor imgMlpIn = AffineScaleShift(backend, imgN2, imgScaleMlp, imgShiftMlp, batch, imgLen);
        imgN2.Dispose();
        Tensor imgFfNorm1 = RmsNorm(backend, imgMlpIn, _imgFfnNorm1!, batch, imgLen);
        imgMlpIn.Dispose();
        Tensor imgMlp = SwiGlu(backend, imgFfNorm1, _imgFf1!, _imgFf3!, _imgFf2!, batch, imgLen);
        imgFfNorm1.Dispose();
        Tensor imgMlpNormed = RmsNorm(backend, imgMlp, _imgFfnNorm2!, batch, imgLen);
        imgMlp.Dispose();
        Tensor imageOut = TanhGatedResidual(backend, image2, imgMlpNormed, imgGateMlp, batch, imgLen);
        imgMlpNormed.Dispose(); image2.Dispose();

        // ── instruction residual updates ──
        Tensor insAttnNormed = RmsNorm(backend, insAttn, _insAttnNorm!, batch, insLen);
        insAttn.Dispose();
        Tensor instruct1 = TanhGatedResidual(backend, instruct, insAttnNormed, insGateMsa, batch, insLen);
        insAttnNormed.Dispose();

        Tensor insMlpIn = AffineScaleShift(backend, insN2, insScaleMlp, insShiftMlp, batch, insLen);
        insN2.Dispose();
        Tensor insFfNorm1 = RmsNorm(backend, insMlpIn, _insFfnNorm1!, batch, insLen);
        insMlpIn.Dispose();
        Tensor insMlp = SwiGlu(backend, insFfNorm1, _insFf1!, _insFf3!, _insFf2!, batch, insLen);
        insFfNorm1.Dispose();
        Tensor insMlpNormed = RmsNorm(backend, insMlp, _insFfnNorm2!, batch, insLen);
        insMlp.Dispose();
        Tensor instructOut = TanhGatedResidual(backend, instruct1, insMlpNormed, insGateMlp, batch, insLen);
        insMlpNormed.Dispose(); instruct1.Dispose();

        imgGateMsa.Dispose(); imgScaleMlp.Dispose(); imgGateMlp.Dispose(); imgShiftMlp.Dispose(); imgGateSelf.Dispose();
        insGateMsa.Dispose(); insScaleMlp.Dispose(); insGateMlp.Dispose(); insShiftMlp.Dispose();

        return (imageOut, instructOut);
    }

    /// <summary>Joint cross-attention. Returns the attention output split into <c>(instruct[B,capLen,H], image[B,imgLen,H])</c>
    /// after all projections (per-stream out + parent to_out). GPU-resident: per-stream Q/K/V Linear into
    /// <c>[B, S, H, D]</c> heads-layout tensors, RMSNorm QK-norm over the head dim, joint <c>Concat</c> on the sequence
    /// axis, device RoPE pre-permute, <c>Permute0213</c> to <c>[B, H, S, D]</c>, GQA <c>RepeatKvHeads</c>, SDPA,
    /// permute back, and <c>SliceRows</c> splits.</summary>
    private (Tensor instruct, Tensor image) JointAttention(IBackend backend, Tensor imgN1, Tensor insN1,
        Tensor ropeCos, Tensor ropeSin, int capLen, int batch, int imgLen, int insLen)
    {
        int qDim = _numQHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int jointLen = insLen + imgLen;

        // Per-stream Q/K/V projected directly into [B, S, H, D] (byte-identical to [B, S, qDim]) so RmsNorm
        // QK-norm normalizes over the head dim and Permute0213 needs no reshape view.
        Tensor imgQ = LinearHeads(backend, imgN1, _jiImgToQ!, batch, imgLen, _numQHeads);
        Tensor imgK = LinearHeads(backend, imgN1, _jiImgToK!, batch, imgLen, _numKvHeads);
        Tensor imgV = LinearHeads(backend, imgN1, _jiImgToV!, batch, imgLen, _numKvHeads);
        Tensor insQ = LinearHeads(backend, insN1, _jiInsToQ!, batch, insLen, _numQHeads);
        Tensor insK = LinearHeads(backend, insN1, _jiInsToK!, batch, insLen, _numKvHeads);
        Tensor insV = LinearHeads(backend, insN1, _jiInsToV!, batch, insLen, _numKvHeads);

        // QK-norm (per-head RMSNorm over the last dim = headDim).
        Tensor imgQn = RmsNormHeads(backend, imgQ, _jiNormQ, batch, imgLen, _numQHeads);
        imgQ.Dispose();
        Tensor imgKn = RmsNormHeads(backend, imgK, _jiNormK, batch, imgLen, _numKvHeads);
        imgK.Dispose();
        Tensor insQn = RmsNormHeads(backend, insQ, _jiNormQ, batch, insLen, _numQHeads);
        insQ.Dispose();
        Tensor insKn = RmsNormHeads(backend, insK, _jiNormK, batch, insLen, _numKvHeads);
        insK.Dispose();

        // Concat [instruct, image] along the sequence dim (contiguous in [B, S, *] layout).
        Tensor qJoint = new Tensor(new TensorShape(batch, jointLen, qDim), DType.F32);
        backend.Concat(qJoint, new Tensor[] { insQn, imgQn }, 1);
        Tensor kJoint = new Tensor(new TensorShape(batch, jointLen, kvDim), DType.F32);
        backend.Concat(kJoint, new Tensor[] { insKn, imgKn }, 1);
        Tensor vJoint = new Tensor(new TensorShape(batch, jointLen, kvDim), DType.F32);
        backend.Concat(vJoint, new Tensor[] { insV, imgV }, 1);
        insQn.Dispose(); imgQn.Dispose(); insKn.Dispose(); imgKn.Dispose(); insV.Dispose(); imgV.Dispose();

        Tensor attnFlat = GqaAttention(backend, qJoint, kJoint, vJoint, ropeCos, ropeSin, batch, jointLen);
        qJoint.Dispose(); kJoint.Dispose(); vJoint.Dispose();

        // Split [instruct, image], per-stream output projections, parent to_out, then return split halves.
        Tensor insPart = new Tensor(new TensorShape(batch, capLen, _hidden), DType.F32);
        backend.SliceRows(insPart, attnFlat, 0);
        Tensor imgPart = new Tensor(new TensorShape(batch, imgLen, _hidden), DType.F32);
        backend.SliceRows(imgPart, attnFlat, capLen);
        attnFlat.Dispose();

        Tensor insProj = Linear(backend, insPart, _jiInsOut!, batch, insLen, _hidden);
        Tensor imgProj = Linear(backend, imgPart, _jiImgOut!, batch, imgLen, _hidden);
        insPart.Dispose(); imgPart.Dispose();

        Tensor merged = new Tensor(new TensorShape(batch, jointLen, _hidden), DType.F32);
        backend.Concat(merged, new Tensor[] { insProj, imgProj }, 1);
        insProj.Dispose(); imgProj.Dispose();
        Tensor outProj = Linear(backend, merged, _jiToOut!, batch, jointLen, _hidden);
        merged.Dispose();

        Tensor insOut = new Tensor(new TensorShape(batch, capLen, _hidden), DType.F32);
        backend.SliceRows(insOut, outProj, 0);
        Tensor imgOut = new Tensor(new TensorShape(batch, imgLen, _hidden), DType.F32);
        backend.SliceRows(imgOut, outProj, capLen);
        outProj.Dispose();
        return (insOut, imgOut);
    }

    /// <summary>Image self-attention (ordinary GQA block attention with image RoPE). GPU-resident, same primitives as
    /// <see cref="JointAttention"/> but over a single stream.</summary>
    private Tensor SelfAttention(IBackend backend, Tensor x,
        Tensor ropeCos, Tensor ropeSin, int batch, int seqLen)
    {
        Tensor q = LinearHeads(backend, x, _saToQ!, batch, seqLen, _numQHeads);
        Tensor k = LinearHeads(backend, x, _saToK!, batch, seqLen, _numKvHeads);
        Tensor v = LinearHeads(backend, x, _saToV!, batch, seqLen, _numKvHeads);

        Tensor qN = RmsNormHeads(backend, q, _saNormQ, batch, seqLen, _numQHeads);
        q.Dispose();
        Tensor kN = RmsNormHeads(backend, k, _saNormK, batch, seqLen, _numKvHeads);
        k.Dispose();

        Tensor attnFlat = GqaAttention(backend, qN, kN, v, ropeCos, ropeSin, batch, seqLen);
        qN.Dispose(); kN.Dispose(); v.Dispose();

        Tensor outProj = Linear(backend, attnFlat, _saToOut!, batch, seqLen, _hidden);
        attnFlat.Dispose();
        return outProj;
    }

    /// <summary>Device RoPE pre-permute, permute to <c>[B, H, S, D]</c>, GQA repeat-interleave K/V, SDPA,
    /// permute back to <c>[B, S, hidden]</c>. <paramref name="qFlat"/> is <c>[B, S, qDim]</c> and
    /// <paramref name="kFlat"/>/<paramref name="vFlat"/> are <c>[B, S, kvDim]</c>, already QK-normed; Q/K are
    /// rotated IN PLACE (callers dispose them right after). B=1 only (the transformer's contract).</summary>
    private Tensor GqaAttention(IBackend backend, Tensor qFlat, Tensor kFlat, Tensor vFlat,
        Tensor ropeCos, Tensor ropeSin, int batch, int seqLen)
    {
        if (batch != 1)
            throw new NotSupportedException("BooguImageDoubleBlock device-rope path requires batch == 1.");

        // Device RoPE on the pre-permute [1, S, H, D] layout (rotation is per (s, h) vector — permute-order
        // independent, bit-equivalent to the old post-permute host loop).
        backend.WanRopeInterleaved(qFlat, ropeCos, ropeSin, seqLen, _numQHeads, _headDim);
        backend.WanRopeInterleaved(kFlat, ropeCos, ropeSin, seqLen, _numKvHeads, _headDim);

        Tensor qMh = new Tensor(new TensorShape(batch, _numQHeads, seqLen, _headDim), DType.F32);
        backend.Permute0213(qMh, qFlat, seqLen, _numQHeads, _headDim);
        Tensor kMh = new Tensor(new TensorShape(batch, _numKvHeads, seqLen, _headDim), DType.F32);
        backend.Permute0213(kMh, kFlat, seqLen, _numKvHeads, _headDim);
        Tensor vMh = new Tensor(new TensorShape(batch, _numKvHeads, seqLen, _headDim), DType.F32);
        backend.Permute0213(vMh, vFlat, seqLen, _numKvHeads, _headDim);

        Tensor kRep = RepeatKv(backend, kMh, batch, seqLen);
        Tensor vRep = RepeatKv(backend, vMh, batch, seqLen);
        kMh.Dispose(); vMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(new TensorShape(batch, _numQHeads, seqLen, _headDim), DType.F32);
        // allowF16: QK-RMS-norm bounds the scores and the mask is null — the proven cuDNN fused
        // flash-attention config (falls back to the materialized F32 path when cuDNN is unavailable).
        backend.ScaledDotProductAttention(attnMh, qMh, kRep, vRep, null, scale, allowF16: true);
        qMh.Dispose(); kRep.Dispose(); vRep.Dispose();

        Tensor attnFlat = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Permute0213(attnFlat, attnMh, _numQHeads, seqLen, _headDim);
        attnMh.Dispose();
        return attnFlat;
    }

    /// <summary>GQA K/V head repeat to Q head count on the GPU (<c>[B, Hkv, S, D]</c> → <c>[B, Hq, S, D]</c>). When the
    /// group is 1 the input is already at full head count, so it is returned as a device-to-device copy via
    /// <c>RepeatKvHeads</c> (group 1).</summary>
    private Tensor RepeatKv(IBackend backend, Tensor kvMh, int batch, int seqLen)
    {
        Tensor output = new Tensor(new TensorShape(batch, _numKvHeads * _kvGroup, seqLen, _headDim), DType.F32);
        backend.RepeatKvHeads(output, kvMh, _numKvHeads, _kvGroup);
        return output;
    }

    /// <summary>Lumina <c>RMSNormZero</c>, GPU-resident. <c>SiLU(temb)</c> → modulation <c>Linear</c> → split the
    /// <c>[B, 4·hidden]</c> projection into four <c>[B, hidden]</c> chunks (<c>SliceLastDim</c>) → <c>RmsNorm(x)</c>
    /// scaled by <c>(1 + chunk0)</c> (<c>AddScalar</c> + <c>AffineBroadcastLastDim</c>). Returns the normed tensor and
    /// chunks 1/2/3 (the gates / scales the caller consumes). Bit-for-bit the old CPU split + scale.</summary>
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
        backend.SliceLastDim(c0, projected, 0);
        Tensor c1 = new Tensor(paramShape, DType.F32);
        backend.SliceLastDim(c1, projected, _hidden);
        Tensor c2 = new Tensor(paramShape, DType.F32);
        backend.SliceLastDim(c2, projected, 2 * _hidden);
        Tensor c3 = new Tensor(paramShape, DType.F32);
        backend.SliceLastDim(c3, projected, 3 * _hidden);
        projected.Dispose();

        TensorShape hShape = new TensorShape(batch, seqLen, _hidden);
        Tensor rms = new Tensor(hShape, DType.F32);
        backend.RmsNorm(rms, x, normW, _normEps);

        Tensor scalePlus1 = new Tensor(paramShape, DType.F32);
        backend.AddScalar(scalePlus1, c0, 1.0f);
        c0.Dispose();
        Tensor normed = new Tensor(hShape, DType.F32);
        backend.AffineBroadcastLastDim(normed, rms, scalePlus1, null);
        rms.Dispose(); scalePlus1.Dispose();
        return (normed, c1, c2, c3);
    }

    /// <summary>Affine modulation <c>out = (1 + scale)·input + shift</c>, GPU-resident
    /// (<c>AddScalar</c> + <c>AffineBroadcastLastDim</c>). <paramref name="scale"/>/<paramref name="shift"/> are
    /// <c>[B, hidden]</c> broadcast over the sequence axis.</summary>
    private Tensor AffineScaleShift(IBackend backend, Tensor input, Tensor scale, Tensor shift, int batch, int seqLen)
    {
        Tensor scalePlus1 = new Tensor(new TensorShape(batch, _hidden), DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.AffineBroadcastLastDim(output, input, scalePlus1, shift);
        scalePlus1.Dispose();
        return output;
    }

    /// <summary>Tanh-gated residual <c>out = residual + tanh(gate)·value</c>, GPU-resident (<c>Tanh</c> +
    /// <c>GatedResidualLastDim</c>). <paramref name="gate"/> is <c>[B, hidden]</c> broadcast over the sequence axis.</summary>
    private Tensor TanhGatedResidual(IBackend backend, Tensor residual, Tensor value, Tensor gate, int batch, int seqLen)
    {
        Tensor gateTanh = new Tensor(new TensorShape(batch, _hidden), DType.F32);
        backend.Tanh(gateTanh, gate);
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.GatedResidualLastDim(output, residual, value, gateTanh);
        gateTanh.Dispose();
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

    /// <summary>Linear projection whose output is declared <c>[B, S, numHeads, headDim]</c> (byte-identical to
    /// <c>[B, S, numHeads·headDim]</c>) so the following RmsNorm normalizes over the head dim.</summary>
    private Tensor LinearHeads(IBackend backend, Tensor input, Tensor weight, int batch, int seqLen, int numHeads)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, numHeads, _headDim), DType.F32);
        backend.Linear(output, input, weight, null);
        return output;
    }

    /// <summary>Per-head RMSNorm over the head dim using the QK-norm scale weight (<c>backend.RmsNorm</c> on the
    /// <c>[B, S, H, D]</c> heads-layout tensor). Numerically equal to the scalar <see cref="QkNorm.Forward"/>.</summary>
    private Tensor RmsNormHeads(IBackend backend, Tensor input, QkNorm qkNorm, int batch, int seqLen, int numHeads)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, numHeads, _headDim), DType.F32);
        backend.RmsNorm(output, input, qkNorm.Weight, qkNorm.Eps);
        return output;
    }

    private static Tensor Linear(IBackend backend, Tensor input, Tensor weight, int batch, int seqLen, int outDim)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, outDim), DType.F32);
        backend.Linear(output, input, weight, null);
        return output;
    }
}
