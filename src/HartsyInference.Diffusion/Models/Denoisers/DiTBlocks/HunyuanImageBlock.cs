using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Hunyuan Image dual-stream MMDiT block (<c>HunyuanImageTransformerBlock</c>). Maintains separate image and text streams with independent AdaLN-Zero modulation (6 params each: <c>shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp</c>) — chunked from a single Linear that emits <c>6 * hidden</c>. Joint attention applies image-only RoPE before concatenating <c>[img, txt]</c> along the sequence dim, then SDPA, then splits back to per-stream output projections. FFN is GELU-approximate (tanh) with <c>ff.net.0.proj</c> + <c>ff.net.2</c> diffusers naming. Mirrors <c>diffusers/models/transformers/transformer_hunyuanimage.py:HunyuanImageTransformerBlock.forward</c> 1:1.</summary>
public sealed unsafe class HunyuanImageBlock : IStreamingBlock
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

    private JointAttention? _attention;

    /// <summary>Creates a Hunyuan Image dual-stream block.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for V2.1).</param>
    /// <param name="numHeads">Number of attention heads (24 for V2.1).</param>
    /// <param name="headDim">Per-head dimension (128 for V2.1; <c>numHeads * headDim == hiddenSize</c>).</param>
    /// <param name="mlpDim">MLP inner dimension (4 * hiddenSize = 12288 for V2.1).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon. Default 1e-6.</param>
    public HunyuanImageBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-6f)
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

    /// <summary>Loads weights using diffusers naming under <c>transformer_blocks.{i}.*</c>. Modulation linears project temb to <c>6 * hidden</c>; attention is bias=True; QK-norm is RMSNorm; FFNs are GELU(approximate) with <c>ff.net.0.proj</c> + <c>ff.net.2</c> naming.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _imgModulation.LoadWeights(
            weights[$"{prefix}.norm1.linear.weight"],
            weights[$"{prefix}.norm1.linear.bias"]);
        _txtModulation.LoadWeights(
            weights[$"{prefix}.norm1_context.linear.weight"],
            weights[$"{prefix}.norm1_context.linear.bias"]);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];
        weights.TryGetValue($"{prefix}.attn.to_q.bias", out _toQBias);
        weights.TryGetValue($"{prefix}.attn.to_k.bias", out _toKBias);
        weights.TryGetValue($"{prefix}.attn.to_v.bias", out _toVBias);

        _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
        _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
        _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];
        weights.TryGetValue($"{prefix}.attn.add_q_proj.bias", out _addQBias);
        weights.TryGetValue($"{prefix}.attn.add_k_proj.bias", out _addKBias);
        weights.TryGetValue($"{prefix}.attn.add_v_proj.bias", out _addVBias);

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
        _normAddedQ.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
        _normAddedK.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);

        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.ff.net.0.proj.weight"],
            weights[$"{prefix}.ff.net.0.proj.bias"],
            weights[$"{prefix}.ff.net.2.weight"],
            weights[$"{prefix}.ff.net.2.bias"]);

        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.ff_context.net.0.proj.weight"],
            weights[$"{prefix}.ff_context.net.0.proj.bias"],
            weights[$"{prefix}.ff_context.net.2.weight"],
            weights[$"{prefix}.ff_context.net.2.bias"]);

        // QK RMS-norm bounds the scores, so the joint call may narrow to F16.
        _attention = new JointAttention(
            new JointAttentionConfig
            {
                Heads = _numHeads, HeadDim = _headDim, ConcatOrder = [0, 1], AllowF16 = true,
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

    /// <summary>Sum of weight bytes in this streamable block (for the block-streaming budget heuristic). Routed through <see cref="DType.ComputeByteCount"/> rather than <c>ElementCount * SizeInBytes</c>: K-quant dtypes report <c>SizeInBytes = 0</c> and carry their real footprint in <c>BlockByteSize</c>/<c>BlockElementCount</c>, so the naive product silently reported 0 for the GGUF checkpoints this arch ships as.</summary>
    public long EstimatedWeightBytes
    {
        get { long t = 0; foreach (Tensor w in EnumerateWeights()) t += w.DType.ComputeByteCount(w.ElementCount); return t; }
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

    /// <summary>Forward pass. Each stream's modulation produces 6 params; image-only RoPE is applied via <see cref="HunyuanImageRope"/> before <c>[img, txt]</c> joint-attention concat. Returns <c>(image, text)</c>.</summary>
    // A batch above one rotates the image rows on the host after the head permute.
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb,
        HunyuanImageRope rope, int imgPackedH, int imgPackedW, int imgPackedT = 1)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];

        // Stream activation dtype follows the input (F16 on the numerics.ditF16 hot path; the transformer casts the
        // img/txt streams to F16 before the block loop). Every stream intermediate below is `act`; the modulation
        // params (imgMod/txtMod) stay F32 because temb is F32, and NormModulate/GatedResidualLastDim/SDPA all accept
        // an F16 stream with F32 params. QK-norm weights + RoPE tables are F32 too.
        DType act = image.DType;

        Tensor[] imgMod = _imgModulation.Forward(backend, temb);
        Tensor[] txtMod = _txtModulation.Forward(backend, temb);

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);

        // ── 1. LayerNorm (no affine) + AdaLN modulate: x*(1+scale)+shift ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

        // ── 2-9. Joint attention over [img, txt]: per-head QK-norm, image-only RoPE (before the concat for one batch item,
        //         on the image rows of the joint head-major Q/K for a batch), attention (QK RMS-norm bounds the scores),
        //         split, output projections, gated residuals ──
        Tensor[] attended = batch == 1
            ? JointAttend(backend, imgModulated, txtModulated, rope, imgPackedT, imgPackedH, imgPackedW)
            : JointAttendHeadMajor(backend, imgModulated, txtModulated, rope, imgSeqLen, imgPackedT, imgPackedH, imgPackedW);
        Tensor imgAttnProj = _attention!.ProjectOutput(backend, 0, attended[0]);
        Tensor imgAfterAttn = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgAfterAttn, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = _attention.ProjectOutput(backend, 1, attended[1]!);
        Tensor txtAfterAttn = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtAfterAttn, text, txtAttnProj, txtMod[2]);
        txtAttnProj.Dispose();

        // ── 10. Image MLP path ──
        Tensor imgMlpModulated = DiTUtils.NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        Tensor imgFinal = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 11. Text MLP path ──
        Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape);
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal);
    }

    /// <summary>Dual-stream forward with per-frame image modulation (MovieRoll's per-frame timestep). Image token i takes the adaLN row of its
    /// group: <paramref name="frameTemb"/> is one <c>[groups, hidden]</c> embedding per group, and <paramref name="layout"/> maps the image tokens
    /// onto those groups (a group may span several frames). The text stream keeps one shared modulation from <paramref name="txtTemb"/>.
    /// The six adaLN projections run as one batched Linear over all groups. Batch 1, F32 streams; inputs are not disposed.</summary>
    public (Tensor image, Tensor text) ForwardPerFrame(IBackend backend, Tensor image, Tensor text, Tensor txtTemb,
        Tensor frameTemb, TokenSegmentLayout layout, HunyuanImageRope rope, int imgPackedH, int imgPackedW, int imgPackedT)
    {
        ArgumentNullException.ThrowIfNull(frameTemb);
        ArgumentNullException.ThrowIfNull(layout);
        if (image.Shape[0] != 1 || text.Shape[0] != 1)
            throw new ArgumentException($"ForwardPerFrame takes batch 1; got image {image.Shape}, text {text.Shape}.", nameof(image));
        if (image.DType != DType.F32 || text.DType != DType.F32)
            throw new ArgumentException("ForwardPerFrame takes F32 streams.", nameof(image));
        if (frameTemb.Shape.Rank != 2 || frameTemb.Shape[1] != _hiddenSize)
            throw new ArgumentException($"frameTemb must be [groups, {_hiddenSize}], got {frameTemb.Shape}.", nameof(frameTemb));
        if (frameTemb.Shape[0] != layout.GroupCount)
            throw new ArgumentException($"{frameTemb.Shape[0]} frame embeddings for a layout with {layout.GroupCount} groups.", nameof(frameTemb));
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        if (layout.TotalLength != imgSeqLen)
            throw new ArgumentException($"The layout covers {layout.TotalLength} image tokens; the image has {imgSeqLen}.", nameof(layout));

        TensorShape imgShape = new(1, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new(1, txtSeqLen, _hiddenSize);
        Tensor[] txtMod = _txtModulation.Forward(backend, txtTemb);
        Tensor[]? imgMod = null;
        SegmentModulation? attnMod = null, ffnMod = null;
        try
        {
            // One batched projection for every group: [groups, 6·hidden] split into six [groups, hidden] tables.
            imgMod = _imgModulation.Forward(backend, frameTemb);
            attnMod = SegmentModulation.FromTables(imgMod[0], imgMod[1], imgMod[2], layout);
            ffnMod = SegmentModulation.FromTables(imgMod[3], imgMod[4], imgMod[5], layout);
            imgMod = null;   // the segment tables now own the six projections

            // Image attention: LayerNorm, then this token's group row (1 + scale) and shift.
            Tensor imgModulated = attnMod.NormAffine(backend, image, 1e-6f);
            Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);
            Tensor[] attended = JointAttend(backend, imgModulated, txtModulated, rope, imgPackedT, imgPackedH, imgPackedW);

            Tensor imgAttnProj = _attention!.ProjectOutput(backend, 0, attended[0]);
            Tensor imgAfterAttn = attnMod.GatedResidual(backend, image, imgAttnProj);
            imgAttnProj.Dispose();

            Tensor txtAttnProj = _attention.ProjectOutput(backend, 1, attended[1]);
            Tensor txtAfterAttn = new(txtShape, DType.F32);
            backend.GatedResidualLastDim(txtAfterAttn, text, txtAttnProj, txtMod[2]);
            txtAttnProj.Dispose();

            // Image FFN, per-group rows again.
            Tensor imgMlpModulated = ffnMod.NormAffine(backend, imgAfterAttn, 1e-6f);
            Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, 1, imgSeqLen);
            imgMlpModulated.Dispose();
            Tensor imgFinal = ffnMod.GatedResidual(backend, imgAfterAttn, imgMlpOut);
            imgMlpOut.Dispose();
            imgAfterAttn.Dispose();

            Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape);
            Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, 1, txtSeqLen);
            txtMlpModulated.Dispose();
            Tensor txtFinal = new(txtShape, DType.F32);
            backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
            txtMlpOut.Dispose();
            txtAfterAttn.Dispose();
            return (imgFinal, txtFinal);
        }
        finally
        {
            attnMod?.Dispose();
            ffnMod?.Dispose();
            if (imgMod is not null)
                foreach (Tensor m in imgMod) m.Dispose();
            foreach (Tensor m in txtMod) m.Dispose();
        }
    }

    /// <summary>Joint attention over batch 1 with image-only RoPE: the call <see cref="Forward"/> makes for batch 1. Both streams are released
    /// into the call; the returned attended streams are owned by the caller.</summary>
    private Tensor[] JointAttend(IBackend backend, Tensor imgModulated, Tensor txtModulated, HunyuanImageRope rope, int packedT, int packedH, int packedW)
    {
        HunyuanImageRotation imageRope = new(rope, packedT, packedH, packedW);
        JointCallContext context = new()
        {
            RotatedStream = 0, StreamRotation = new AttentionCallContext { Rotation = imageRope }, ReleaseInputs = true,
        };
        Span<Tensor?> attended = [null, null];
        _attention!.Attend(backend, [imgModulated, txtModulated], attended, in context);
        return [attended[0]!, attended[1]!];
    }

    /// <summary>Joint attention for batch greater than 1 with head-major RoPE on the leading image rows (the call <see cref="Forward"/> makes).
    /// Both streams are released into the call; the attended streams are owned by the caller.</summary>
    private Tensor[] JointAttendHeadMajor(IBackend backend, Tensor imgModulated, Tensor txtModulated, HunyuanImageRope rope,
        int imgSeqLen, int packedT, int packedH, int packedW)
    {
        HunyuanImageRotation imageRope = new(rope, packedT, packedH, packedW);
        JointCallContext context = new()
        {
            Joint = new AttentionCallContext { HeadMajorRotation = new LeadingRowsRotation(imgSeqLen, imageRope) }, ReleaseInputs = true,
        };
        Span<Tensor?> attended = [null, null];
        _attention!.Attend(backend, [imgModulated, txtModulated], attended, in context);
        return [attended[0]!, attended[1]!];
    }
}
