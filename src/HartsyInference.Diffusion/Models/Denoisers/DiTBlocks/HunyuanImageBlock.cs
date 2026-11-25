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
    private AttentionLayer? _imgAttn, _txtAttn;

    // The token-major, QK-norm-free attention the window path runs over the cache-extended keys (the joint layer's config, without the
    // stream norms, which the window path applies to its own rows before the RoPE).
    private AttentionLayer? _windowAttend;

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
        _imgAttn = Stream(_toQWeight!, _toQBias, _toKWeight!, _toKBias, _toVWeight!, _toVBias, _toOutWeight!, _toOutBias, _normQ, _normK);
        _txtAttn = Stream(_addQWeight!, _addQBias, _addKWeight!, _addKBias, _addVWeight!, _addVBias, _toAddOutWeight!, _toAddOutBias,
            _normAddedQ, _normAddedK);
        _attention = new JointAttention(
            new JointAttentionConfig
            {
                Heads = _numHeads, HeadDim = _headDim, ConcatOrder = [0, 1], AllowF16 = true,
                TokenMajorRoute = AttentionRoute.TokenMajor, HeadMajorForSage = true,
            },
            [_imgAttn, _txtAttn]);
        // AllowF16 as the joint layer: the QK RMS norm bounds the scores, so the window attention may narrow to F16 too.
        _windowAttend = new AttentionLayer(new AttentionLayerConfig
        {
            Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, AllowF16 = true,
            TokenMajorRoute = AttentionRoute.TokenMajor, HeadMajorForSage = true,
        });
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

    /// <summary>One stream's adaLN in the mode the entry point runs: chunk tables (six <c>[1, hidden]</c> rows per batch item, as <see cref="Forward"/>
    /// and the text stream use) or per-frame segment tables (the image stream of <see cref="ForwardPerFrame"/> and <see cref="ForwardWindow"/>).
    /// The tables are not owned here; the caller releases them.</summary>
    private readonly struct StreamAdaLN
    {
        private readonly Tensor[]? _chunks;
        private readonly SegmentModulation? _attn;
        private readonly SegmentModulation? _ffn;

        private StreamAdaLN(Tensor[]? chunks, SegmentModulation? attn, SegmentModulation? ffn)
        {
            _chunks = chunks;
            _attn = attn;
            _ffn = ffn;
        }

        /// <summary>The six chunk tables: shift, scale, gate of attention, then shift, scale, gate of the FFN.</summary>
        public static StreamAdaLN Chunks(Tensor[] six) => new(six, null, null);

        /// <summary>The attention and FFN segment tables of a per-frame block.</summary>
        public static StreamAdaLN Segments(SegmentModulation attn, SegmentModulation ffn) => new(null, attn, ffn);

        /// <summary>The pre-sub-block norm: the FFN's when <paramref name="ffn"/> is set, else the attention's.</summary>
        public Tensor Norm(IBackend backend, Tensor x, bool ffn, TensorShape shape)
        {
            if (_chunks is null) return (ffn ? _ffn! : _attn!).NormAffine(backend, x, 1e-6f);
            int first = ffn ? 3 : 0;
            return DiTUtils.NormModulate(backend, x, _chunks[first], _chunks[first + 1], shape);
        }

        /// <summary>The gated residual of a sub-block: <c>residual + gate · value</c>, gate from the attention's or FFN's table.</summary>
        public Tensor Gate(IBackend backend, Tensor residual, Tensor value, bool ffn, TensorShape shape, DType act)
        {
            if (_chunks is null) return (ffn ? _ffn! : _attn!).GatedResidual(backend, residual, value);
            Tensor output = new(shape, act);
            backend.GatedResidualLastDim(output, residual, value, _chunks[ffn ? 5 : 2]);
            return output;
        }
    }

    /// <summary>The post-attention sub-blocks shared by every entry point: each stream's output projection and gated residual, then its FFN with
    /// the FFN's own norm and gate. The image stream takes <paramref name="imgAdaLN"/>; the text stream takes <paramref name="txtAdaLN"/> and is
    /// skipped when <paramref name="text"/> is null (an encode pass). The attended operands are consumed (their projections dispose them).</summary>
    private (Tensor Image, Tensor? Text) FinishBlock(IBackend backend, StreamAdaLN imgAdaLN, Tensor image, Tensor imgAttended,
        StreamAdaLN? txtAdaLN, Tensor? text, Tensor? txtAttended, TensorShape imgShape, TensorShape txtShape, DType act, int batch,
        int imgSeqLen, int txtSeqLen)
    {
        Tensor imgProj = _imgAttn!.ProjectOutput(backend, imgAttended);
        Tensor imgAfterAttn = imgAdaLN.Gate(backend, image, imgProj, ffn: false, imgShape, act);
        imgProj.Dispose();
        Tensor imgMlpIn = imgAdaLN.Norm(backend, imgAfterAttn, ffn: true, imgShape);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpIn, batch, imgSeqLen);
        imgMlpIn.Dispose();
        Tensor imgFinal = imgAdaLN.Gate(backend, imgAfterAttn, imgMlpOut, ffn: true, imgShape, act);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();
        if (text is null) return (imgFinal, null);

        Tensor txtProj = _txtAttn!.ProjectOutput(backend, txtAttended!);
        StreamAdaLN txt = txtAdaLN!.Value;
        Tensor txtAfterAttn = txt.Gate(backend, text, txtProj, ffn: false, txtShape, act);
        txtProj.Dispose();
        Tensor txtMlpIn = txt.Norm(backend, txtAfterAttn, ffn: true, txtShape);
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpIn, batch, txtSeqLen);
        txtMlpIn.Dispose();
        Tensor txtFinal = txt.Gate(backend, txtAfterAttn, txtMlpOut, ffn: true, txtShape, act);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();
        return (imgFinal, txtFinal);
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
        (Tensor imgFinal, Tensor? txtFinal) = FinishBlock(backend, StreamAdaLN.Chunks(imgMod), image, attended[0], StreamAdaLN.Chunks(txtMod),
            text, attended[1], imgShape, txtShape, act, batch, imgSeqLen, txtSeqLen);

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        return (imgFinal, txtFinal!);
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

            (Tensor imgFinal, Tensor? txtFinal) = FinishBlock(backend, StreamAdaLN.Segments(attnMod, ffnMod), image, attended[0],
                StreamAdaLN.Chunks(txtMod), text, attended[1], imgShape, txtShape, DType.F32, 1, imgSeqLen, txtSeqLen);
            return (imgFinal, txtFinal!);
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

    /// <summary>The MovieRoll inference-regime window of this block (<c>forward_inference</c>): per-frame image modulation as in
    /// <see cref="ForwardPerFrame"/>, batch 1, F32. The window's Q and K are projected and QK-normed, then the cached history is
    /// concatenated in front of the window's K and V, and RoPE runs over the concatenation: <paramref name="positions"/> holds one row
    /// of rope-axis coordinates per token of the cache then the window, <c>(Sc + Sw) × axes</c>, and the window takes the trailing rows.
    /// The cache is post-norm and pre-RoPE, so each cached row rotates at the position it occupies in this window, which a history
    /// re-packed after a trim need not share with the position it was captured at. <paramref name="returnKv"/> encodes the window
    /// for the cache: the text stream is skipped and the window's own pre-RoPE K and V are returned. Otherwise the window attends
    /// over its text rows too. Inputs are not disposed; the result is owned by the caller.</summary>
    public MovieRollWindowOutput ForwardWindow(IBackend backend, Tensor image, Tensor? text, Tensor? txtTemb, Tensor frameTemb,
        TokenSegmentLayout layout, HunyuanImageRope rope, float[] positions, Tensor? cacheKeys, Tensor? cacheValues, bool returnKv)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(frameTemb);
        if (!returnKv) ArgumentNullException.ThrowIfNull(txtTemb);
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(rope);
        ArgumentNullException.ThrowIfNull(positions);
        if (_imgAttn is null || _txtAttn is null || _windowAttend is null)
            throw new InvalidOperationException("LoadWeights before ForwardWindow.");
        if (image.Shape.Rank != 3 || image.Shape[0] != 1 || image.DType != DType.F32)
            throw new ArgumentException($"ForwardWindow takes a batch-1 F32 image; got {image.Shape} {image.DType}.", nameof(image));
        int sw = (int)image.Shape[1];
        if (layout.TotalLength != sw)
            throw new ArgumentException($"The layout covers {layout.TotalLength} image tokens; the image has {sw}.", nameof(layout));
        if (frameTemb.Shape.Rank != 2 || frameTemb.Shape[1] != _hiddenSize || frameTemb.Shape[0] != layout.GroupCount)
            throw new ArgumentException($"frameTemb must be [{layout.GroupCount}, {_hiddenSize}], got {frameTemb.Shape}.", nameof(frameTemb));
        if (!returnKv && text is null)
            throw new ArgumentException("A denoising window takes its text stream; pass returnKv to encode a window without text.", nameof(text));
        if (text is not null && (text.Shape.Rank != 3 || text.Shape[0] != 1 || text.Shape[2] != _hiddenSize || text.DType != DType.F32))
            throw new ArgumentException($"The text stream is [1, St, {_hiddenSize}] F32; got {text.Shape} {text.DType}.", nameof(text));
        if ((cacheKeys is null) != (cacheValues is null))
            throw new ArgumentException("The cache's keys and values come together.", nameof(cacheKeys));
        int sc = 0;
        if (cacheKeys is not null && cacheValues is not null)
        {
            if (cacheKeys.Shape.Rank != 4 || cacheKeys.Shape[0] != 1 || cacheKeys.Shape[2] != _numHeads || cacheKeys.Shape[3] != _headDim
                || cacheKeys.DType != DType.F32 || !cacheValues.Shape.Equals(cacheKeys.Shape) || cacheValues.DType != DType.F32)
                throw new ArgumentException($"The cache is [1, Sc, {_numHeads}, {_headDim}] F32 keys and values; got {cacheKeys.Shape} and {cacheValues.Shape}.",
                    nameof(cacheKeys));
            sc = (int)cacheKeys.Shape[1];
        }
        int axes = rope.AxesDim.Length;
        if (positions.Length != checked((sc + sw) * axes))
            throw new ArgumentException($"{sc + sw} cached and window tokens over {axes} axes take {(sc + sw) * axes} positions; got {positions.Length}.",
                nameof(positions));

        if (rope.HeadDim != _headDim)
            throw new ArgumentException($"The rope is {rope.HeadDim} wide per head; the block has {_headDim}.", nameof(rope));
        int st = returnKv ? 0 : (int)text!.Shape[1];
        int total = sw + st;
        TensorShape imgShape = new(1, sw, _hiddenSize);
        TensorShape txtShape = new(1, st, _hiddenSize);
        Tensor[]? imgMod = null, txtMod = null;
        SegmentModulation? attnMod = null, ffnMod = null;
        Tensor? imgQ = null, imgK = null, imgV = null, txtQ = null, txtK = null, txtV = null;
        Tensor? keysOut = null, valuesOut = null;
        Tensor? attendQ = null, attendK = null, attendV = null, attended = null, imgAttended = null, txtAttended = null;
        try
        {
            imgMod = _imgModulation.Forward(backend, frameTemb);
            attnMod = SegmentModulation.FromTables(imgMod[0], imgMod[1], imgMod[2], layout);
            ffnMod = SegmentModulation.FromTables(imgMod[3], imgMod[4], imgMod[5], layout);
            imgMod = null;   // the segment tables now own the six projections
            if (!returnKv) txtMod = _txtModulation.Forward(backend, txtTemb!);

            // Projection and QK norm of the window, before any rotation: the cache stores exactly these K and V.
            Tensor imgModulated = attnMod.NormAffine(backend, image, 1e-6f);
            (imgQ, imgK, imgV) = ProjectWindowStream(backend, _imgAttn, imgModulated);
            if (returnKv)
            {
                keysOut = new Tensor(imgK.Shape, DType.F32);
                valuesOut = new Tensor(imgV.Shape, DType.F32);
                backend.CopyTo(keysOut, imgK);
                backend.CopyTo(valuesOut, imgV);
            }
            if (!returnKv)
            {
                Tensor txtModulated = DiTUtils.NormModulate(backend, text!, txtMod![0], txtMod[1], txtShape);
                (txtQ, txtK, txtV) = ProjectWindowStream(backend, _txtAttn, txtModulated);
            }

            // The cache is concatenated before the RoPE; the window's rows take the trailing positions.
            if (cacheKeys is not null && cacheValues is not null)
            {
                // headOwned false: the cache belongs to the caller, so only the window's own K and V are consumed here.
                imgK = ConcatTokens(backend, cacheKeys, imgK!, sc + sw, headOwned: false);
                imgV = ConcatTokens(backend, cacheValues, imgV!, sc + sw, headOwned: false);
            }
            // Separate slots: the query and key tables keep their own signatures, so every block of this window reuses them.
            rope.RotateRows(backend, imgQ!, positions.AsSpan(sc * axes), sw, _numHeads, slot: 0);
            rope.RotateRows(backend, imgK!, positions, sc + sw, _numHeads, slot: 1);

            // Joint attention over [cache, window] keys and [window, text] queries; the text rows attend over everything. Each operand stays
            // tracked by the finally until the attention call takes it, so a throw before that disposes it exactly once.
            attendQ = imgQ; attendK = imgK; attendV = imgV;
            imgQ = imgK = imgV = null;
            if (!returnKv)
            {
                attendQ = ConcatTokens(backend, attendQ!, txtQ!, total);
                txtQ = null;
                attendK = ConcatTokens(backend, attendK!, txtK!, total + sc);
                txtK = null;
                attendV = ConcatTokens(backend, attendV!, txtV!, total + sc);
                txtV = null;
            }
            TensorShape flatShape = new(1, total, _hiddenSize);
            attended = _windowAttend.Attend(backend, attendQ!, attendK!, attendV!, flatShape, default(AttentionCallContext));
            attendQ = attendK = attendV = null;   // Attend took the operands

            imgAttended = new Tensor(new TensorShape(1, sw, _hiddenSize), DType.F32);
            JointAttention.SplitRows(backend, imgAttended, attended, 1, total, 0);
            if (!returnKv)
            {
                txtAttended = new Tensor(txtShape, DType.F32);
                JointAttention.SplitRows(backend, txtAttended, attended, 1, total, sw);
            }
            attended.Dispose();
            attended = null;

            // The post-attention sub-blocks are shared with the other entry points. An encode pass has no text stream and returns the window's K and V.
            (Tensor imgFinal, Tensor? txtFinal) = FinishBlock(backend, StreamAdaLN.Segments(attnMod!, ffnMod!), image, imgAttended!,
                txtMod is null ? (StreamAdaLN?)null : StreamAdaLN.Chunks(txtMod), text, txtAttended, imgShape, txtShape, DType.F32, 1, sw, st);
            if (returnKv)
            {
                MovieRollWindowOutput encoded = new(imgFinal, null, keysOut, valuesOut);
                keysOut = valuesOut = null;   // the result owns the cache tensors now
                return encoded;
            }
            return new MovieRollWindowOutput(imgFinal, txtFinal, null, null);
        }
        finally
        {
            attnMod?.Dispose();
            ffnMod?.Dispose();
            if (imgMod is not null)
                foreach (Tensor m in imgMod) m.Dispose();
            if (txtMod is not null)
                foreach (Tensor m in txtMod) m.Dispose();
            imgQ?.Dispose();
            imgK?.Dispose();
            imgV?.Dispose();
            txtQ?.Dispose();
            txtK?.Dispose();
            txtV?.Dispose();
            keysOut?.Dispose();
            valuesOut?.Dispose();
            attendQ?.Dispose();
            attendK?.Dispose();
            attendV?.Dispose();
            attended?.Dispose();
            imgAttended?.Dispose();
            txtAttended?.Dispose();
        }
    }

    /// <summary>The window's own post-norm, pre-RoPE K and V of this block, without the attention, the FFN or the output: what an encode pass
    /// stores in the cache when the block's output is discarded (the last block of a window). The same K and V as
    /// <see cref="ForwardWindow"/> with <c>returnKv</c>, which depend only on the window's input. Batch 1, F32; the input is not disposed.</summary>
    public (Tensor Keys, Tensor Values) EncodeWindowKeysAndValues(IBackend backend, Tensor image, Tensor frameTemb, TokenSegmentLayout layout)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(frameTemb);
        ArgumentNullException.ThrowIfNull(layout);
        if (_imgAttn is null) throw new InvalidOperationException("LoadWeights before EncodeWindowKeysAndValues.");
        if (image.Shape.Rank != 3 || image.Shape[0] != 1 || image.DType != DType.F32)
            throw new ArgumentException($"EncodeWindowKeysAndValues takes a batch-1 F32 image; got {image.Shape} {image.DType}.", nameof(image));
        if (layout.TotalLength != image.Shape[1])
            throw new ArgumentException($"The layout covers {layout.TotalLength} image tokens; the image has {image.Shape[1]}.", nameof(layout));
        if (frameTemb.Shape.Rank != 2 || frameTemb.Shape[1] != _hiddenSize || frameTemb.Shape[0] != layout.GroupCount)
            throw new ArgumentException($"frameTemb must be [{layout.GroupCount}, {_hiddenSize}], got {frameTemb.Shape}.", nameof(frameTemb));
        Tensor[]? imgMod = null;
        SegmentModulation? attnMod = null;
        try
        {
            imgMod = _imgModulation.Forward(backend, frameTemb);
            attnMod = SegmentModulation.FromTables(imgMod[0], imgMod[1], imgMod[2], layout);
            // The FFN's shift, scale and gate are not needed here; the segment tables above took the first three.
            imgMod[3].Dispose();
            imgMod[4].Dispose();
            imgMod[5].Dispose();
            imgMod = null;
            Tensor modulated = attnMod.NormAffine(backend, image, 1e-6f);
            (Tensor q, Tensor k, Tensor v) = ProjectWindowStream(backend, _imgAttn, modulated);
            q.Dispose();
            return (k, v);
        }
        finally
        {
            attnMod?.Dispose();
            if (imgMod is not null)
                foreach (Tensor m in imgMod) m.Dispose();
        }
    }

    /// <summary>Projects one stream's modulated rows to Q, K and V and QK-norms Q and K, before any rotation. Releases the input.</summary>
    private static (Tensor Q, Tensor K, Tensor V) ProjectWindowStream(IBackend backend, AttentionLayer stream, Tensor modulated)
    {
        (Tensor q, Tensor k, Tensor v) = stream.ProjectSelf(backend, modulated, releaseInput: true);
        try
        {
            stream.NormalizeQk(backend, ref q, ref k);
            return (q, k, v);
        }
        catch
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
            throw;
        }
    }

    /// <summary>Concatenates two token-major <c>[1, S, heads, D]</c> operands along the token axis into a new <c>[1, totalTokens, heads, D]</c>.
    /// <paramref name="tail"/> is always disposed (also when the concatenation throws); <paramref name="head"/> is disposed only when
    /// <paramref name="headOwned"/> is set, since the cache's tensors belong to the caller.</summary>
    private Tensor ConcatTokens(IBackend backend, Tensor head, Tensor tail, int totalTokens, bool headOwned = true)
    {
        try
        {
            Tensor joined = new(new TensorShape(1, totalTokens, _numHeads, _headDim), DType.F32);
            backend.Concat(joined, [head, tail], 1);
            return joined;
        }
        finally
        {
            if (headOwned) head.Dispose();
            tail.Dispose();
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
