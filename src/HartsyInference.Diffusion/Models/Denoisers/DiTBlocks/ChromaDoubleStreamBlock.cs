using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Chroma double-stream (joint-attention) block. Mirrors <see cref="FluxDoubleStreamBlock"/> but takes
/// the modulation parameters precomputed as a <c>[B, 12, hidden]</c> slice from the approximator's modulation table —
/// the per-block <c>norm1.linear</c> and <c>norm1_context.linear</c> projections are pruned (Chroma's defining
/// architectural change). Rows <c>0..6</c> are the image stream's <c>(shift_msa, scale_msa, gate_msa, shift_mlp,
/// scale_mlp, gate_mlp)</c>; rows <c>6..12</c> are the same for the text stream.
///
/// Reference: <c>diffusers/models/transformers/transformer_chroma.py:276-369</c>.
/// Also supports an optional attention mask <c>[B, totalSeqLen]</c> that gets expanded into the SDPA mask.</summary>
public sealed unsafe class ChromaDoubleStreamBlock : IStreamingBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

    // Image attention projections
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _toOutWeight, _toOutBias;

    // Text attention projections
    private Tensor? _addQWeight, _addQBias;
    private Tensor? _addKWeight, _addKBias;
    private Tensor? _addVWeight, _addVBias;
    private Tensor? _toAddOutWeight, _toAddOutBias;

    // Fused attention projections (numerics.chromaFusedQkv: the converter kept the BFL qkv whole).
    // When set, the split projections above stay null and the forward runs one qkv GEMM + QkvSplitNorm
    // per stream (the Hunyuan3DFluxBlocks recipe).
    private Tensor? _imgQkvWeight, _imgQkvBias;
    private Tensor? _txtQkvWeight, _txtQkvBias;

    // QK-norm (per-head RMSNorm with affine scale)
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;
    private readonly QkNorm _normAddedQ;
    private readonly QkNorm _normAddedK;

    // Image MLP (GELU(tanh) approximate per diffusers FeedForward(activation_fn="gelu-approximate"))
    private readonly SwiGluFfn _imgFfn;

    // Text MLP (GELU(tanh) approximate)
    private readonly SwiGluFfn _txtFfn;

    private JointAttention? _attention;

    /// <summary>Creates a ChromaDoubleStreamBlock.</summary>
    /// <param name="hiddenSize">Model hidden dimension (3072 for Chroma v1).</param>
    /// <param name="numHeads">Number of attention heads (24 for v1).</param>
    /// <param name="headDim">Per-head dimension (128 for v1; product = hiddenSize).</param>
    /// <param name="qkNormEps">QK-norm RMSNorm epsilon. Default 1e-6.</param>
    public ChromaDoubleStreamBlock(int hiddenSize, int numHeads, int headDim, float qkNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = hiddenSize * 4;

        _normQ = new QkNorm(_headDim, qkNormEps);
        _normK = new QkNorm(_headDim, qkNormEps);
        _normAddedQ = new QkNorm(_headDim, qkNormEps);
        _normAddedK = new QkNorm(_headDim, qkNormEps);

        _imgFfn = new SwiGluFfn(hiddenSize, _mlpDim);
        _txtFfn = new SwiGluFfn(hiddenSize, _mlpDim);
    }

    /// <summary>Loads weights using post-conversion diffusers-style naming under <c>transformer_blocks.{i}.*</c>.
    /// The <c>norm1</c> / <c>norm1_context</c> linears are intentionally absent (Chroma's pruned-AdaLN architecture).</summary>
    /// <param name="branchDamp">Residual-stream damp for the F16 activation path (see <see cref="ChromaF16"/>):
    /// applied to every branch-OUTPUT projection (to_out.0, to_add_out, both ff net.2) so the token stream rides
    /// at damp scale while all branch inputs stay baseline via the scale-invariant no-affine LayerNorm. 1.0 = off.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, float branchDamp = 1.0f)
    {
        // Image Q/K/V (bias=True per FluxAttention). Fused qkv when the converter kept it whole
        // (numerics.chromaFusedQkv), split projections otherwise.
        if (weights.TryGetValue($"{prefix}.attn.qkv.weight", out Tensor? imgQkvW))
        {
            _imgQkvWeight = imgQkvW;
            _imgQkvBias = weights[$"{prefix}.attn.qkv.bias"];
        }
        else
        {
            _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
            _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
            _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
            _toQBias = weights[$"{prefix}.attn.to_q.bias"];
            _toKBias = weights[$"{prefix}.attn.to_k.bias"];
            _toVBias = weights[$"{prefix}.attn.to_v.bias"];
        }
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];
        _toOutBias = weights[$"{prefix}.attn.to_out.0.bias"];

        // Text Q/K/V
        if (weights.TryGetValue($"{prefix}.attn.add_qkv.weight", out Tensor? txtQkvW))
        {
            _txtQkvWeight = txtQkvW;
            _txtQkvBias = weights[$"{prefix}.attn.add_qkv.bias"];
        }
        else
        {
            _addQWeight = weights[$"{prefix}.attn.add_q_proj.weight"];
            _addKWeight = weights[$"{prefix}.attn.add_k_proj.weight"];
            _addVWeight = weights[$"{prefix}.attn.add_v_proj.weight"];
            _addQBias = weights[$"{prefix}.attn.add_q_proj.bias"];
            _addKBias = weights[$"{prefix}.attn.add_k_proj.bias"];
            _addVBias = weights[$"{prefix}.attn.add_v_proj.bias"];
        }
        _toAddOutWeight = weights[$"{prefix}.attn.to_add_out.weight"];
        _toAddOutBias = weights[$"{prefix}.attn.to_add_out.bias"];

        Tensor imgFfnOutWeight = weights[$"{prefix}.ff.net.2.weight"];
        Tensor imgFfnOutBias = weights[$"{prefix}.ff.net.2.bias"];
        Tensor txtFfnOutWeight = weights[$"{prefix}.ff_context.net.2.weight"];
        Tensor txtFfnOutBias = weights[$"{prefix}.ff_context.net.2.bias"];
        if (branchDamp != 1.0f)
        {
            // Weight damp rides the GEMM alpha (any dtype, zero extra kernels); bias needs value-scaled copies
            // because the epilogue adds bias after alpha. Runs once per transformer instance at load.
            _toOutWeight.Fp8ScaleFactor *= branchDamp;
            _toOutBias = ChromaF16.DampBias(_toOutBias, branchDamp);
            _toAddOutWeight.Fp8ScaleFactor *= branchDamp;
            _toAddOutBias = ChromaF16.DampBias(_toAddOutBias, branchDamp);
            imgFfnOutWeight.Fp8ScaleFactor *= branchDamp;
            imgFfnOutBias = ChromaF16.DampBias(imgFfnOutBias, branchDamp);
            txtFfnOutWeight.Fp8ScaleFactor *= branchDamp;
            txtFfnOutBias = ChromaF16.DampBias(txtFfnOutBias, branchDamp);
        }

        // QK-norm
        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
        _normAddedQ.LoadWeights(weights[$"{prefix}.attn.norm_added_q.weight"]);
        _normAddedK.LoadWeights(weights[$"{prefix}.attn.norm_added_k.weight"]);

        // Image MLP (GELU mode: net.0.proj → projection, net.2 → output)
        _imgFfn.LoadGeluWeights(
            weights[$"{prefix}.ff.net.0.proj.weight"],
            weights[$"{prefix}.ff.net.0.proj.bias"],
            imgFfnOutWeight,
            imgFfnOutBias);

        // Text MLP
        _txtFfn.LoadGeluWeights(
            weights[$"{prefix}.ff_context.net.0.proj.weight"],
            weights[$"{prefix}.ff_context.net.0.proj.bias"],
            txtFfnOutWeight,
            txtFfnOutBias);

        // Built after the damp, which replaces the output biases. QK RMS-norm bounds the scores, so the call may narrow to F16.
        _attention = new JointAttention(
            new JointAttentionConfig { Heads = _numHeads, HeadDim = _headDim, ConcatOrder = [1, 0], AllowF16 = true },
            [Stream(_imgQkvWeight, _imgQkvBias, _toQWeight, _toQBias, _toKWeight, _toKBias, _toVWeight, _toVBias, _toOutWeight,
                 _toOutBias, _normQ, _normK),
             Stream(_txtQkvWeight, _txtQkvBias, _addQWeight, _addQBias, _addKWeight, _addKBias, _addVWeight, _addVBias,
                 _toAddOutWeight, _toAddOutBias, _normAddedQ, _normAddedK)]);
    }

    /// <summary>One stream: a fused qkv weight (split and normed in one op) when the checkpoint has one, else three Linears.</summary>
    private AttentionLayer Stream(Tensor? qkv, Tensor? qkvBias, Tensor? q, Tensor? qBias, Tensor? k, Tensor? kBias, Tensor? v,
        Tensor? vBias, Tensor output, Tensor? outputBias, QkNorm normQ, QkNorm normK) => new(
        new AttentionLayerConfig
        {
            Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead,
            Projection = qkv is null ? AttentionProjection.PerWeight : AttentionProjection.FusedSplitNorm,
        },
        new AttentionLayerWeights
        {
            Qkv = qkv, QkvBias = qkvBias, Query = q, QueryBias = qBias, Key = k, KeyBias = kBias, Value = v, ValueBias = vBias,
            Output = output, OutputBias = outputBias, QueryNorm = normQ.Weight, KeyNorm = normK.Weight,
            QueryNormEps = normQ.Eps, KeyNormEps = qkv is null ? normK.Eps : normQ.Eps,
        });

    /// <summary>The ChromaF16 trace points over the joint call: the joint normed Q and V and the head-major attention output.</summary>
    private sealed class TraceProbe : IAttentionProbe
    {
        public static readonly TraceProbe Instance = new();

        public void Observe(string point, Tensor value)
        {
            if (point == "q") ChromaF16.Probe("dbl-jointQ", value);
            else if (point == "v") ChromaF16.Probe("dbl-jointV", value);
            else if (point == "attention") ChromaF16.Probe("dbl-attnOut", value);
        }
    }

    /// <inheritdoc/>
    /// <remarks>Via <see cref="DType.ComputeByteCount"/>, not <c>ElementCount * SizeInBytes</c>: block-quantized
    /// dtypes report <c>SizeInBytes == 0</c>, so the naive product totals to zero and silently disables streaming.</remarks>
    public long EstimatedWeightBytes
    {
        get
        {
            long total = 0;
            foreach (Tensor w in EnumerateWeights()) total += w.DType.ComputeByteCount(w.ElementCount);
            return total;
        }
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
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
        if (_imgQkvWeight is not null) yield return _imgQkvWeight;
        if (_imgQkvBias is not null) yield return _imgQkvBias;
        if (_txtQkvWeight is not null) yield return _txtQkvWeight;
        if (_txtQkvBias is not null) yield return _txtQkvBias;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normAddedQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normAddedK.EnumerateWeights()) yield return w;
        foreach (Tensor w in _imgFfn.EnumerateWeights()) yield return w;
        foreach (Tensor w in _txtFfn.EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass. <paramref name="temb"/> is a <c>[B, 12, hidden]</c> slice of the global modulation
    /// table: rows 0..6 modulate the image stream, rows 6..12 modulate the text stream. Returns
    /// <c>(image, text)</c> in the diffusers' <c>(hidden_states, encoder_hidden_states)</c> argument order.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="image">Image tokens [B, N_img, hidden].</param>
    /// <param name="text">Text tokens [B, N_txt, hidden].</param>
    /// <param name="temb">Precomputed modulation rows [B, 12, hidden].</param>
    /// <param name="rope">Precomputed FluxRope.</param>
    /// <param name="sdpaMask">Optional additive [B, 1, S, S] SDPA mask, pre-built once per forward by ChromaTransformer (shared; not disposed here).</param>
    // ChromaPipeline runs CFG as two batch-1 passes; a batch above one rotates on the host after the head permute.
    public (Tensor image, Tensor text) Forward(
        IBackend backend, Tensor image, Tensor text, Tensor temb, FluxRope rope, Tensor? sdpaMask)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        // Activation dtype follows the INPUT (the Krea2/Z-Image pattern): ChromaTransformer casts the token
        // streams to F16 once before the block loop on the numerics.ditF16 path; the tiny modulation vectors
        // stay F32 (the F16 norm/affine/gate kernels take an F16 activation + F32 params), and the additive
        // SDPA mask stays F32 (cuDNN fused bias). The F32 path is byte-identical to the baseline.
        DType act = image.DType;

        // ── 1. Slice the precomputed modulation rows into tiny [B, hidden] tensors ──
        // imgMod[0..6]: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp (image stream)
        // txtMod[0..6]: same for text stream. Device slices (B=1): the old host path read the
        // device-produced temb's DataPointer — a full pipeline drain per block (the temb stream-stall).
        Tensor[] imgMod = batch == 1 ? SliceModRowsDevice(backend, temb, rowStart: 0, rowCount: 6)
            : SliceModRows(temb, batch, rowStart: 0, rowCount: 6);
        Tensor[] txtMod = batch == 1 ? SliceModRowsDevice(backend, temb, rowStart: 6, rowCount: 6)
            : SliceModRows(temb, batch, rowStart: 6, rowCount: 6);

        TensorShape imgShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtShape = new TensorShape(batch, txtSeqLen, _hiddenSize);

        // ── 2. LayerNorm (no affine, eps=1e-6) + modulate: x*(1+scale)+shift ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

        // ── 3-12. Joint attention over [txt, img] (each stream projects fused or split and norms per head; the shared
        //          [B,1,S,S] mask is borrowed, never disposed here), output projections, gated residuals ──
        AttentionCallContext joint = batch == 1
            ? new AttentionCallContext { Rotation = rope, Bias = sdpaMask, Probe = ChromaF16.Tracing ? TraceProbe.Instance : null }
            : new AttentionCallContext { HeadMajorRotation = rope, Bias = sdpaMask, Probe = ChromaF16.Tracing ? TraceProbe.Instance : null };
        Span<Tensor?> attended = [null, null];
        _attention!.Attend(backend, [imgModulated, txtModulated], attended, new JointCallContext { Joint = joint, ReleaseInputs = true });
        Tensor imgAttnProj = _attention.ProjectOutput(backend, 0, attended[0]!);
        Tensor imgAfterAttn = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgAfterAttn, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = _attention.ProjectOutput(backend, 1, attended[1]!);
        Tensor txtAfterAttn = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtAfterAttn, text, txtAttnProj, txtMod[2]);
        txtAttnProj.Dispose();

        // ── 13. Image MLP path ──
        Tensor imgMlpModulated = DiTUtils.NormModulate(backend, imgAfterAttn, imgMod[3], imgMod[4], imgShape);
        Tensor imgMlpOut = _imgFfn.Forward(backend, imgMlpModulated, batch, imgSeqLen);
        imgMlpModulated.Dispose();
        ChromaF16.Probe("dbl-imgMlpOut", imgMlpOut);
        Tensor imgFinal = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgFinal, imgAfterAttn, imgMlpOut, imgMod[5]);
        imgMlpOut.Dispose();
        imgAfterAttn.Dispose();

        // ── 14. Text MLP path ──
        Tensor txtMlpModulated = DiTUtils.NormModulate(backend, txtAfterAttn, txtMod[3], txtMod[4], txtShape);
        Tensor txtMlpOut = _txtFfn.Forward(backend, txtMlpModulated, batch, txtSeqLen);
        txtMlpModulated.Dispose();
        Tensor txtFinal = new Tensor(txtShape, act);
        backend.GatedResidualLastDim(txtFinal, txtAfterAttn, txtMlpOut, txtMod[5]);
        txtMlpOut.Dispose();
        txtAfterAttn.Dispose();

        for (int i = 0; i < imgMod.Length; i++) imgMod[i].Dispose();
        for (int i = 0; i < txtMod.Length; i++) txtMod[i].Dispose();

        ChromaF16.Probe("dbl-imgOut", imgFinal);
        ChromaF16.Probe("dbl-txtOut", txtFinal);
        return (imgFinal, txtFinal);
    }

    /// <summary>Device twin of <see cref="SliceModRows"/> (B=1): per-row <c>backend.SliceRows</c> so the
    /// device-resident temb is never drained to the host mid-forward.</summary>
    internal static Tensor[] SliceModRowsDevice(IBackend backend, Tensor temb, int rowStart, int rowCount)
    {
        int hidden = (int)temb.Shape[2];
        Tensor[] rows = new Tensor[rowCount];
        for (int r = 0; r < rowCount; r++)
        {
            Tensor row = new Tensor(new TensorShape(1, hidden), DType.F32);
            backend.SliceRows(row, temb, rowStart + r);
            rows[r] = row;
        }
        return rows;
    }

    /// <summary>Slices <paramref name="rowCount"/> consecutive rows out of a <c>[B, K, hidden]</c> modulation table
    /// starting at <paramref name="rowStart"/>, returning each row as its own <c>[B, hidden]</c> tensor (B&gt;1 host path).</summary>
    internal static Tensor[] SliceModRows(Tensor temb, int batch, int rowStart, int rowCount)
    {
        int totalRows = (int)temb.Shape[1];
        int hidden = (int)temb.Shape[2];
        Tensor[] rows = new Tensor[rowCount];
        float* tembPtr = (float*)temb.DataPointer;

        for (int r = 0; r < rowCount; r++)
        {
            TensorShape rowShape = new TensorShape(batch, hidden);
            Tensor row = new Tensor(rowShape, DType.F32);
            float* rowPtr = (float*)row.DataPointer;

            for (int b = 0; b < batch; b++)
            {
                long src = ((long)b * totalRows + (rowStart + r)) * hidden;
                long dst = (long)b * hidden;
                Buffer.MemoryCopy(tembPtr + src, rowPtr + dst, hidden * sizeof(float), hidden * sizeof(float));
            }
            rows[r] = row;
        }
        return rows;
    }
}
