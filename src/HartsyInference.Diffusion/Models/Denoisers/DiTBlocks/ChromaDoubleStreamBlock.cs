using HartsyInference.Core.Backends;
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

    // Fused attention projections (HARTSY_CHROMA_FUSED_QKV: the converter kept the BFL qkv whole).
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
        // (HARTSY_CHROMA_FUSED_QKV), split projections otherwise.
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
    // GPU-residency rewrite (Chroma-only): all glue (LayerNorm / modulation / QK-norm / reshape-to-heads /
    // joint concat / split / gated-residual) runs as IBackend GPU ops so the activation stays device-resident
    // across the block — no per-op DataPointer reads / D2H sync barriers (which dominated the old ~56s/forward).
    // The single op left on the CPU is RoPE: the GPU `ApplyRope` kernel is rotate-half (NEOX) pairing, whereas
    // Chroma/Flux use interleaved (GPT-J) pairing (`FluxRope` / `ApplyRopeInterleaved`), and the CUDA backend has
    // no interleaved-rope kernel. RoPE is ~5s/run (verified via HARTSY_SKIP_ROPE), so it stays on `rope.Forward`;
    // its D2H/H2D for Q,K is coherent with the activation cache (the sync callback evicts then re-uploads).
    // Mirrors Ideogram4Block. NOTE: batch is always 1
    // for Chroma (ChromaPipeline runs CFG as two separate batch-1 passes); the seq-dim split uses that.
    public (Tensor image, Tensor text) Forward(
        IBackend backend, Tensor image, Tensor text, Tensor temb, FluxRope rope, Tensor? sdpaMask)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        // Activation dtype follows the INPUT (the Krea2/Z-Image pattern): ChromaTransformer casts the token
        // streams to F16 once before the block loop on the HARTSY_DIT_F16 path; the tiny modulation vectors
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
        // [B, S, H, D] views (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape imgHeads = new TensorShape(batch, imgSeqLen, _numHeads, _headDim);
        TensorShape txtHeads = new TensorShape(batch, txtSeqLen, _numHeads, _headDim);
        TensorShape jointFlat = new TensorShape(batch, totalSeqLen, _hiddenSize);
        TensorShape jointMh = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);

        // ── 2. LayerNorm (no affine, eps=1e-6) + modulate: x*(1+scale)+shift ──
        Tensor imgModulated = DiTUtils.NormModulate(backend, image, imgMod[0], imgMod[1], imgShape);
        Tensor txtModulated = DiTUtils.NormModulate(backend, text, txtMod[0], txtMod[1], txtShape);

        // ── 3+4. Q/K/V projections + QK-Norm. Fused path (HARTSY_CHROMA_FUSED_QKV): one qkv GEMM per
        //         stream + QkvSplitNorm (split + per-head RMSNorm + v copy in one kernel — 3 GEMMs +
        //         2 RmsNorm passes → 1 GEMM + 1 kernel, the Hunyuan3DFluxBlocks recipe). Split path:
        //         3 Linears per stream + 2 separate RmsNorm passes (outputs declared [B, S, H, D]). ──
        Tensor imgQn, imgKn, imgV, txtQn, txtKn, txtV;
        if (_imgQkvWeight is not null)
        {
            Tensor imgQkv = new Tensor(new TensorShape(batch, imgSeqLen, 3 * _hiddenSize), act);
            backend.Linear(imgQkv, imgModulated, _imgQkvWeight, _imgQkvBias);
            imgModulated.Dispose();
            imgQn = new Tensor(imgHeads, act);
            imgKn = new Tensor(imgHeads, act);
            imgV = new Tensor(imgHeads, act);
            backend.QkvSplitNorm(imgQn, imgKn, imgV, imgQkv, _normQ.Weight, _normK.Weight, _normQ.Eps);
            imgQkv.Dispose();

            Tensor txtQkv = new Tensor(new TensorShape(batch, txtSeqLen, 3 * _hiddenSize), act);
            backend.Linear(txtQkv, txtModulated, _txtQkvWeight!, _txtQkvBias);
            txtModulated.Dispose();
            txtQn = new Tensor(txtHeads, act);
            txtKn = new Tensor(txtHeads, act);
            txtV = new Tensor(txtHeads, act);
            backend.QkvSplitNorm(txtQn, txtKn, txtV, txtQkv, _normAddedQ.Weight, _normAddedK.Weight, _normAddedQ.Eps);
            txtQkv.Dispose();
            ChromaF16.Probe("dbl-imgQ", imgQn);
            ChromaF16.Probe("dbl-imgV", imgV);
        }
        else
        {
            Tensor imgQ = new Tensor(imgHeads, act);
            backend.Linear(imgQ, imgModulated, _toQWeight!, _toQBias);
            Tensor imgK = new Tensor(imgHeads, act);
            backend.Linear(imgK, imgModulated, _toKWeight!, _toKBias);
            imgV = new Tensor(imgHeads, act);
            backend.Linear(imgV, imgModulated, _toVWeight!, _toVBias);
            imgModulated.Dispose();

            Tensor txtQ = new Tensor(txtHeads, act);
            backend.Linear(txtQ, txtModulated, _addQWeight!, _addQBias);
            Tensor txtK = new Tensor(txtHeads, act);
            backend.Linear(txtK, txtModulated, _addKWeight!, _addKBias);
            txtV = new Tensor(txtHeads, act);
            backend.Linear(txtV, txtModulated, _addVWeight!, _addVBias);
            txtModulated.Dispose();
            ChromaF16.Probe("dbl-imgQ", imgQ);
            ChromaF16.Probe("dbl-imgV", imgV);

            imgQn = new Tensor(imgHeads, act);
            backend.RmsNorm(imgQn, imgQ, _normQ.Weight, _normQ.Eps);
            imgQ.Dispose();
            imgKn = new Tensor(imgHeads, act);
            backend.RmsNorm(imgKn, imgK, _normK.Weight, _normK.Eps);
            imgK.Dispose();
            txtQn = new Tensor(txtHeads, act);
            backend.RmsNorm(txtQn, txtQ, _normAddedQ.Weight, _normAddedQ.Eps);
            txtQ.Dispose();
            txtKn = new Tensor(txtHeads, act);
            backend.RmsNorm(txtKn, txtK, _normAddedK.Weight, _normAddedK.Eps);
            txtK.Dispose();
        }

        // ── 5. Concat [txt, img] along the seq dim (contiguous row-concat in [B, S, hidden] layout) ──
        Tensor jointQf = new Tensor(jointFlat, act);
        backend.Concat(jointQf, new Tensor[] { txtQn, imgQn }, 1);
        Tensor jointKf = new Tensor(jointFlat, act);
        backend.Concat(jointKf, new Tensor[] { txtKn, imgKn }, 1);
        Tensor jointVf = new Tensor(jointFlat, act);
        backend.Concat(jointVf, new Tensor[] { txtV, imgV }, 1);
        txtQn.Dispose(); imgQn.Dispose();
        txtKn.Dispose(); imgKn.Dispose();
        txtV.Dispose(); imgV.Dispose();

        // ── 6. RoPE on concatenated Q and K. B=1 (pipeline case): GPU-resident on the pre-permute
        //       [B, S, H, D] layout (device WanRopeInterleaved, no D2H). B>1: host Forward post-permute. ──
        bool gpuRope = batch == 1;
        if (gpuRope)
        {
            rope.ApplyGpu(backend, jointQf, jointKf, _numHeads);
        }

        // ── 7. Permute [B, S, H, D] → [B, H, S, D] for SDPA ──
        Tensor jointQ = new Tensor(jointMh, act);
        backend.Permute0213(jointQ, jointQf, totalSeqLen, _numHeads, _headDim);
        jointQf.Dispose();
        Tensor jointK = new Tensor(jointMh, act);
        backend.Permute0213(jointK, jointKf, totalSeqLen, _numHeads, _headDim);
        jointKf.Dispose();
        Tensor jointV = new Tensor(jointMh, act);
        backend.Permute0213(jointV, jointVf, totalSeqLen, _numHeads, _headDim);
        jointVf.Dispose();

        if (!gpuRope)
        {
            rope.Forward(jointQ, jointK, batch, _numHeads, totalSeqLen);
        }

        // ── 8. Scaled dot-product attention. The additive [B,1,S,S] mask is built ONCE per forward by
        //       ChromaTransformer and shared across all blocks — use it directly, do NOT dispose here.
        //       allowF16 is safe: Q/K are RMS-normed (QkNorm) so scores are bounded; the mask rides the cuDNN
        //       fused engine as an fp32 bias score-modifier, never rounded through F16. ──
        Tensor jointAttnOut = new Tensor(jointMh, act);
        backend.ScaledDotProductAttention(jointAttnOut, jointQ, jointK, jointV, sdpaMask, scale, allowF16: true);
        jointQ.Dispose();
        jointK.Dispose();
        jointV.Dispose();
        ChromaF16.Probe("dbl-attnOut", jointAttnOut);

        // ── 10. Permute back [B, H, S, D] → [B, S, hidden] ──
        Tensor jointAttnFlat = new Tensor(jointFlat, act);
        backend.Permute0213(jointAttnFlat, jointAttnOut, _numHeads, totalSeqLen, _headDim);
        jointAttnOut.Dispose();

        // ── 11. Split [txt, img] back along the seq dim (B=1: contiguous row ranges) ──
        Tensor txtAttn = new Tensor(txtShape, act);
        backend.SliceRows(txtAttn, jointAttnFlat, 0);
        Tensor imgAttn = new Tensor(imgShape, act);
        backend.SliceRows(imgAttn, jointAttnFlat, txtSeqLen);
        jointAttnFlat.Dispose();

        // ── 12. Output projections + gated residual ──
        Tensor imgAttnProj = new Tensor(imgShape, act);
        backend.Linear(imgAttnProj, imgAttn, _toOutWeight!, _toOutBias);
        imgAttn.Dispose();
        Tensor imgAfterAttn = new Tensor(imgShape, act);
        backend.GatedResidualLastDim(imgAfterAttn, image, imgAttnProj, imgMod[2]);
        imgAttnProj.Dispose();

        Tensor txtAttnProj = new Tensor(txtShape, act);
        backend.Linear(txtAttnProj, txtAttn, _toAddOutWeight!, _toAddOutBias);
        txtAttn.Dispose();
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
