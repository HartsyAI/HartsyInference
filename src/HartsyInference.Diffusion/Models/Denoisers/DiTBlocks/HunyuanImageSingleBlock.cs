using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Hunyuan Image single-stream block (<c>HunyuanImageSingleTransformerBlock</c>). Concatenates <c>[img, txt]</c> first, then runs AdaLayerNormZeroSingle (3 params: <c>shift_msa, scale_msa, gate_msa</c>) on the joint sequence. Q/K/V projection happens on the joint normed sequence; image-only RoPE is applied to the image portion of Q/K. Parallel <c>proj_mlp + GELU(tanh)</c> path is concatenated with the attention output along the feature dim, then <c>proj_out</c> reduces back to <c>hidden_size</c> and a single gated residual is added to the joint sequence. Splits back to <c>(image, text)</c> on output.</summary>
public sealed unsafe class HunyuanImageSingleBlock : IStreamingBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;

    private readonly AdaLNModulation _modulation;
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _projMlpWeight, _projMlpBias;
    private Tensor? _projOutWeight, _projOutBias;

    /// <summary>Creates a Hunyuan Image single-stream block.</summary>
    public HunyuanImageSingleBlock(int hiddenSize, int numHeads, int headDim, int mlpDim, float qkNormEps = 1e-6f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim}) must equal hiddenSize ({hiddenSize}).");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = mlpDim;

        _modulation = new AdaLNModulation(hiddenSize, 3);
        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);
    }

    /// <summary>Loads weights using diffusers naming under <c>single_transformer_blocks.{i}.*</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _modulation.LoadWeights(
            weights[$"{prefix}.norm.linear.weight"],
            weights[$"{prefix}.norm.linear.bias"]);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        weights.TryGetValue($"{prefix}.attn.to_q.bias", out _toQBias);
        weights.TryGetValue($"{prefix}.attn.to_k.bias", out _toKBias);
        weights.TryGetValue($"{prefix}.attn.to_v.bias", out _toVBias);

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);

        _projMlpWeight = weights[$"{prefix}.proj_mlp.weight"];
        weights.TryGetValue($"{prefix}.proj_mlp.bias", out _projMlpBias);

        _projOutWeight = weights[$"{prefix}.proj_out.weight"];
        weights.TryGetValue($"{prefix}.proj_out.bias", out _projOutBias);
    }

    /// <summary>Sum of weight bytes in this streamable block (for the block-streaming budget heuristic).</summary>
    public long EstimatedWeightBytes
    {
        get { long t = 0; foreach (Tensor w in EnumerateWeights()) t += w.ElementCount * w.DType.SizeInBytes; return t; }
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _modulation.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_projMlpWeight is not null) yield return _projMlpWeight;
        if (_projMlpBias is not null) yield return _projMlpBias;
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass on concatenated <c>[image, text]</c> sequence. Image-only RoPE is applied to the image portion of joint Q/K. Returns <c>(image, text)</c>.</summary>
    // GPU-residency rewrite (mirrors the verified QwenImageBlock): every glue op (join concat / LayerNorm / AdaLN
    // modulation / QK-norm / head reshape / feature-dim concat / gated residual / split) runs as an IBackend op so the
    // activation stays device-resident — no per-op DataPointer D2H sync barriers (the old ConcatImageText /
    // DiTUtils / QkNorm / AdaLNModulation / ConcatFeatureDim / SplitImageText host path D2H-synced every intermediate).
    // Head reshape = declaring Q/K/V directly as [B, S, H, D] (byte-identical to [B, S, hidden]) so QK-norm runs over
    // headDim with no reshape, then Permute0213 to/from [B, H, S, D]. The [img, txt] concat preserves image-first
    // order (Concat dim 1); the output split is SliceRows (B=1: image rows first). The ONLY op left on the CPU is the
    // image-only RoPE and its image-portion extract/write (ApplyRopeToImagePortion) — the image sub-region of the
    // joint [B, H, S, D] is per-head strided (not a contiguous row-block), so a clean GPU slice-write is not available;
    // this is one contained host excursion over a small tensor, and RoPE itself is host-only (as in QwenImageBlock).
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor temb,
        HunyuanImageRope rope, int imgPackedH, int imgPackedW, int imgPackedT = 1)
    {
        int batch = (int)image.Shape[0];
        int imgSeqLen = (int)image.Shape[1];
        int txtSeqLen = (int)text.Shape[1];
        int totalSeqLen = imgSeqLen + txtSeqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);

        TensorShape jointShape = new TensorShape(batch, totalSeqLen, _hiddenSize);
        // [B, S, H, D] view (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim.
        TensorShape jointHeads = new TensorShape(batch, totalSeqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, totalSeqLen, _headDim);
        TensorShape mlpShape = new TensorShape(batch, totalSeqLen, _mlpDim);

        // ── 1. Concat [img, txt] along the seq dim (image first) ──
        Tensor joint = new Tensor(jointShape, DType.F32);
        backend.Concat(joint, new Tensor[] { image, text }, 1);

        Tensor[] mod = _modulation.Forward(backend, temb);

        // ── 2. LayerNorm (no affine) + AdaLN modulate ──
        Tensor modulated = DiTUtils.NormModulate(backend, joint, mod[0], mod[1], jointShape);

        // ── 3. Q/K/V (declared [B, S, H, D]) + parallel MLP proj/GELU ──
        Tensor q = new Tensor(jointHeads, DType.F32);
        backend.Linear(q, modulated, _toQWeight!, _toQBias);
        Tensor k = new Tensor(jointHeads, DType.F32);
        backend.Linear(k, modulated, _toKWeight!, _toKBias);
        Tensor v = new Tensor(jointHeads, DType.F32);
        backend.Linear(v, modulated, _toVWeight!, _toVBias);

        Tensor mlpProj = new Tensor(mlpShape, DType.F32);
        backend.Linear(mlpProj, modulated, _projMlpWeight!, _projMlpBias);
        modulated.Dispose();

        Tensor mlpActivated = new Tensor(mlpShape, DType.F32);
        backend.Gelu(mlpActivated, mlpProj);
        mlpProj.Dispose();

        // ── 4. QK-norm (per-head RMSNorm over the last dim = headDim) ──
        Tensor qn = new Tensor(jointHeads, DType.F32);
        backend.RmsNorm(qn, q, _normQ.Weight, _normQ.Eps);
        q.Dispose();
        Tensor kn = new Tensor(jointHeads, DType.F32);
        backend.RmsNorm(kn, k, _normK.Weight, _normK.Eps);
        k.Dispose();

        // ── 5. Permute [B, S, H, D] → [B, H, S, D] ──
        Tensor qMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(qMh, qn, totalSeqLen, _numHeads, _headDim);
        qn.Dispose();
        Tensor kMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(kMh, kn, totalSeqLen, _numHeads, _headDim);
        kn.Dispose();
        Tensor vMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(vMh, v, totalSeqLen, _numHeads, _headDim);
        v.Dispose();

        // ── 6. Image-only RoPE on the image portion of joint Q/K (CPU — see class note) ──
        ApplyRopeToImagePortion(qMh, kMh, rope, batch, _numHeads, totalSeqLen, imgSeqLen, imgPackedH, imgPackedW, imgPackedT);

        // ── 7. Joint scaled dot-product attention (no mask) ──
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // ── 8. Permute back [B, H, S, D] → [B, S, hidden] ──
        Tensor attnFlat = new Tensor(jointShape, DType.F32);
        backend.Permute0213(attnFlat, attnOut, _numHeads, totalSeqLen, _headDim);
        attnOut.Dispose();

        // ── 9. Concat [attn, mlp] along the feature dim, then proj_out ──
        int concatDim = _hiddenSize + _mlpDim;
        TensorShape concatShape = new TensorShape(batch, totalSeqLen, concatDim);
        Tensor concatted = new Tensor(concatShape, DType.F32);
        backend.Concat(concatted, new Tensor[] { attnFlat, mlpActivated }, 2);
        attnFlat.Dispose();
        mlpActivated.Dispose();

        Tensor projOut = new Tensor(jointShape, DType.F32);
        backend.Linear(projOut, concatted, _projOutWeight!, _projOutBias);
        concatted.Dispose();

        // ── 10. Single gated residual + split [img, txt] (B=1: contiguous rows) ──
        Tensor result = new Tensor(jointShape, DType.F32);
        backend.GatedResidualLastDim(result, joint, projOut, mod[2]);
        joint.Dispose();
        projOut.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        TensorShape imgOutShape = new TensorShape(batch, imgSeqLen, _hiddenSize);
        TensorShape txtOutShape = new TensorShape(batch, txtSeqLen, _hiddenSize);
        Tensor imgOut = new Tensor(imgOutShape, DType.F32);
        backend.SliceRows(imgOut, result, 0);
        Tensor txtOut = new Tensor(txtOutShape, DType.F32);
        backend.SliceRows(txtOut, result, imgSeqLen);
        result.Dispose();

        return (imgOut, txtOut);
    }

    /// <summary>Applies image-only RoPE to a joint <c>[B, H, S_img+S_txt, D]</c> Q/K. Image tokens occupy the first <paramref name="imgSeqLen"/> sequence positions.</summary>
    private static void ApplyRopeToImagePortion(Tensor qMh, Tensor kMh, HunyuanImageRope rope,
        int batch, int numHeads, int totalSeqLen, int imgSeqLen, int imgPackedH, int imgPackedW, int imgPackedT = 1)
    {
        int headDim = rope.HeadDim;
        TensorShape imgPortionShape = new TensorShape(batch, numHeads, imgSeqLen, headDim);
        Tensor qImg = new Tensor(imgPortionShape, DType.F32);
        Tensor kImg = new Tensor(imgPortionShape, DType.F32);
        ExtractImagePortion(qImg, qMh, batch, numHeads, totalSeqLen, imgSeqLen, headDim);
        ExtractImagePortion(kImg, kMh, batch, numHeads, totalSeqLen, imgSeqLen, headDim);

        if (imgPackedT > 1)
        {
            Span<int> dims = stackalloc int[3] { imgPackedT, imgPackedH, imgPackedW };
            rope.ApplyJoint(qImg, kImg, batch, numHeads, dims);
        }
        else
            rope.ApplyImage(qImg, kImg, batch, numHeads, imgPackedH, imgPackedW);

        WriteImagePortion(qMh, qImg, batch, numHeads, totalSeqLen, imgSeqLen, headDim);
        WriteImagePortion(kMh, kImg, batch, numHeads, totalSeqLen, imgSeqLen, headDim);
        qImg.Dispose();
        kImg.Dispose();
    }

    private static void ExtractImagePortion(Tensor output, Tensor input,
        int batch, int numHeads, int totalSeqLen, int imgSeqLen, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long copyBytes = (long)imgSeqLen * headDim * sizeof(float);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                long inOffset = ((long)b * numHeads + h) * totalSeqLen * headDim;
                long outOffset = ((long)b * numHeads + h) * imgSeqLen * headDim;
                Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, copyBytes, copyBytes);
            }
        }
    }

    private static void WriteImagePortion(Tensor output, Tensor input,
        int batch, int numHeads, int totalSeqLen, int imgSeqLen, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long copyBytes = (long)imgSeqLen * headDim * sizeof(float);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                long outOffset = ((long)b * numHeads + h) * totalSeqLen * headDim;
                long inOffset = ((long)b * numHeads + h) * imgSeqLen * headDim;
                Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, copyBytes, copyBytes);
            }
        }
    }
}
