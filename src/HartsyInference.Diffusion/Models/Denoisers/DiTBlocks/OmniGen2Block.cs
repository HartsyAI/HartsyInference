using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>OmniGen2 single-stream transformer block (<c>OmniGen2TransformerBlock</c> from
/// <c>VectorSpaceLab/OmniGen2/omnigen2/models/transformers/transformer_omnigen2.py</c>). Each block holds a
/// self-attention layer with grouped-query attention (Q heads = <c>NumAttentionHeads</c>, KV heads =
/// <c>NumKvHeads</c>, with K/V repeat-interleaved up to Q head count before SDPA — the upstream code does the
/// same to avoid the slow MATH SDPA path), per-head RMSNorm on Q and K (<c>qk_norm="rms_norm"</c>), a SwiGLU
/// (Lumina-style) FFN with three bias-free linears, and four norms: <c>norm1</c> (LuminaRMSNormZero when
/// <see cref="_modulation"/>; plain RMSNorm otherwise), <c>norm2</c>, <c>ffn_norm1</c>, <c>ffn_norm2</c>.
/// <para>Modulated forward (used by <c>noise_refiner</c> and the joint <c>layers</c> stack):
/// <code>
/// (norm_h, gate_msa, scale_mlp, gate_mlp) = norm1(h, temb)            // RMSNormZero: shift=None
/// h = h + tanh(gate_msa).unsqueeze(1) * norm2(attn(norm_h))
/// h = h + tanh(gate_mlp).unsqueeze(1) * ffn_norm2(ffn(ffn_norm1(h) * (1 + scale_mlp.unsqueeze(1))))
/// </code></para>
/// <para>Non-modulated forward (used by <c>context_refiner</c>): same structure with no gates and no MLP scale.</para>
/// <para>RoPE is applied to Q and K outside the block (in the transformer's RoPE pass) before this method is
/// called — the block expects already-rotated Q/K in its internal attention call.</para></summary>
public sealed unsafe class OmniGen2Block
{
    private readonly int _hiddenSize;
    private readonly int _numQHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _kvGroupSize;
    private readonly int _ffnInnerDim;
    private readonly int _conditioningDim;
    private readonly bool _modulation;
    private readonly float _normEps;

    private Tensor? _toQWeight, _toKWeight, _toVWeight, _toOutWeight;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    private Tensor? _ffnLinear1Weight;
    private Tensor? _ffnLinear2Weight;
    private Tensor? _ffnLinear3Weight;

    private Tensor? _norm1Weight;
    private Tensor? _norm1ModulationWeight, _norm1ModulationBias;
    private Tensor? _norm2Weight;
    private Tensor? _ffnNorm1Weight;
    private Tensor? _ffnNorm2Weight;

    /// <summary>Creates an OmniGen2 block.</summary>
    /// <param name="hiddenSize">Model hidden dimension (2520 for V1).</param>
    /// <param name="numAttentionHeads">Number of Q heads (21 for V1).</param>
    /// <param name="numKvHeads">Number of KV heads for GQA (7 for V1).</param>
    /// <param name="headDim">Per-head dim (= hiddenSize / numAttentionHeads = 120 for V1).</param>
    /// <param name="ffnInnerDim">Lumina-rounded FFN inner dim (caller computes via <c>multiple_of</c> + optional multiplier).</param>
    /// <param name="conditioningDim">Time-embedding dim fed into the modulation linear (<c>min(hidden, 1024)</c>).</param>
    /// <param name="modulation">When true, <c>norm1</c> is <c>LuminaRMSNormZero</c> and gates are applied; when false, plain RMSNorm and no gates (used by <c>context_refiner</c>).</param>
    /// <param name="normEps">RMSNorm epsilon (1e-5 for V1).</param>
    /// <param name="qkNormEps">Per-head Q/K RMSNorm epsilon (1e-5 for V1).</param>
    public OmniGen2Block(int hiddenSize, int numAttentionHeads, int numKvHeads, int headDim, int ffnInnerDim,
        int conditioningDim, bool modulation, float normEps = 1e-5f, float qkNormEps = 1e-5f)
    {
        if (numAttentionHeads * headDim != hiddenSize)
            throw new ArgumentException($"numAttentionHeads * headDim ({numAttentionHeads} * {headDim}) must equal hiddenSize ({hiddenSize}).");
        if (numAttentionHeads % numKvHeads != 0)
            throw new ArgumentException($"numAttentionHeads ({numAttentionHeads}) must be divisible by numKvHeads ({numKvHeads}).");

        _hiddenSize = hiddenSize;
        _numQHeads = numAttentionHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _kvGroupSize = numAttentionHeads / numKvHeads;
        _ffnInnerDim = ffnInnerDim;
        _conditioningDim = conditioningDim;
        _modulation = modulation;
        _normEps = normEps;

        _normQ = new QkNorm(headDim, qkNormEps);
        _normK = new QkNorm(headDim, qkNormEps);
    }

    /// <summary>Whether this block applies AdaRMSNorm-Zero modulation. Read by callers that want to skip
    /// passing temb to non-modulated <c>context_refiner</c> blocks.</summary>
    public bool Modulation => _modulation;

    /// <summary>Loads weights using the diffusers / upstream Python naming (<c>{prefix}.attn.to_q.weight</c>,
    /// <c>{prefix}.feed_forward.linear_1.weight</c>, <c>{prefix}.norm1.linear.weight</c>, etc.). All linears in
    /// the attention path and the FFN are bias-free; only the modulation linear has bias (matching upstream
    /// <c>LuminaRMSNormZero(linear, bias=True)</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);

        _ffnLinear1Weight = weights[$"{prefix}.feed_forward.linear_1.weight"];
        _ffnLinear2Weight = weights[$"{prefix}.feed_forward.linear_2.weight"];
        _ffnLinear3Weight = weights[$"{prefix}.feed_forward.linear_3.weight"];

        if (_modulation)
        {
            _norm1ModulationWeight = weights[$"{prefix}.norm1.linear.weight"];
            _norm1ModulationBias = weights[$"{prefix}.norm1.linear.bias"];
            _norm1Weight = CastToF32IfNeeded(weights[$"{prefix}.norm1.norm.weight"]);
        }
        else
        {
            _norm1Weight = CastToF32IfNeeded(weights[$"{prefix}.norm1.weight"]);
        }

        _norm2Weight = CastToF32IfNeeded(weights[$"{prefix}.norm2.weight"]);
        _ffnNorm1Weight = CastToF32IfNeeded(weights[$"{prefix}.ffn_norm1.weight"]);
        _ffnNorm2Weight = CastToF32IfNeeded(weights[$"{prefix}.ffn_norm2.weight"]);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toOutWeight is not null) yield return _toOutWeight;

        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;

        if (_ffnLinear1Weight is not null) yield return _ffnLinear1Weight;
        if (_ffnLinear2Weight is not null) yield return _ffnLinear2Weight;
        if (_ffnLinear3Weight is not null) yield return _ffnLinear3Weight;

        if (_norm1Weight is not null) yield return _norm1Weight;
        if (_norm1ModulationWeight is not null) yield return _norm1ModulationWeight;
        if (_norm1ModulationBias is not null) yield return _norm1ModulationBias;
        if (_norm2Weight is not null) yield return _norm2Weight;
        if (_ffnNorm1Weight is not null) yield return _ffnNorm1Weight;
        if (_ffnNorm2Weight is not null) yield return _ffnNorm2Weight;
    }

    /// <summary>Forward pass.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="hidden">Block input <c>[B, S, hiddenSize]</c>. Caller owns the lifetime — this method allocates a new output tensor.</param>
    /// <param name="rope">Shared 3-axis RoPE.</param>
    /// <param name="ropeMode">How to position tokens for RoPE. Either <see cref="RopeApplyMode.Text"/> (positions <c>(s,s,s)</c>) or <see cref="RopeApplyMode.Image"/> (positions <c>(timeOffset, row, col)</c>).</param>
    /// <param name="hPacked">Image-mode only: packed grid height (latent_h / patch).</param>
    /// <param name="wPacked">Image-mode only: packed grid width (latent_w / patch).</param>
    /// <param name="timeOffset">Image-mode only: time-axis offset (= text caption length).</param>
    /// <param name="temb">Conditioning <c>[B, conditioningDim]</c> for modulated blocks; ignored when <see cref="Modulation"/> is false.</param>
    public Tensor Forward(IBackend backend, Tensor hidden, OmniGen2Rope rope, RopeApplyMode ropeMode,
        int hPacked, int wPacked, int timeOffset, Tensor? temb)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        TensorShape hShape = new TensorShape(batch, seqLen, _hiddenSize);

        Tensor? gateMsa = null;
        Tensor? scaleMlp = null;
        Tensor? gateMlp = null;
        Tensor norm1Out;
        if (_modulation)
        {
            if (temb is null)
                throw new InvalidOperationException("OmniGen2Block: temb is required when modulation=true.");
            (norm1Out, gateMsa, scaleMlp, gateMlp) = ApplyLuminaRmsNormZero(backend, hidden, temb, batch, seqLen);
        }
        else
        {
            norm1Out = new Tensor(hShape, DType.F32);
            backend.RmsNorm(norm1Out, hidden, _norm1Weight!, _normEps);
        }

        Tensor attnOut = ComputeSelfAttention(backend, norm1Out, rope, ropeMode, hPacked, wPacked, timeOffset, batch, seqLen);
        norm1Out.Dispose();

        Tensor norm2Out = new Tensor(hShape, DType.F32);
        backend.RmsNorm(norm2Out, attnOut, _norm2Weight!, _normEps);
        attnOut.Dispose();

        Tensor afterAttn;
        if (_modulation)
        {
            afterAttn = ApplyTanhGatedResidual(hidden, norm2Out, gateMsa!, batch, seqLen);
            norm2Out.Dispose();
        }
        else
        {
            afterAttn = new Tensor(hShape, DType.F32);
            backend.Add(afterAttn, hidden, norm2Out);
            norm2Out.Dispose();
        }

        Tensor ffnNorm1Out = new Tensor(hShape, DType.F32);
        backend.RmsNorm(ffnNorm1Out, afterAttn, _ffnNorm1Weight!, _normEps);

        Tensor mlpInput;
        if (_modulation)
        {
            mlpInput = ApplyMlpScale(ffnNorm1Out, scaleMlp!, batch, seqLen);
            ffnNorm1Out.Dispose();
        }
        else
        {
            mlpInput = ffnNorm1Out;
        }

        Tensor mlpOut = ApplyLuminaSwiGluFfn(backend, mlpInput, batch, seqLen);
        mlpInput.Dispose();

        Tensor ffnNorm2Out = new Tensor(hShape, DType.F32);
        backend.RmsNorm(ffnNorm2Out, mlpOut, _ffnNorm2Weight!, _normEps);
        mlpOut.Dispose();

        Tensor output;
        if (_modulation)
        {
            output = ApplyTanhGatedResidual(afterAttn, ffnNorm2Out, gateMlp!, batch, seqLen);
            ffnNorm2Out.Dispose();
            afterAttn.Dispose();

            gateMsa!.Dispose();
            scaleMlp!.Dispose();
            gateMlp!.Dispose();
        }
        else
        {
            output = new Tensor(hShape, DType.F32);
            backend.Add(output, afterAttn, ffnNorm2Out);
            afterAttn.Dispose();
            ffnNorm2Out.Dispose();
        }

        return output;
    }

    /// <summary>Table-driven forward: identical to <see cref="Forward"/> but rotates Q/K with a caller-supplied
    /// precomputed <c>(cos, sin)</c> RoPE table (sized <c>seqLen · headDim/2</c>) instead of deriving positions from a
    /// <see cref="RopeApplyMode"/>. Used by Boogu-Image, whose edit path assigns non-default position ids
    /// (reference-image <c>pe_shift</c> offsets) that the mode-based builders don't cover. Numerically equal to the
    /// mode-based path when fed the table that mode would have produced.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, OmniGen2Rope rope,
        ReadOnlySpan<float> ropeCos, ReadOnlySpan<float> ropeSin, Tensor? temb)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        TensorShape hShape = new TensorShape(batch, seqLen, _hiddenSize);

        Tensor? gateMsa = null, scaleMlp = null, gateMlp = null;
        Tensor norm1Out;
        if (_modulation)
        {
            if (temb is null)
                throw new InvalidOperationException("OmniGen2Block: temb is required when modulation=true.");
            (norm1Out, gateMsa, scaleMlp, gateMlp) = ApplyLuminaRmsNormZero(backend, hidden, temb, batch, seqLen);
        }
        else
        {
            norm1Out = new Tensor(hShape, DType.F32);
            backend.RmsNorm(norm1Out, hidden, _norm1Weight!, _normEps);
        }

        Tensor attnOut = ComputeSelfAttentionWithTable(backend, norm1Out, rope, ropeCos, ropeSin, batch, seqLen);
        norm1Out.Dispose();

        Tensor norm2Out = new Tensor(hShape, DType.F32);
        backend.RmsNorm(norm2Out, attnOut, _norm2Weight!, _normEps);
        attnOut.Dispose();

        Tensor afterAttn;
        if (_modulation)
        {
            afterAttn = ApplyTanhGatedResidual(hidden, norm2Out, gateMsa!, batch, seqLen);
            norm2Out.Dispose();
        }
        else
        {
            afterAttn = new Tensor(hShape, DType.F32);
            backend.Add(afterAttn, hidden, norm2Out);
            norm2Out.Dispose();
        }

        Tensor ffnNorm1Out = new Tensor(hShape, DType.F32);
        backend.RmsNorm(ffnNorm1Out, afterAttn, _ffnNorm1Weight!, _normEps);

        Tensor mlpInput;
        if (_modulation)
        {
            mlpInput = ApplyMlpScale(ffnNorm1Out, scaleMlp!, batch, seqLen);
            ffnNorm1Out.Dispose();
        }
        else
        {
            mlpInput = ffnNorm1Out;
        }

        Tensor mlpOut = ApplyLuminaSwiGluFfn(backend, mlpInput, batch, seqLen);
        mlpInput.Dispose();

        Tensor ffnNorm2Out = new Tensor(hShape, DType.F32);
        backend.RmsNorm(ffnNorm2Out, mlpOut, _ffnNorm2Weight!, _normEps);
        mlpOut.Dispose();

        Tensor output;
        if (_modulation)
        {
            output = ApplyTanhGatedResidual(afterAttn, ffnNorm2Out, gateMlp!, batch, seqLen);
            ffnNorm2Out.Dispose();
            afterAttn.Dispose();
            gateMsa!.Dispose();
            scaleMlp!.Dispose();
            gateMlp!.Dispose();
        }
        else
        {
            output = new Tensor(hShape, DType.F32);
            backend.Add(output, afterAttn, ffnNorm2Out);
            afterAttn.Dispose();
            ffnNorm2Out.Dispose();
        }

        return output;
    }

    private Tensor ComputeSelfAttentionWithTable(IBackend backend, Tensor input, OmniGen2Rope rope,
        ReadOnlySpan<float> ropeCos, ReadOnlySpan<float> ropeSin, int batch, int seqLen)
    {
        int qDim = _numQHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;

        TensorShape qShape = new TensorShape(batch, seqLen, qDim);
        TensorShape kvShape = new TensorShape(batch, seqLen, kvDim);
        Tensor qProj = new Tensor(qShape, DType.F32);
        Tensor kProj = new Tensor(kvShape, DType.F32);
        Tensor vProj = new Tensor(kvShape, DType.F32);
        backend.Linear(qProj, input, _toQWeight!, null);
        backend.Linear(kProj, input, _toKWeight!, null);
        backend.Linear(vProj, input, _toVWeight!, null);

        Tensor qNormed = new Tensor(qShape, DType.F32);
        Tensor kNormed = new Tensor(kvShape, DType.F32);
        _normQ.Forward(qNormed, qProj, batch * seqLen * _numQHeads);
        _normK.Forward(kNormed, kProj, batch * seqLen * _numKvHeads);
        qProj.Dispose();
        kProj.Dispose();

        Tensor qMh = DiTUtils.ReshapeToMultiHead(qNormed, batch, seqLen, _numQHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(kNormed, batch, seqLen, _numKvHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(vProj, batch, seqLen, _numKvHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        vProj.Dispose();

        rope.Apply(qMh, ropeCos, ropeSin, batch, _numQHeads, seqLen);
        rope.Apply(kMh, ropeCos, ropeSin, batch, _numKvHeads, seqLen);

        Tensor kRepeated = RepeatHeads(kMh, batch, _numKvHeads, _kvGroupSize, seqLen, _headDim);
        Tensor vRepeated = RepeatHeads(vMh, batch, _numKvHeads, _kvGroupSize, seqLen, _headDim);
        kMh.Dispose();
        vMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(new TensorShape(batch, _numQHeads, seqLen, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kRepeated, vRepeated, null, scale);
        qMh.Dispose();
        kRepeated.Dispose();
        vRepeated.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor attnFlat = new Tensor(outShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(attnFlat, attnMh, batch, seqLen, _numQHeads, _headDim);
        attnMh.Dispose();

        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, attnFlat, _toOutWeight!, null);
        attnFlat.Dispose();
        return projected;
    }

    private (Tensor normed, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp) ApplyLuminaRmsNormZero(
        IBackend backend, Tensor input, Tensor temb, int batch, int seqLen)
    {
        TensorShape modIn = new TensorShape(batch, _conditioningDim);
        Tensor activated = new Tensor(modIn, DType.F32);
        backend.Silu(activated, temb);

        TensorShape projShape = new TensorShape(batch, 4 * _hiddenSize);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, activated, _norm1ModulationWeight!, _norm1ModulationBias);
        activated.Dispose();

        TensorShape paramShape = new TensorShape(batch, _hiddenSize);
        Tensor scaleMsa = new Tensor(paramShape, DType.F32);
        Tensor gateMsa = new Tensor(paramShape, DType.F32);
        Tensor scaleMlp = new Tensor(paramShape, DType.F32);
        Tensor gateMlp = new Tensor(paramShape, DType.F32);

        float* projPtr = (float*)projected.DataPointer;
        float* scaleMsaPtr = (float*)scaleMsa.DataPointer;
        float* gateMsaPtr = (float*)gateMsa.DataPointer;
        float* scaleMlpPtr = (float*)scaleMlp.DataPointer;
        float* gateMlpPtr = (float*)gateMlp.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int srcBase = b * 4 * _hiddenSize;
            int dstBase = b * _hiddenSize;
            for (int d = 0; d < _hiddenSize; d++)
            {
                scaleMsaPtr[dstBase + d] = projPtr[srcBase + 0 * _hiddenSize + d];
                gateMsaPtr[dstBase + d] = projPtr[srcBase + 1 * _hiddenSize + d];
                scaleMlpPtr[dstBase + d] = projPtr[srcBase + 2 * _hiddenSize + d];
                gateMlpPtr[dstBase + d] = projPtr[srcBase + 3 * _hiddenSize + d];
            }
        }
        projected.Dispose();

        TensorShape hShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor rms = new Tensor(hShape, DType.F32);
        backend.RmsNorm(rms, input, _norm1Weight!, _normEps);

        Tensor normed = new Tensor(hShape, DType.F32);
        float* rmsPtr = (float*)rms.DataPointer;
        float* normPtr = (float*)normed.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int condBase = b * _hiddenSize;
            for (int s = 0; s < seqLen; s++)
            {
                int vecBase = (b * seqLen + s) * _hiddenSize;
                for (int d = 0; d < _hiddenSize; d++)
                {
                    normPtr[vecBase + d] = rmsPtr[vecBase + d] * (1.0f + scaleMsaPtr[condBase + d]);
                }
            }
        }
        rms.Dispose();
        scaleMsa.Dispose();

        return (normed, gateMsa, scaleMlp, gateMlp);
    }

    private Tensor ComputeSelfAttention(IBackend backend, Tensor input, OmniGen2Rope rope, RopeApplyMode ropeMode,
        int hPacked, int wPacked, int timeOffset, int batch, int seqLen)
    {
        int qDim = _numQHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;

        TensorShape qShape = new TensorShape(batch, seqLen, qDim);
        TensorShape kvShape = new TensorShape(batch, seqLen, kvDim);
        Tensor qProj = new Tensor(qShape, DType.F32);
        Tensor kProj = new Tensor(kvShape, DType.F32);
        Tensor vProj = new Tensor(kvShape, DType.F32);
        backend.Linear(qProj, input, _toQWeight!, null);
        backend.Linear(kProj, input, _toKWeight!, null);
        backend.Linear(vProj, input, _toVWeight!, null);

        int qVectors = batch * seqLen * _numQHeads;
        int kvVectors = batch * seqLen * _numKvHeads;
        Tensor qNormed = new Tensor(qShape, DType.F32);
        Tensor kNormed = new Tensor(kvShape, DType.F32);
        _normQ.Forward(qNormed, qProj, qVectors);
        _normK.Forward(kNormed, kProj, kvVectors);
        qProj.Dispose();
        kProj.Dispose();

        TensorShape qMhShape = new TensorShape(batch, _numQHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(batch, _numKvHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(qMhShape, DType.F32);
        Tensor kMh = new Tensor(kvMhShape, DType.F32);
        Tensor vMh = new Tensor(kvMhShape, DType.F32);
        DiTUtils.ReshapeToMultiHead(qMh, qNormed, batch, seqLen, _numQHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(kMh, kNormed, batch, seqLen, _numKvHeads, _headDim);
        DiTUtils.ReshapeToMultiHead(vMh, vProj, batch, seqLen, _numKvHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        vProj.Dispose();

        if (ropeMode == RopeApplyMode.Text)
        {
            rope.ApplyText(qMh, kMh, batch, _numQHeads, _numKvHeads, seqLen);
        }
        else if (ropeMode == RopeApplyMode.Image)
        {
            rope.ApplyImage(qMh, kMh, batch, _numQHeads, _numKvHeads, hPacked, wPacked, timeOffset);
        }
        else if (ropeMode == RopeApplyMode.Joint)
        {
            int txtSeqLen = seqLen - hPacked * wPacked;
            rope.ApplyJoint(qMh, kMh, batch, _numQHeads, _numKvHeads, txtSeqLen, hPacked, wPacked);
        }

        Tensor kRepeated = RepeatHeads(kMh, batch, _numKvHeads, _kvGroupSize, seqLen, _headDim);
        Tensor vRepeated = RepeatHeads(vMh, batch, _numKvHeads, _kvGroupSize, seqLen, _headDim);
        kMh.Dispose();
        vMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        TensorShape attnOutShape = new TensorShape(batch, _numQHeads, seqLen, _headDim);
        Tensor attnMh = new Tensor(attnOutShape, DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kRepeated, vRepeated, null, scale);
        qMh.Dispose();
        kRepeated.Dispose();
        vRepeated.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor attnFlat = new Tensor(outShape, DType.F32);
        DiTUtils.ReshapeFromMultiHead(attnFlat, attnMh, batch, seqLen, _numQHeads, _headDim);
        attnMh.Dispose();

        Tensor projected = new Tensor(outShape, DType.F32);
        backend.Linear(projected, attnFlat, _toOutWeight!, null);
        attnFlat.Dispose();
        return projected;
    }

    /// <summary>Repeat-interleave KV heads up to Q head count along the head axis. Mirrors the upstream
    /// <c>key.repeat_interleave(query.size(-3) // key.size(-3), -3)</c> after the transpose to
    /// <c>(B, H, S, D)</c>.</summary>
    private static Tensor RepeatHeads(Tensor input, int batch, int kvHeads, int groupSize, int seqLen, int headDim)
    {
        if (groupSize == 1)
        {
            TensorShape shape = new TensorShape(batch, kvHeads, seqLen, headDim);
            Tensor copy = new Tensor(shape, DType.F32);
            long bytes = (long)batch * kvHeads * seqLen * headDim * sizeof(float);
            Buffer.MemoryCopy((void*)input.DataPointer, (void*)copy.DataPointer, bytes, bytes);
            return copy;
        }

        int qHeads = kvHeads * groupSize;
        TensorShape outShape = new TensorShape(batch, qHeads, seqLen, headDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long perHeadBytes = (long)seqLen * headDim * sizeof(float);

        for (int b = 0; b < batch; b++)
        {
            for (int kv = 0; kv < kvHeads; kv++)
            {
                long srcOff = ((long)b * kvHeads + kv) * seqLen * headDim;
                for (int g = 0; g < groupSize; g++)
                {
                    int outHead = kv * groupSize + g;
                    long dstOff = ((long)b * qHeads + outHead) * seqLen * headDim;
                    Buffer.MemoryCopy(inPtr + srcOff, outPtr + dstOff, perHeadBytes, perHeadBytes);
                }
            }
        }
        return output;
    }

    private Tensor ApplyLuminaSwiGluFfn(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnInnerDim);
        Tensor h1 = new Tensor(ffShape, DType.F32);
        Tensor h3 = new Tensor(ffShape, DType.F32);
        backend.Linear(h1, input, _ffnLinear1Weight!, null);
        backend.Linear(h3, input, _ffnLinear3Weight!, null);

        Tensor h1Activated = new Tensor(ffShape, DType.F32);
        backend.Silu(h1Activated, h1);
        h1.Dispose();

        Tensor gated = new Tensor(ffShape, DType.F32);
        backend.Mul(gated, h1Activated, h3);
        h1Activated.Dispose();
        h3.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, gated, _ffnLinear2Weight!, null);
        gated.Dispose();
        return output;
    }

    private Tensor ApplyMlpScale(Tensor input, Tensor scaleMlp, int batch, int seqLen)
    {
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(shape, DType.F32);
        float* inPtr = (float*)input.DataPointer;
        float* scalePtr = (float*)scaleMlp.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int condBase = b * _hiddenSize;
            for (int s = 0; s < seqLen; s++)
            {
                int vecBase = (b * seqLen + s) * _hiddenSize;
                for (int d = 0; d < _hiddenSize; d++)
                    outPtr[vecBase + d] = inPtr[vecBase + d] * (1.0f + scalePtr[condBase + d]);
            }
        }
        return output;
    }

    /// <summary>Applies <c>residual + tanh(gate).unsqueeze(1) * value</c> matching the upstream OmniGen2 block
    /// (<c>hidden = hidden + gate.tanh().unsqueeze(1) * x</c>). Gate is <c>[B, hiddenSize]</c>, broadcast over the
    /// sequence axis; tanh is applied per element.</summary>
    private Tensor ApplyTanhGatedResidual(Tensor residual, Tensor value, Tensor gate, int batch, int seqLen)
    {
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(shape, DType.F32);

        float* resPtr = (float*)residual.DataPointer;
        float* valPtr = (float*)value.DataPointer;
        float* gatePtr = (float*)gate.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int gateBase = b * _hiddenSize;
            for (int s = 0; s < seqLen; s++)
            {
                int vecBase = (b * seqLen + s) * _hiddenSize;
                for (int d = 0; d < _hiddenSize; d++)
                {
                    float g = MathF.Tanh(gatePtr[gateBase + d]);
                    outPtr[vecBase + d] = resPtr[vecBase + d] + g * valPtr[vecBase + d];
                }
            }
        }
        return output;
    }

    private static Tensor CastToF32IfNeeded(Tensor t) =>
        t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}

/// <summary>How an <see cref="OmniGen2Block"/> should rotate Q/K. Picked at the call site: text-stream blocks
/// (<c>context_refiner</c>) use <see cref="Text"/>; image-stream blocks (<c>noise_refiner</c>) use <see cref="Image"/>;
/// the joint <c>layers</c> stack uses <see cref="Joint"/> which rotates each token according to whether it falls in
/// the text or image partition.</summary>
public enum RopeApplyMode
{
    /// <summary>No rotation (only used for diagnostic / ablation paths; production code should pick a real mode).</summary>
    None,
    /// <summary>Apply text-stream RoPE: position <c>(s, s, s)</c> per token.</summary>
    Text,
    /// <summary>Apply image-stream RoPE: position <c>(timeOffset, row, col)</c> per token.</summary>
    Image,
    /// <summary>Apply joint-sequence RoPE for <c>[text || image]</c>. The block derives <c>txtSeqLen = seqLen - hPacked * wPacked</c>.</summary>
    Joint,
}
