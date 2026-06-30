using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>GPU-resident single-stream transformer block for Boogu-Image — numerically a drop-in for the table-driven
/// <see cref="OmniGen2Block"/> forward (same Lumina sandwich-norm block), used for Boogu's <c>single_stream_layers</c>
/// and the three refiner stacks (<c>context_refiner</c> non-modulated, <c>noise_refiner</c> / <c>ref_image_refiner</c>
/// modulated). It exists separately from <see cref="OmniGen2Block"/> only so the Boogu perf rewrite can move all glue
/// to the GPU without touching the shared block (which the OmniGen2 model also uses).
///
/// <para>Modulated forward (<c>noise_refiner</c> / <c>ref_image_refiner</c> / <c>single_stream_layers</c>):
/// <code>
/// (norm_h, gate_msa, scale_mlp, gate_mlp) = norm1(h, temb)            // RMSNormZero: shift=None
/// h = h + tanh(gate_msa) * norm2(attn(norm_h))
/// h = h + tanh(gate_mlp) * ffn_norm2(ffn(ffn_norm1(h) * (1 + scale_mlp)))
/// </code>
/// Non-modulated forward (<c>context_refiner</c>): same structure with no gates and no MLP scale.</para>
///
/// <para>GPU-residency (mirrors <see cref="BooguImageDoubleBlock"/> / the verified <see cref="QwenImageBlock"/>): every
/// glue op runs as an <see cref="IBackend"/> GPU op (RMSNorm-zero split via <c>SliceLastDim</c>, <c>(1+scale)</c> via
/// <c>AddScalar</c>+<c>AffineBroadcastLastDim</c>, tanh gate via <c>Tanh</c>+<c>GatedResidualLastDim</c>, heads via
/// <c>Permute0213</c>+<c>RepeatKvHeads</c>+SDPA). Only the interleaved-pairing RoPE stays on the CPU
/// (<c>rope.Apply</c>) — bit-for-bit the old path.</para></summary>
public sealed unsafe class BooguImageSingleBlock
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

    /// <summary>Creates a Boogu single-stream block. Geometry matches <see cref="OmniGen2Block"/>.</summary>
    public BooguImageSingleBlock(int hiddenSize, int numAttentionHeads, int numKvHeads, int headDim, int ffnInnerDim,
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

    /// <summary>Whether this block applies AdaRMSNorm-Zero modulation.</summary>
    public bool Modulation => _modulation;

    /// <summary>Loads weights using the diffusers / upstream Python naming (identical to <see cref="OmniGen2Block"/>).</summary>
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

    /// <summary>Enumerates all weight tensors for GPU preloading (identical set to <see cref="OmniGen2Block"/>).</summary>
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

    /// <summary>Table-driven forward (matches <see cref="OmniGen2Block.Forward(IBackend, Tensor, OmniGen2Rope,
    /// ReadOnlySpan{float}, ReadOnlySpan{float}, Tensor?)"/>): rotates Q/K with the caller-supplied precomputed
    /// <c>(cos, sin)</c> RoPE table. Caller owns <paramref name="hidden"/>; a new output tensor is returned.</summary>
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
                throw new InvalidOperationException("BooguImageSingleBlock: temb is required when modulation=true.");
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
            afterAttn = ApplyTanhGatedResidual(backend, hidden, norm2Out, gateMsa!, batch);
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
            mlpInput = ApplyMlpScale(backend, ffnNorm1Out, scaleMlp!, batch, seqLen);
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
            output = ApplyTanhGatedResidual(backend, afterAttn, ffnNorm2Out, gateMlp!, batch);
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
        // Q/K/V projected into [B, S, H, D] (byte-identical to [B, S, H·D]) so RmsNorm QK-norm normalizes over the
        // head dim and Permute0213 needs no reshape view.
        Tensor q = new Tensor(new TensorShape(batch, seqLen, _numQHeads, _headDim), DType.F32);
        backend.Linear(q, input, _toQWeight!, null);
        Tensor k = new Tensor(new TensorShape(batch, seqLen, _numKvHeads, _headDim), DType.F32);
        backend.Linear(k, input, _toKWeight!, null);
        Tensor v = new Tensor(new TensorShape(batch, seqLen, _numKvHeads, _headDim), DType.F32);
        backend.Linear(v, input, _toVWeight!, null);

        Tensor qn = new Tensor(new TensorShape(batch, seqLen, _numQHeads, _headDim), DType.F32);
        backend.RmsNorm(qn, q, _normQ.Weight, _normQ.Eps);
        q.Dispose();
        Tensor kn = new Tensor(new TensorShape(batch, seqLen, _numKvHeads, _headDim), DType.F32);
        backend.RmsNorm(kn, k, _normK.Weight, _normK.Eps);
        k.Dispose();

        Tensor qMh = new Tensor(new TensorShape(batch, _numQHeads, seqLen, _headDim), DType.F32);
        backend.Permute0213(qMh, qn, seqLen, _numQHeads, _headDim);
        qn.Dispose();
        Tensor kMh = new Tensor(new TensorShape(batch, _numKvHeads, seqLen, _headDim), DType.F32);
        backend.Permute0213(kMh, kn, seqLen, _numKvHeads, _headDim);
        kn.Dispose();
        Tensor vMh = new Tensor(new TensorShape(batch, _numKvHeads, seqLen, _headDim), DType.F32);
        backend.Permute0213(vMh, v, seqLen, _numKvHeads, _headDim);
        v.Dispose();

        // RoPE (CPU; interleaved pairing — see class note).
        rope.Apply(qMh, ropeCos, ropeSin, batch, _numQHeads, seqLen);
        rope.Apply(kMh, ropeCos, ropeSin, batch, _numKvHeads, seqLen);

        Tensor kRep = new Tensor(new TensorShape(batch, _numKvHeads * _kvGroupSize, seqLen, _headDim), DType.F32);
        backend.RepeatKvHeads(kRep, kMh, _numKvHeads, _kvGroupSize);
        Tensor vRep = new Tensor(new TensorShape(batch, _numKvHeads * _kvGroupSize, seqLen, _headDim), DType.F32);
        backend.RepeatKvHeads(vRep, vMh, _numKvHeads, _kvGroupSize);
        kMh.Dispose(); vMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(new TensorShape(batch, _numQHeads, seqLen, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kRep, vRep, null, scale);
        qMh.Dispose(); kRep.Dispose(); vRep.Dispose();

        Tensor attnFlat = new Tensor(new TensorShape(batch, seqLen, _hiddenSize), DType.F32);
        backend.Permute0213(attnFlat, attnMh, _numQHeads, seqLen, _headDim);
        attnMh.Dispose();

        Tensor projected = new Tensor(new TensorShape(batch, seqLen, _hiddenSize), DType.F32);
        backend.Linear(projected, attnFlat, _toOutWeight!, null);
        attnFlat.Dispose();
        return projected;
    }

    /// <summary>Lumina <c>RMSNormZero</c>, GPU-resident. Chunks the <c>[B, 4·hidden]</c> modulation projection into
    /// <c>(scale_msa, gate_msa, scale_mlp, gate_mlp)</c> via <c>SliceLastDim</c>, returns <c>RmsNorm(x)·(1+scale_msa)</c>
    /// and the three remaining chunks. Bit-for-bit the old CPU split.</summary>
    private (Tensor normed, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp) ApplyLuminaRmsNormZero(
        IBackend backend, Tensor input, Tensor temb, int batch, int seqLen)
    {
        Tensor activated = new Tensor(new TensorShape(batch, _conditioningDim), DType.F32);
        backend.Silu(activated, temb);
        Tensor projected = new Tensor(new TensorShape(batch, 4 * _hiddenSize), DType.F32);
        backend.Linear(projected, activated, _norm1ModulationWeight!, _norm1ModulationBias);
        activated.Dispose();

        TensorShape paramShape = new TensorShape(batch, _hiddenSize);
        Tensor scaleMsa = new Tensor(paramShape, DType.F32);
        backend.SliceLastDim(scaleMsa, projected, 0);
        Tensor gateMsa = new Tensor(paramShape, DType.F32);
        backend.SliceLastDim(gateMsa, projected, _hiddenSize);
        Tensor scaleMlp = new Tensor(paramShape, DType.F32);
        backend.SliceLastDim(scaleMlp, projected, 2 * _hiddenSize);
        Tensor gateMlp = new Tensor(paramShape, DType.F32);
        backend.SliceLastDim(gateMlp, projected, 3 * _hiddenSize);
        projected.Dispose();

        TensorShape hShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor rms = new Tensor(hShape, DType.F32);
        backend.RmsNorm(rms, input, _norm1Weight!, _normEps);

        Tensor scalePlus1 = new Tensor(paramShape, DType.F32);
        backend.AddScalar(scalePlus1, scaleMsa, 1.0f);
        scaleMsa.Dispose();
        Tensor normed = new Tensor(hShape, DType.F32);
        backend.AffineBroadcastLastDim(normed, rms, scalePlus1, null);
        rms.Dispose(); scalePlus1.Dispose();

        return (normed, gateMsa, scaleMlp, gateMlp);
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

        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hiddenSize), DType.F32);
        backend.Linear(output, gated, _ffnLinear2Weight!, null);
        gated.Dispose();
        return output;
    }

    /// <summary>MLP scale <c>out = input·(1 + scale_mlp)</c>, GPU-resident (<c>AddScalar</c>+<c>AffineBroadcastLastDim</c>).</summary>
    private Tensor ApplyMlpScale(IBackend backend, Tensor input, Tensor scaleMlp, int batch, int seqLen)
    {
        Tensor scalePlus1 = new Tensor(new TensorShape(batch, _hiddenSize), DType.F32);
        backend.AddScalar(scalePlus1, scaleMlp, 1.0f);
        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hiddenSize), DType.F32);
        backend.AffineBroadcastLastDim(output, input, scalePlus1, null);
        scalePlus1.Dispose();
        return output;
    }

    /// <summary>Tanh-gated residual <c>out = residual + tanh(gate)·value</c>, GPU-resident
    /// (<c>Tanh</c>+<c>GatedResidualLastDim</c>). <paramref name="gate"/> is <c>[B, hidden]</c> broadcast over seq.</summary>
    private Tensor ApplyTanhGatedResidual(IBackend backend, Tensor residual, Tensor value, Tensor gate, int batch)
    {
        Tensor gateTanh = new Tensor(new TensorShape(batch, _hiddenSize), DType.F32);
        backend.Tanh(gateTanh, gate);
        Tensor output = new Tensor(residual.Shape, DType.F32);
        backend.GatedResidualLastDim(output, residual, value, gateTanh);
        gateTanh.Dispose();
        return output;
    }

    private static Tensor CastToF32IfNeeded(Tensor t) =>
        t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
