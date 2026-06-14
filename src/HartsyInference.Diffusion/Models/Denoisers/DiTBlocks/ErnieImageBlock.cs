using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>ERNIE-Image single-stream block (<c>ErnieImageSharedAdaLNBlock</c> in diffusers). Identical structure for every layer; the 6-vector AdaLN modulation is computed ONCE at the transformer level and broadcast into every block (the "shared AdaLN" architecture choice — saves ~300 MB of per-block linear weights vs Flux).
///
/// Sub-block order (per <c>transformer_ernie_image.py:257-275</c>):
/// <list type="number">
///   <item>RMSNorm (<c>adaLN_sa_ln</c>) → <c>x*(1 + scale_msa) + shift_msa</c>.</item>
///   <item>Self-attention with separate Q/K/V projections (NO fused QKV), QK-RMSNorm, 3D RoPE, all linears <c>bias=False</c>.</item>
///   <item>Gated residual: <c>x = residual + gate_msa * attn_out</c>.</item>
///   <item>RMSNorm (<c>adaLN_mlp_ln</c>) → <c>x*(1 + scale_mlp) + shift_mlp</c>.</item>
///   <item>GELU-gated FFN (NOT SwiGLU): <c>linear_fc2(up_proj(x) * gelu(gate_proj(x)))</c>. All linears <c>bias=False</c>.</item>
///   <item>Gated residual: <c>x = residual + gate_mlp * mlp_out</c>.</item>
/// </list>
///
/// Internal layout is <c>[B, S, H]</c> (the Python code internally permutes to <c>[S, B, H]</c> at block boundaries for legacy Megatron parity, but our backend SDPA expects <c>[B, S, H]</c> so we skip the round-trip).</summary>
public sealed unsafe class ErnieImageBlock
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnHidden;
    private readonly float _eps;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // RMSNorm scales — read as float* directly, must be F32 at load time.
    private Tensor? _adaLnSaWeight;
    private Tensor? _adaLnMlpWeight;

    // Attention linears — separate Q/K/V/Out, all bias=False.
    private Tensor? _toQWeight;
    private Tensor? _toKWeight;
    private Tensor? _toVWeight;
    private Tensor? _toOutWeight;

    // GELU-gated FFN linears — bias=False.
    private Tensor? _gateProjWeight;
    private Tensor? _upProjWeight;
    private Tensor? _linearFc2Weight;

    public ErnieImageBlock(int hidden, int numHeads, int ffnHidden, float eps = 1e-6f)
    {
        if (hidden % numHeads != 0)
            throw new ArgumentException($"hidden {hidden} must be divisible by numHeads {numHeads}.", nameof(hidden));
        _hidden = hidden;
        _numHeads = numHeads;
        _headDim = hidden / numHeads;
        _ffnHidden = ffnHidden;
        _eps = eps;
        _normQ = new QkNorm(_headDim, eps);
        _normK = new QkNorm(_headDim, eps);
    }

    /// <summary>Loads weights using diffusers naming. <paramref name="prefix"/> is e.g. <c>"layers.0"</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // RMSNorm scales must be F32 (CudaBackend.RmsNorm + CPU CastTo paths read float* directly).
        _adaLnSaWeight = LoadAsF32(weights, $"{prefix}.adaLN_sa_ln.weight");
        _adaLnMlpWeight = LoadAsF32(weights, $"{prefix}.adaLN_mlp_ln.weight");

        _toQWeight = weights[$"{prefix}.self_attention.to_q.weight"];
        _toKWeight = weights[$"{prefix}.self_attention.to_k.weight"];
        _toVWeight = weights[$"{prefix}.self_attention.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.self_attention.to_out.0.weight"];

        _normQ.LoadWeights(weights[$"{prefix}.self_attention.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.self_attention.norm_k.weight"]);

        _gateProjWeight = weights[$"{prefix}.mlp.gate_proj.weight"];
        _upProjWeight = weights[$"{prefix}.mlp.up_proj.weight"];
        _linearFc2Weight = weights[$"{prefix}.mlp.linear_fc2.weight"];
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_adaLnSaWeight is not null) yield return _adaLnSaWeight;
        if (_adaLnMlpWeight is not null) yield return _adaLnMlpWeight;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toOutWeight is not null) yield return _toOutWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_gateProjWeight is not null) yield return _gateProjWeight;
        if (_upProjWeight is not null) yield return _upProjWeight;
        if (_linearFc2Weight is not null) yield return _linearFc2Weight;
    }

    /// <summary>Forward pass with the shared 6-vector modulation broadcast in.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="x">Token sequence <c>[B, S, hidden]</c>.</param>
    /// <param name="shiftMsa">Attn shift, shape <c>[B, hidden]</c>.</param>
    /// <param name="scaleMsa">Attn scale, shape <c>[B, hidden]</c>.</param>
    /// <param name="gateMsa">Attn output gate, shape <c>[B, hidden]</c>.</param>
    /// <param name="shiftMlp">MLP shift.</param>
    /// <param name="scaleMlp">MLP scale.</param>
    /// <param name="gateMlp">MLP output gate.</param>
    /// <param name="freqs">RoPE freqs from <see cref="ErnieImageRope.BuildFreqs"/>, shape <c>[B, S, 2*head_dim]</c>.</param>
    /// <param name="rope">RoPE applier (holds the axes_dim/theta config).</param>
    /// <param name="attentionMask">Optional attention mask <c>[B, 1, 1, S]</c> — bool-style, where 0=mask out.</param>
    public Tensor Forward(IBackend backend, Tensor x,
        Tensor shiftMsa, Tensor scaleMsa, Tensor gateMsa,
        Tensor shiftMlp, Tensor scaleMlp, Tensor gateMlp,
        Tensor freqs, ErnieImageRope rope, Tensor? attentionMask)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hidden);

        // ── Attention sub-block ────────────────────────────────────────────
        Tensor norm1 = new Tensor(shape, x.DType);
        backend.RmsNorm(norm1, x, _adaLnSaWeight!, _eps);
        Tensor modulated = ApplyShiftScale(norm1, shiftMsa, scaleMsa, batch, seqLen, _hidden);
        norm1.Dispose();

        // Separate Q, K, V projections. Each [B, S, hidden] → [B, S, hidden]; bias=False.
        Tensor q = new Tensor(shape, x.DType);
        Tensor k = new Tensor(shape, x.DType);
        Tensor v = new Tensor(shape, x.DType);
        backend.Linear(q, modulated, _toQWeight!, null);
        backend.Linear(k, modulated, _toKWeight!, null);
        backend.Linear(v, modulated, _toVWeight!, null);
        modulated.Dispose();

        // Reshape to [B, S, numHeads, headDim] for QK-norm + RoPE.
        TensorShape headsShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        Tensor qHeads = new Tensor(headsShape, DType.F32);
        Tensor kHeads = new Tensor(headsShape, DType.F32);
        ReshapeFlatToHeads(qHeads, q, batch, seqLen, _numHeads, _headDim);
        ReshapeFlatToHeads(kHeads, k, batch, seqLen, _numHeads, _headDim);
        q.Dispose();
        k.Dispose();

        // QK-RMSNorm: each [B, S, numHeads, headDim] flattened to (batch*seq*heads) vectors of headDim.
        int totalVecs = batch * seqLen * _numHeads;
        Tensor qNormed = new Tensor(headsShape, DType.F32);
        Tensor kNormed = new Tensor(headsShape, DType.F32);
        _normQ.Forward(qNormed, qHeads, totalVecs);
        _normK.Forward(kNormed, kHeads, totalVecs);
        qHeads.Dispose();
        kHeads.Dispose();

        // 3D RoPE (in-place on qNormed/kNormed, both still [B, S, numHeads, headDim]).
        rope.ApplyRotaryEmb(qNormed, kNormed, freqs);

        // SDPA expects [B, numHeads, S, headDim]. Permute (B, S, H, D) → (B, H, S, D).
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        PermuteBshdToBhsd(qMh, qNormed, batch, seqLen, _numHeads, _headDim);
        PermuteBshdToBhsd(kMh, kNormed, batch, seqLen, _numHeads, _headDim);
        // V never went through the head-split, so reshape from flat [B, S, hidden] directly.
        ReshapeFlatToBhsd(vMh, v, batch, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attentionMask, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // [B, H, S, D] → [B, S, hidden]
        Tensor attnFlat = new Tensor(shape, DType.F32);
        PermuteBhsdToBsh(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, x.DType);
        backend.Linear(projected, attnFlat, _toOutWeight!, null);
        attnFlat.Dispose();

        Tensor afterAttn = ApplyGatedResidual(x, projected, gateMsa, batch, seqLen, _hidden);
        projected.Dispose();

        // ── MLP sub-block ──────────────────────────────────────────────────
        Tensor norm2 = new Tensor(shape, x.DType);
        backend.RmsNorm(norm2, afterAttn, _adaLnMlpWeight!, _eps);
        Tensor modulated2 = ApplyShiftScale(norm2, shiftMlp, scaleMlp, batch, seqLen, _hidden);
        norm2.Dispose();

        Tensor mlpOut = ForwardGeluGatedFfn(backend, modulated2, batch, seqLen);
        modulated2.Dispose();

        Tensor result = ApplyGatedResidual(afterAttn, mlpOut, gateMlp, batch, seqLen, _hidden);
        afterAttn.Dispose();
        mlpOut.Dispose();

        return result;
    }

    /// <summary>GELU-gated FFN: <c>linear_fc2(up_proj(x) * gelu(gate_proj(x)))</c>. Note the multiplicative order — diffusers uses <c>up_proj(x) * gelu(gate_proj(x))</c>, NOT <c>gelu(gate_proj(x)) * up_proj(x)</c>. Float-mul is commutative so the result is identical, but matching layout simplifies dump-diffing.</summary>
    private Tensor ForwardGeluGatedFfn(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnHidden);

        Tensor gate = new Tensor(ffShape, input.DType);
        backend.Linear(gate, input, _gateProjWeight!, null);
        Tensor gateActivated = new Tensor(ffShape, input.DType);
        backend.Gelu(gateActivated, gate);
        gate.Dispose();

        Tensor up = new Tensor(ffShape, input.DType);
        backend.Linear(up, input, _upProjWeight!, null);

        Tensor combined = new Tensor(ffShape, input.DType);
        backend.Mul(combined, up, gateActivated);
        up.Dispose();
        gateActivated.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hidden);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, combined, _linearFc2Weight!, null);
        combined.Dispose();

        return output;
    }

    /// <summary><c>output = input * (1 + scale[b]) + shift[b]</c>. Modulation broadcast over the sequence dim.</summary>
    private static Tensor ApplyShiftScale(Tensor input, Tensor shift, Tensor scale, int batch, int seqLen, int hidden)
    {
        TensorShape shape = new TensorShape(batch, seqLen, hidden);
        Tensor output = new Tensor(shape, input.DType);

        float* inPtr = (float*)input.DataPointer;
        float* shiftPtr = (float*)shift.DataPointer;
        float* scalePtr = (float*)scale.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int condBase = b * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int rowOff = (b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    outPtr[rowOff + d] = inPtr[rowOff + d] * (1.0f + scalePtr[condBase + d]) + shiftPtr[condBase + d];
                }
            }
        }
        return output;
    }

    /// <summary><c>output = residual + gate[b] * value</c>. Broadcast gate over the sequence dim.</summary>
    private static Tensor ApplyGatedResidual(Tensor residual, Tensor value, Tensor gate, int batch, int seqLen, int hidden)
    {
        TensorShape shape = new TensorShape(batch, seqLen, hidden);
        Tensor output = new Tensor(shape, residual.DType);

        float* resPtr = (float*)residual.DataPointer;
        float* valPtr = (float*)value.DataPointer;
        float* gatePtr = (float*)gate.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int condBase = b * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int rowOff = (b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    outPtr[rowOff + d] = resPtr[rowOff + d] + gatePtr[condBase + d] * valPtr[rowOff + d];
                }
            }
        }
        return output;
    }

    /// <summary>Reshape <c>[B, S, hidden]</c> → <c>[B, S, H, D]</c> in-place into a pre-allocated tensor.</summary>
    private static void ReshapeFlatToHeads(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        // Memory layout is identical (just a logical reshape) — copy verbatim.
        long bytes = (long)batch * seqLen * numHeads * headDim * sizeof(float);
        Buffer.MemoryCopy(input.DataPointer, output.DataPointer, bytes, bytes);
    }

    /// <summary>Reshape + permute <c>[B, S, hidden]</c> → <c>[B, H, S, D]</c>.</summary>
    private static void ReshapeFlatToBhsd(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    long inOff = ((long)b * seqLen + s) * numHeads * headDim + (long)h * headDim;
                    long outOff = (((long)b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    /// <summary>Permute <c>[B, S, H, D]</c> → <c>[B, H, S, D]</c>.</summary>
    private static void PermuteBshdToBhsd(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    long inOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    long outOff = (((long)b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    /// <summary>Permute <c>[B, H, S, D]</c> → <c>[B, S, hidden]</c> (flatten heads back into hidden).</summary>
    private static void PermuteBhsdToBsh(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    long inOff = (((long)b * numHeads + h) * seqLen + s) * headDim;
                    long outOff = ((long)b * seqLen + s) * numHeads * headDim + (long)h * headDim;
                    Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    /// <summary>Loads a norm weight, casting to F32 if necessary (RmsNorm reads float* directly).</summary>
    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
