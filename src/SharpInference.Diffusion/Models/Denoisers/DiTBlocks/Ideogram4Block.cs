using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Ideogram 4 single-stream transformer block (<c>Ideogram4TransformerBlock</c>), ported verbatim from <c>modeling_ideogram4.py</c>.
///
/// <para>Per-block modulation comes from the shared 512-d AdaLN conditioning <c>c</c>: <c>mod = adaln_modulation(c)</c> → <c>chunk(4)</c> = <c>(scale_msa, gate_msa, scale_mlp, gate_mlp)</c>, then <c>gate = tanh(gate)</c>, <c>scale = 1 + scale</c>. **There is NO shift term** — modulation is scale-only.</para>
///
/// <para>Forward (sandwich norms — a norm before AND after each sublayer):</para>
/// <code>
/// attn = attention(attention_norm1(x) * scale_msa)
/// x    = x + gate_msa * attention_norm2(attn)
/// x    = x + gate_mlp * ffn_norm2(feed_forward(ffn_norm1(x) * scale_mlp))
/// </code>
/// Attention is fused-QKV (<c>Linear(hidden → 3*hidden, bias=False)</c>), per-head QK-RMSNorm, 3D MRoPE, then <c>o</c> (bias=False). FFN is SwiGLU <c>w2(silu(w1)·w3)</c>, all bias=False.</summary>
public sealed unsafe class Ideogram4Block
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnHidden;
    private readonly float _eps;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // RMSNorm scales (read as float* — must be F32 at load).
    private Tensor? _attnNorm1;
    private Tensor? _attnNorm2;
    private Tensor? _ffnNorm1;
    private Tensor? _ffnNorm2;

    // Attention: fused QKV + output, all bias=False.
    private Tensor? _qkvWeight;
    private Tensor? _oWeight;

    // SwiGLU FFN, bias=False.
    private Tensor? _w1; // gate
    private Tensor? _w2; // down
    private Tensor? _w3; // up

    // AdaLN modulation: Linear(adalnDim → 4*hidden), bias=True.
    private Tensor? _adalnWeight;
    private Tensor? _adalnBias;

    public Ideogram4Block(int hidden, int numHeads, int ffnHidden, float eps)
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

    /// <summary>Loads weights using upstream naming. <paramref name="prefix"/> is e.g. <c>"layers.0"</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _attnNorm1 = LoadAsF32(weights, $"{prefix}.attention_norm1.weight");
        _attnNorm2 = LoadAsF32(weights, $"{prefix}.attention_norm2.weight");
        _ffnNorm1 = LoadAsF32(weights, $"{prefix}.ffn_norm1.weight");
        _ffnNorm2 = LoadAsF32(weights, $"{prefix}.ffn_norm2.weight");

        _qkvWeight = weights[$"{prefix}.attention.qkv.weight"];
        _oWeight = weights[$"{prefix}.attention.o.weight"];
        _normQ.LoadWeights(weights[$"{prefix}.attention.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attention.norm_k.weight"]);

        _w1 = weights[$"{prefix}.feed_forward.w1.weight"];
        _w2 = weights[$"{prefix}.feed_forward.w2.weight"];
        _w3 = weights[$"{prefix}.feed_forward.w3.weight"];

        _adalnWeight = weights[$"{prefix}.adaln_modulation.weight"];
        weights.TryGetValue($"{prefix}.adaln_modulation.bias", out _adalnBias);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_attnNorm1 is not null) yield return _attnNorm1;
        if (_attnNorm2 is not null) yield return _attnNorm2;
        if (_ffnNorm1 is not null) yield return _ffnNorm1;
        if (_ffnNorm2 is not null) yield return _ffnNorm2;
        if (_qkvWeight is not null) yield return _qkvWeight;
        if (_oWeight is not null) yield return _oWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_w1 is not null) yield return _w1;
        if (_w2 is not null) yield return _w2;
        if (_w3 is not null) yield return _w3;
        if (_adalnWeight is not null) yield return _adalnWeight;
        if (_adalnBias is not null) yield return _adalnBias;
    }

    /// <summary>Forward pass. <paramref name="x"/> is <c>[B, L, hidden]</c>; <paramref name="adalnInput"/> is the shared conditioning <c>[B, adalnDim]</c>; <paramref name="cos"/>/<paramref name="sin"/> are <c>[B, L, headDim]</c>; <paramref name="attentionMask"/> is optional (null = full attention, correct for single-prompt B=1).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor adalnInput, Tensor cos, Tensor sin,
        Ideogram4Mrope rope, Tensor? attentionMask)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hidden);

        // ── Modulation: chunk(4) → scale_msa, gate_msa, scale_mlp, gate_mlp ──
        (Tensor scaleMsa, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp) = ComputeModulation(backend, adalnInput, batch);

        // ── Attention sublayer ──
        Tensor norm1 = new Tensor(shape, x.DType);
        backend.RmsNorm(norm1, x, _attnNorm1!, _eps);
        Tensor mod1 = ApplyScale(norm1, scaleMsa, batch, seqLen, _hidden);
        norm1.Dispose();

        Tensor attn = Attention(backend, mod1, cos, sin, rope, attentionMask, batch, seqLen);
        mod1.Dispose();

        Tensor attnNormed = new Tensor(shape, x.DType);
        backend.RmsNorm(attnNormed, attn, _attnNorm2!, _eps);
        attn.Dispose();

        Tensor afterAttn = ApplyGatedResidual(x, attnNormed, gateMsa, batch, seqLen, _hidden);
        attnNormed.Dispose();

        // ── MLP sublayer ──
        Tensor norm2 = new Tensor(shape, x.DType);
        backend.RmsNorm(norm2, afterAttn, _ffnNorm1!, _eps);
        Tensor mod2 = ApplyScale(norm2, scaleMlp, batch, seqLen, _hidden);
        norm2.Dispose();

        Tensor mlp = ForwardSwiGlu(backend, mod2, batch, seqLen);
        mod2.Dispose();

        Tensor mlpNormed = new Tensor(shape, x.DType);
        backend.RmsNorm(mlpNormed, mlp, _ffnNorm2!, _eps);
        mlp.Dispose();

        Tensor result = ApplyGatedResidual(afterAttn, mlpNormed, gateMlp, batch, seqLen, _hidden);
        afterAttn.Dispose();
        mlpNormed.Dispose();

        scaleMsa.Dispose();
        gateMsa.Dispose();
        scaleMlp.Dispose();
        gateMlp.Dispose();
        return result;
    }

    /// <summary>Self-attention: fused QKV → per-head QK-RMSNorm → MRoPE → SDPA → output proj.</summary>
    private Tensor Attention(IBackend backend, Tensor input, Tensor cos, Tensor sin,
        Ideogram4Mrope rope, Tensor? attentionMask, int batch, int seqLen)
    {
        TensorShape flat = new TensorShape(batch, seqLen, _hidden);
        TensorShape fused = new TensorShape(batch, seqLen, 3 * _hidden);
        Tensor qkv = new Tensor(fused, input.DType);
        backend.Linear(qkv, input, _qkvWeight!, null);

        // Split fused [B, L, 3*hidden] (layout [q|k|v]) into three [B, L, hidden] chunks.
        Tensor q = SliceChunk(qkv, 0, batch, seqLen);
        Tensor k = SliceChunk(qkv, 1, batch, seqLen);
        Tensor v = SliceChunk(qkv, 2, batch, seqLen);
        qkv.Dispose();

        // Reshape to [B, L, numHeads, headDim] (logical) for QK-norm + RoPE.
        TensorShape heads = new TensorShape(batch, seqLen, _numHeads, _headDim);
        Tensor qH = new Tensor(heads, DType.F32);
        Tensor kH = new Tensor(heads, DType.F32);
        ReshapeFlatToHeads(qH, q);
        ReshapeFlatToHeads(kH, k);
        q.Dispose();
        k.Dispose();

        int totalVecs = batch * seqLen * _numHeads;
        Tensor qN = new Tensor(heads, DType.F32);
        Tensor kN = new Tensor(heads, DType.F32);
        _normQ.Forward(qN, qH, totalVecs);
        _normK.Forward(kN, kH, totalVecs);
        qH.Dispose();
        kH.Dispose();

        rope.ApplyRotary(qN, kN, cos, sin);

        // Permute to [B, numHeads, L, headDim] for SDPA.
        TensorShape mh = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(mh, DType.F32);
        Tensor kMh = new Tensor(mh, DType.F32);
        Tensor vMh = new Tensor(mh, DType.F32);
        PermuteBshdToBhsd(qMh, qN, batch, seqLen);
        PermuteBshdToBhsd(kMh, kN, batch, seqLen);
        ReshapeFlatToBhsd(vMh, v, batch, seqLen);
        qN.Dispose();
        kN.Dispose();
        v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mh, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attentionMask, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        Tensor attnFlat = new Tensor(flat, DType.F32);
        PermuteBhsdToBsh(attnFlat, attnOut, batch, seqLen);
        attnOut.Dispose();

        Tensor projected = new Tensor(flat, input.DType);
        backend.Linear(projected, attnFlat, _oWeight!, null);
        attnFlat.Dispose();
        return projected;
    }

    /// <summary>SwiGLU FFN: <c>w2(silu(w1(x)) * w3(x))</c>, all bias=False.</summary>
    private Tensor ForwardSwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ff = new TensorShape(batch, seqLen, _ffnHidden);
        Tensor gate = new Tensor(ff, input.DType);
        backend.Linear(gate, input, _w1!, null);
        Tensor gateAct = new Tensor(ff, input.DType);
        backend.Silu(gateAct, gate);
        gate.Dispose();

        Tensor up = new Tensor(ff, input.DType);
        backend.Linear(up, input, _w3!, null);

        Tensor combined = new Tensor(ff, input.DType);
        backend.Mul(combined, gateAct, up);
        gateAct.Dispose();
        up.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hidden);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, combined, _w2!, null);
        combined.Dispose();
        return output;
    }

    /// <summary>Computes the 4 modulation vectors from <c>adaln_modulation(c)</c>, applying <c>tanh</c> to gates and <c>1 + ·</c> to scales. Each returned tensor is <c>[B, hidden]</c>.</summary>
    private (Tensor ScaleMsa, Tensor GateMsa, Tensor ScaleMlp, Tensor GateMlp) ComputeModulation(IBackend backend, Tensor adalnInput, int batch)
    {
        int four = 4 * _hidden;
        TensorShape projShape = new TensorShape(batch, four);
        Tensor proj = new Tensor(projShape, adalnInput.DType);
        backend.Linear(proj, adalnInput, _adalnWeight!, _adalnBias);

        Tensor scaleMsa = new Tensor(new TensorShape(batch, _hidden), DType.F32);
        Tensor gateMsa = new Tensor(new TensorShape(batch, _hidden), DType.F32);
        Tensor scaleMlp = new Tensor(new TensorShape(batch, _hidden), DType.F32);
        Tensor gateMlp = new Tensor(new TensorShape(batch, _hidden), DType.F32);

        float* p = (float*)proj.DataPointer;
        float* sMsa = (float*)scaleMsa.DataPointer;
        float* gMsa = (float*)gateMsa.DataPointer;
        float* sMlp = (float*)scaleMlp.DataPointer;
        float* gMlp = (float*)gateMlp.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int src = b * four;
            int dst = b * _hidden;
            for (int d = 0; d < _hidden; d++)
            {
                sMsa[dst + d] = 1.0f + p[src + d];
                gMsa[dst + d] = MathF.Tanh(p[src + _hidden + d]);
                sMlp[dst + d] = 1.0f + p[src + 2 * _hidden + d];
                gMlp[dst + d] = MathF.Tanh(p[src + 3 * _hidden + d]);
            }
        }
        proj.Dispose();
        return (scaleMsa, gateMsa, scaleMlp, gateMlp);
    }

    /// <summary><c>out = input * scale[b]</c> (scale already includes the <c>+1</c>), broadcast over the sequence.</summary>
    private static Tensor ApplyScale(Tensor input, Tensor scale, int batch, int seqLen, int hidden)
    {
        TensorShape shape = new TensorShape(batch, seqLen, hidden);
        Tensor output = new Tensor(shape, input.DType);
        float* inPtr = (float*)input.DataPointer;
        float* scPtr = (float*)scale.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int condBase = b * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                int rowOff = (b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                    outPtr[rowOff + d] = inPtr[rowOff + d] * scPtr[condBase + d];
            }
        }
        return output;
    }

    /// <summary><c>out = residual + gate[b] * value</c>, gate broadcast over the sequence.</summary>
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
                    outPtr[rowOff + d] = resPtr[rowOff + d] + gatePtr[condBase + d] * valPtr[rowOff + d];
            }
        }
        return output;
    }

    /// <summary>Extracts chunk <paramref name="which"/> (0=q, 1=k, 2=v) from a fused <c>[B, L, 3*hidden]</c> tensor into a new <c>[B, L, hidden]</c>.</summary>
    private Tensor SliceChunk(Tensor fused, int which, int batch, int seqLen)
    {
        TensorShape shape = new TensorShape(batch, seqLen, _hidden);
        Tensor output = new Tensor(shape, DType.F32);
        float* src = (float*)fused.DataPointer;
        float* dst = (float*)output.DataPointer;
        int stride = 3 * _hidden;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long srcOff = ((long)b * seqLen + s) * stride + (long)which * _hidden;
                long dstOff = ((long)b * seqLen + s) * _hidden;
                Buffer.MemoryCopy(src + srcOff, dst + dstOff, _hidden * sizeof(float), _hidden * sizeof(float));
            }
        }
        return output;
    }

    private void ReshapeFlatToHeads(Tensor output, Tensor input)
    {
        long bytes = input.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy(input.DataPointer, output.DataPointer, bytes, bytes);
    }

    private void ReshapeFlatToBhsd(Tensor output, Tensor input, int batch, int seqLen)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < _numHeads; h++)
                {
                    long inOff = ((long)b * seqLen + s) * _hidden + (long)h * _headDim;
                    long outOff = (((long)b * _numHeads + h) * seqLen + s) * _headDim;
                    Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, _headDim * sizeof(float), _headDim * sizeof(float));
                }
    }

    private void PermuteBshdToBhsd(Tensor output, Tensor input, int batch, int seqLen)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < _numHeads; h++)
                {
                    long inOff = (((long)b * seqLen + s) * _numHeads + h) * _headDim;
                    long outOff = (((long)b * _numHeads + h) * seqLen + s) * _headDim;
                    Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, _headDim * sizeof(float), _headDim * sizeof(float));
                }
    }

    private void PermuteBhsdToBsh(Tensor output, Tensor input, int batch, int seqLen)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < _numHeads; h++)
                {
                    long inOff = (((long)b * _numHeads + h) * seqLen + s) * _headDim;
                    long outOff = ((long)b * seqLen + s) * _hidden + (long)h * _headDim;
                    Buffer.MemoryCopy(inPtr + inOff, outPtr + outOff, _headDim * sizeof(float), _headDim * sizeof(float));
                }
    }

    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
