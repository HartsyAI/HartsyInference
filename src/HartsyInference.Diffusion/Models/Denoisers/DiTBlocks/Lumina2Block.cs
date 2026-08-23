using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Lumina-Image-2.0 transformer block (NextDiT variant). Used for both <c>noise_refiner</c> blocks and the main 26 <c>layers</c> — they share the same forward (modulation=true). Architecturally the same as <see cref="ZImageBlock"/> in spirit (RMSNorm-Zero modulation with 4 outputs gate_msa, scale_msa, scale_mlp, gate_mlp; tanh on gates; SwiGLU FFN; QK-RMSNorm) but with three concrete differences: (1) <strong>separate Q/K/V projections</strong> (<c>attn.to_q/to_k/to_v</c>) instead of one fused QKV, (2) <strong>grouped-query attention</strong> with <c>num_kv_heads &lt; num_heads</c> requiring K/V repeat-interleave before SDPA, and (3) <strong>diffusers key naming</strong> (<c>norm1.linear.{weight,bias}</c>, <c>norm1.norm.weight</c>, <c>feed_forward.linear_{1,2,3}</c>, <c>attn.norm_q/k.weight</c>, <c>attn.to_out.0.weight</c>, <c>norm2.weight</c>, <c>ffn_norm{1,2}.weight</c>). All linears NO bias except <c>norm1.linear.bias</c>.</summary>
public sealed unsafe class Lumina2Block
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _qDim;
    private readonly int _kvDim;
    private readonly int _ffnDim;
    private readonly int _adaLNEmbedDim;
    private readonly float _eps;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // norm1: LuminaRMSNormZero. Combines pre-attention RMSNorm (norm1.norm.weight, [hidden]) and the
    // AdaLN modulation Linear (norm1.linear.{weight,bias}, [4*hidden, adaLNEmbedDim]) into a single
    // module. Output: (norm(x)*(1+scale_msa), gate_msa, scale_mlp, gate_mlp). The Linear projects from
    // adaLNEmbedDim (1024 for Lumina-Image-2.0) to 4*hidden.
    private Tensor? _norm1NormWeight;
    private Tensor? _norm1LinearWeight;
    private Tensor? _norm1LinearBias;

    // Separate Q/K/V projections. With GQA, Q is [numHeads*headDim, hidden] and K/V are
    // [numKvHeads*headDim, hidden] — different first-dim sizes.
    private Tensor? _toQWeight;
    private Tensor? _toKWeight;
    private Tensor? _toVWeight;

    // Output projection: Linear(numHeads*headDim → hidden), no bias.
    private Tensor? _toOutWeight;

    // norm2: post-attention RMSNorm scale [hidden].
    private Tensor? _norm2Weight;

    // ffn_norm1 / ffn_norm2: post-FFN-input + post-FFN RMSNorm scales [hidden].
    private Tensor? _ffnNorm1Weight;
    private Tensor? _ffnNorm2Weight;

    // SwiGLU FFN: linear_1 (gate), linear_2 (output projection ffnDim->hidden), linear_3 (up). No biases.
    // Note Lumina 2.0 uses LuminaFeedForward = w2(silu(w1(x)) * w3(x)) — same math as Z-Image's
    // SwiGluFfn, just different weight names.
    private Tensor? _ffWeight1;
    private Tensor? _ffWeight2;
    private Tensor? _ffWeight3;

    public Lumina2Block(int hiddenSize, int numHeads, int numKvHeads, int headDim, int ffnDim, int adaLNEmbedDim, float eps = 1e-5f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _qDim = numHeads * headDim;
        _kvDim = numKvHeads * headDim;
        _ffnDim = ffnDim;
        _adaLNEmbedDim = adaLNEmbedDim;
        _eps = eps;

        _normQ = new QkNorm(_headDim, eps);
        _normK = new QkNorm(_headDim, eps);
    }

    /// <summary>Loads weights using Lumina 2.0's diffusers naming. <paramref name="prefix"/> is e.g. "noise_refiner.0" or "layers.5".</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1NormWeight = TensorCasts.LoadF32(weights, $"{prefix}.norm1.norm.weight");
        _norm1LinearWeight = weights[$"{prefix}.norm1.linear.weight"];
        weights.TryGetValue($"{prefix}.norm1.linear.bias", out _norm1LinearBias);

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);

        _norm2Weight = TensorCasts.LoadF32(weights, $"{prefix}.norm2.weight");
        _ffnNorm1Weight = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm1.weight");
        _ffnNorm2Weight = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm2.weight");

        _ffWeight1 = weights[$"{prefix}.feed_forward.linear_1.weight"];
        _ffWeight2 = weights[$"{prefix}.feed_forward.linear_2.weight"];
        _ffWeight3 = weights[$"{prefix}.feed_forward.linear_3.weight"];
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_norm1NormWeight is not null) yield return _norm1NormWeight;
        if (_norm1LinearWeight is not null) yield return _norm1LinearWeight;
        if (_norm1LinearBias is not null) yield return _norm1LinearBias;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toOutWeight is not null) yield return _toOutWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_norm2Weight is not null) yield return _norm2Weight;
        if (_ffnNorm1Weight is not null) yield return _ffnNorm1Weight;
        if (_ffnNorm2Weight is not null) yield return _ffnNorm2Weight;
        if (_ffWeight1 is not null) yield return _ffWeight1;
        if (_ffWeight2 is not null) yield return _ffWeight2;
        if (_ffWeight3 is not null) yield return _ffWeight3;
    }

    /// <summary>Forward pass with timestep modulation and RoPE. Mirrors Lumina 2.0's <c>Lumina2TransformerBlock.forward</c>: <c>norm_h, gate_msa, scale_mlp, gate_mlp = norm1(h, t)</c>; <c>h += tanh(gate_msa) * norm2(attn(norm_h))</c>; <c>h += tanh(gate_mlp) * ffn_norm2(ffn(ffn_norm1(h) * (1 + scale_mlp)))</c>.
    /// B=1 (the pipeline case: CFG runs as two batch-1 passes) uses the GPU-resident path — every glue op
    /// (AdaLN split/tanh, scale, head permutes, GQA expand, rope, gated residual) is an IBackend op, so the
    /// activation never leaves the device (the old host loops were a full pipeline drain per op per block —
    /// the dominant cost of the 650s Lumina2 wall). Batched callers keep the host reference path.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor tEmb, ZImageRope? rope)
        => (int)x.Shape[0] == 1 ? ForwardDevice(backend, x, tEmb, rope) : ForwardHost(backend, x, tEmb, rope);

    private Tensor ForwardDevice(IBackend backend, Tensor x, Tensor tEmb, ZImageRope? rope)
    {
        int seqLen = (int)x.Shape[1];
        DType act = x.DType;
        TensorShape shape = new TensorShape(1, seqLen, _hiddenSize);
        TensorShape qHeads = new TensorShape(1, seqLen, _numHeads, _headDim);
        TensorShape kvHeads = new TensorShape(1, seqLen, _numKvHeads, _headDim);
        TensorShape qMhShape = new TensorShape(1, _numHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(1, _numKvHeads, seqLen, _headDim);

        // ── 1. norm1 RMSNorm + AdaLN(SiLU(temb)) → (scale_msa, tanh(gate_msa), scale_mlp, tanh(gate_mlp)) ──
        Tensor normed1 = new Tensor(shape, act);
        backend.RmsNorm(normed1, x, _norm1NormWeight!, _eps);
        Tensor[] mod = ComputeAdaLNDevice(backend, tEmb);
        Tensor normedScaled = ScaleDevice(backend, normed1, mod[0], shape, act);
        normed1.Dispose();

        // ── 2. Q/K/V ([1, S, H, D] — byte-identical to [1, S, H·D], so QK-norm + rope + permute need no reshape) ──
        Tensor q = new Tensor(qHeads, act);
        backend.Linear(q, normedScaled, _toQWeight!, null);
        Tensor k = new Tensor(kvHeads, act);
        backend.Linear(k, normedScaled, _toKWeight!, null);
        Tensor v = new Tensor(kvHeads, act);
        backend.Linear(v, normedScaled, _toVWeight!, null);
        normedScaled.Dispose();

        // ── 3. QK-norm (per-head RMSNorm over headDim) ──
        Tensor qN = new Tensor(qHeads, act);
        backend.RmsNorm(qN, q, _normQ.Weight, _normQ.Eps);
        q.Dispose();
        Tensor kN = new Tensor(kvHeads, act);
        backend.RmsNorm(kN, k, _normK.Weight, _normK.Eps);
        k.Dispose();

        // ── 4. RoPE on the pre-permute layout (per-tensor head counts — GQA-safe) ──
        rope?.ApplyGpuGqa(backend, qN, kN, _numHeads, _numKvHeads);

        // ── 5. Permute [1, S, H, D] → [1, H, S, D]; GQA-expand K/V to the Q head count ──
        Tensor qMh = new Tensor(qMhShape, act);
        backend.Permute0213(qMh, qN, seqLen, _numHeads, _headDim);
        qN.Dispose();
        Tensor kMh = new Tensor(kvMhShape, act);
        backend.Permute0213(kMh, kN, seqLen, _numKvHeads, _headDim);
        kN.Dispose();
        Tensor vMh = new Tensor(kvMhShape, act);
        backend.Permute0213(vMh, v, seqLen, _numKvHeads, _headDim);
        v.Dispose();

        Tensor kFull = kMh, vFull = vMh;
        if (_numKvHeads != _numHeads)
        {
            int groupSize = _numHeads / _numKvHeads;
            kFull = new Tensor(qMhShape, act);
            backend.RepeatKvHeads(kFull, kMh, _numKvHeads, groupSize);
            kMh.Dispose();
            vFull = new Tensor(qMhShape, act);
            backend.RepeatKvHeads(vFull, vMh, _numKvHeads, groupSize);
            vMh.Dispose();
        }

        // ── 6. SDPA (QK-norm bounds scores → F16-safe, engages the cuDNN fused flash path) ──
        Tensor attnOut = new Tensor(qMhShape, act);
        backend.ScaledDotProductAttention(attnOut, qMh, kFull, vFull, null, 1.0f / MathF.Sqrt(_headDim), allowF16: true);
        qMh.Dispose();
        kFull.Dispose();
        vFull.Dispose();

        // ── 7. Out-proj + sandwich norm2 + tanh-gated residual ──
        Tensor attnFlat = new Tensor(new TensorShape(1, seqLen, _qDim), act);
        backend.Permute0213(attnFlat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();
        Tensor projected = new Tensor(shape, act);
        backend.Linear(projected, attnFlat, _toOutWeight!, null);
        attnFlat.Dispose();
        Tensor postAttnNorm = new Tensor(shape, act);
        backend.RmsNorm(postAttnNorm, projected, _norm2Weight!, _eps);
        projected.Dispose();
        Tensor afterAttn = new Tensor(shape, act);
        backend.GatedResidualLastDim(afterAttn, x, postAttnNorm, mod[1]);
        postAttnNorm.Dispose();

        // ── 8. FFN: ffn_norm1 · (1 + scale_mlp) → SwiGLU → ffn_norm2 → tanh-gated residual ──
        Tensor ffNormed = new Tensor(shape, act);
        backend.RmsNorm(ffNormed, afterAttn, _ffnNorm1Weight!, _eps);
        Tensor ffModulated = ScaleDevice(backend, ffNormed, mod[2], shape, act);
        ffNormed.Dispose();
        Tensor ffOut = ForwardSwiGlu(backend, ffModulated, 1, seqLen);
        ffModulated.Dispose();
        Tensor postFfNorm = new Tensor(shape, act);
        backend.RmsNorm(postFfNorm, ffOut, _ffnNorm2Weight!, _eps);
        ffOut.Dispose();
        Tensor result = new Tensor(shape, act);
        backend.GatedResidualLastDim(result, afterAttn, postFfNorm, mod[3]);
        afterAttn.Dispose();
        postFfNorm.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();
        return result;
    }

    /// <summary>Device AdaLN: <c>norm1.linear(SiLU(temb))</c> → four <c>[1, hidden]</c> chunks (row-slices of
    /// the <c>[1, 4·hidden]</c> projection), tanh on the gate chunks (1, 3). The old host split read the
    /// device-produced projection via DataPointer — a full drain per block per step.</summary>
    private Tensor[] ComputeAdaLNDevice(IBackend backend, Tensor tEmb)
    {
        TensorShape tShape = new TensorShape(1, _adaLNEmbedDim);
        Tensor activated = new Tensor(tShape, DType.F32);
        backend.Silu(activated, tEmb);
        Tensor projected = new Tensor(new TensorShape(1, 4 * _hiddenSize), DType.F32);
        backend.Linear(projected, activated, _norm1LinearWeight!, _norm1LinearBias);
        activated.Dispose();

        Tensor[] results = new Tensor[4];
        for (int p = 0; p < 4; p++)
        {
            Tensor chunk = new Tensor(new TensorShape(1, _hiddenSize), DType.F32);
            backend.SliceRows(chunk, projected, p);   // [1, 4H] rows of H: offset = p·H
            if (p == 1 || p == 3)
            {
                Tensor gated = new Tensor(chunk.Shape, DType.F32);
                backend.Tanh(gated, chunk);
                chunk.Dispose();
                chunk = gated;
            }
            results[p] = chunk;
        }
        projected.Dispose();
        return results;
    }

    /// <summary>Device <c>x · (1 + scale)</c> with a per-channel <c>[1, hidden]</c> scale row.</summary>
    private static Tensor ScaleDevice(IBackend backend, Tensor input, Tensor scale, TensorShape shape, DType act)
    {
        Tensor scalePlus1 = new Tensor(scale.Shape, DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        Tensor output = new Tensor(shape, act);
        backend.AffineBroadcastLastDim(output, input, scalePlus1, null);
        scalePlus1.Dispose();
        return output;
    }

    private Tensor ForwardHost(IBackend backend, Tensor x, Tensor tEmb, ZImageRope? rope)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);

        // ── 1. norm1: RMSNorm(x) then modulation Linear on tEmb to produce 4 outputs. ──
        Tensor normed1 = new Tensor(shape, x.DType);
        backend.RmsNorm(normed1, x, _norm1NormWeight!, _eps);

        Tensor[] mod = ComputeAdaLNFromSiluTemb(backend, tEmb, batch);
        Tensor normedScaled = ApplyScale(normed1, mod[0], batch, seqLen, _hiddenSize);
        normed1.Dispose();

        // ── 2. Attention: separate Q/K/V projections, QK-norm, RoPE, GQA repeat-K/V, SDPA. ──
        TensorShape qShape = new TensorShape(batch, seqLen, _qDim);
        TensorShape kvShape = new TensorShape(batch, seqLen, _kvDim);

        Tensor q = new Tensor(qShape, x.DType);
        Tensor k = new Tensor(kvShape, x.DType);
        Tensor v = new Tensor(kvShape, x.DType);
        backend.Linear(q, normedScaled, _toQWeight!, null);
        backend.Linear(k, normedScaled, _toKWeight!, null);
        backend.Linear(v, normedScaled, _toVWeight!, null);
        normedScaled.Dispose();

        int qVecs = batch * seqLen * _numHeads;
        int kvVecs = batch * seqLen * _numKvHeads;
        Tensor qN = new Tensor(qShape, DType.F32);
        Tensor kN = new Tensor(kvShape, DType.F32);
        _normQ.Forward(qN, q, qVecs);
        _normK.Forward(kN, k, kvVecs);
        q.Dispose();
        k.Dispose();

        TensorShape qMhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(batch, _numKvHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(qMhShape, DType.F32);
        Tensor kMh = new Tensor(kvMhShape, DType.F32);
        Tensor vMh = new Tensor(kvMhShape, DType.F32);
        ReshapeToMultiHead(qMh, qN, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(kMh, kN, batch, seqLen, _numKvHeads, _headDim);
        ReshapeToMultiHead(vMh, v, batch, seqLen, _numKvHeads, _headDim);
        qN.Dispose();
        kN.Dispose();
        v.Dispose();

        // RoPE rotation per-tensor with each tensor's own head count. GQA-safe: Q has numHeads,
        // K has numKvHeads; the rotation is per-position (broadcast across heads) so calling
        // ForwardSingle on each with its own head count produces correct rotations.
        if (rope is not null)
        {
            rope.ForwardSingle(qMh, batch, _numHeads, seqLen);
            rope.ForwardSingle(kMh, batch, _numKvHeads, seqLen);
        }

        // GQA expand: K[B,Hkv,S,D] -> K_full[B,Hq,S,D], same for V.
        Tensor kFull, vFull;
        if (_numKvHeads != _numHeads)
        {
            int groupSize = _numHeads / _numKvHeads;
            kFull = new Tensor(qMhShape, DType.F32);
            vFull = new Tensor(qMhShape, DType.F32);
            RepeatKvHeads(kFull, kMh, batch, _numKvHeads, groupSize, seqLen, _headDim);
            RepeatKvHeads(vFull, vMh, batch, _numKvHeads, groupSize, seqLen, _headDim);
            kMh.Dispose();
            vMh.Dispose();
        }
        else
        {
            kFull = kMh;
            vFull = vMh;
        }

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kFull, vFull, null, scale);
        qMh.Dispose();
        kFull.Dispose();
        vFull.Dispose();

        TensorShape qFlatShape = new TensorShape(batch, seqLen, _qDim);
        Tensor attnFlat = new Tensor(qFlatShape, DType.F32);
        ReshapeFromMultiHead(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, x.DType);
        backend.Linear(projected, attnFlat, _toOutWeight!, null);
        attnFlat.Dispose();

        Tensor postAttnNorm = new Tensor(shape, x.DType);
        backend.RmsNorm(postAttnNorm, projected, _norm2Weight!, _eps);
        projected.Dispose();

        Tensor afterAttn = ApplyGatedResidual(x, postAttnNorm, mod[1], batch, seqLen, _hiddenSize);
        postAttnNorm.Dispose();

        // ── 3. FFN: ffn_norm1(afterAttn) * (1 + scale_mlp); SwiGLU; ffn_norm2; gate_mlp residual. ──
        Tensor ffNormed = new Tensor(shape, x.DType);
        backend.RmsNorm(ffNormed, afterAttn, _ffnNorm1Weight!, _eps);
        Tensor ffModulated = ApplyScale(ffNormed, mod[2], batch, seqLen, _hiddenSize);
        ffNormed.Dispose();

        Tensor ffOut = ForwardSwiGlu(backend, ffModulated, batch, seqLen);
        ffModulated.Dispose();

        Tensor postFfNorm = new Tensor(shape, x.DType);
        backend.RmsNorm(postFfNorm, ffOut, _ffnNorm2Weight!, _eps);
        ffOut.Dispose();

        Tensor result = ApplyGatedResidual(afterAttn, postFfNorm, mod[3], batch, seqLen, _hiddenSize);
        afterAttn.Dispose();
        postFfNorm.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return result;
    }

    /// <summary>Computes <c>norm1.linear(SiLU(t_emb))</c> and splits the [B, 4*hidden] result into 4 [B, hidden] chunks. Lumina 2.0 applies tanh() to gate_msa (chunk index 1) and gate_mlp (chunk index 3) before they're used as multipliers — without this gates blow up at init. Mirrors diffusers <c>LuminaRMSNormZero</c>.</summary>
    private Tensor[] ComputeAdaLNFromSiluTemb(IBackend backend, Tensor tEmb, int batch)
    {
        // SiLU(t_emb) — LuminaRMSNormZero composes Sequential(SiLU, Linear).
        TensorShape tShape = new TensorShape(batch, _adaLNEmbedDim);
        Tensor activated = new Tensor(tShape, tEmb.DType);
        backend.Silu(activated, tEmb);

        int outDim = 4 * _hiddenSize;
        TensorShape projShape = new TensorShape(batch, outDim);
        Tensor projected = new Tensor(projShape, tEmb.DType);
        backend.Linear(projected, activated, _norm1LinearWeight!, _norm1LinearBias);
        activated.Dispose();

        Tensor[] results = new Tensor[4];
        float* projPtr = (float*)projected.DataPointer;

        for (int p = 0; p < 4; p++)
        {
            TensorShape paramShape = new TensorShape(batch, _hiddenSize);
            Tensor param = new Tensor(paramShape, projected.DType);
            float* paramPtr = (float*)param.DataPointer;
            bool isGate = (p == 1 || p == 3);

            for (int b = 0; b < batch; b++)
            {
                int srcOffset = b * outDim + p * _hiddenSize;
                int dstOffset = b * _hiddenSize;
                if (isGate)
                {
                    for (int d = 0; d < _hiddenSize; d++)
                        paramPtr[dstOffset + d] = MathF.Tanh(projPtr[srcOffset + d]);
                }
                else
                {
                    Buffer.MemoryCopy(projPtr + srcOffset, paramPtr + dstOffset,
                        _hiddenSize * sizeof(float), _hiddenSize * sizeof(float));
                }
            }

            results[p] = param;
        }

        projected.Dispose();
        return results;
    }

    private static Tensor ApplyScale(Tensor input, Tensor scale, int batch, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(batch, seqLen, hiddenSize);
        Tensor output = new Tensor(shape, input.DType);

        float* inPtr = (float*)input.DataPointer;
        float* scalePtr = (float*)scale.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int seqOffset = (b * seqLen + s) * hiddenSize;
                int condOffset = b * hiddenSize;
                for (int d = 0; d < hiddenSize; d++)
                {
                    outPtr[seqOffset + d] = inPtr[seqOffset + d] * (1.0f + scalePtr[condOffset + d]);
                }
            }
        }
        return output;
    }

    private static Tensor ApplyGatedResidual(Tensor residual, Tensor value, Tensor gate, int batch, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(batch, seqLen, hiddenSize);
        Tensor output = new Tensor(shape, residual.DType);

        float* resPtr = (float*)residual.DataPointer;
        float* valPtr = (float*)value.DataPointer;
        float* gatePtr = (float*)gate.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int seqOffset = (b * seqLen + s) * hiddenSize;
                int condOffset = b * hiddenSize;
                for (int d = 0; d < hiddenSize; d++)
                {
                    outPtr[seqOffset + d] = resPtr[seqOffset + d] + gatePtr[condOffset + d] * valPtr[seqOffset + d];
                }
            }
        }
        return output;
    }

    /// <summary>SwiGLU: output = linear_2(SiLU(linear_1(x)) * linear_3(x)). Lumina-Image-2.0 names match Z-Image's w1/w2/w3 layout — only the key strings differ.</summary>
    private Tensor ForwardSwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnDim);

        Tensor gate = new Tensor(ffShape, input.DType);
        backend.Linear(gate, input, _ffWeight1!, null);
        Tensor gateActivated = new Tensor(ffShape, input.DType);
        backend.Silu(gateActivated, gate);
        gate.Dispose();

        Tensor up = new Tensor(ffShape, input.DType);
        backend.Linear(up, input, _ffWeight3!, null);

        Tensor gated = new Tensor(ffShape, input.DType);
        backend.Mul(gated, gateActivated, up);
        gateActivated.Dispose();
        up.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, gated, _ffWeight2!, null);
        gated.Dispose();

        return output;
    }

    private static void ReshapeToMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    private static void ReshapeFromMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    /// <summary>GQA expand: replicate each KV head <paramref name="groupSize"/> times across Q-head positions. <c>[B, Hkv, S, D] -&gt; [B, Hkv*groupSize=Hq, S, D]</c>.</summary>
    private static void RepeatKvHeads(Tensor output, Tensor input, int batch, int kvHeads, int groupSize, int seqLen, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long perHead = (long)seqLen * headDim;
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < kvHeads; h++)
            {
                long srcOff = ((long)b * kvHeads + h) * perHead;
                for (int g = 0; g < groupSize; g++)
                {
                    int qHead = h * groupSize + g;
                    long dstOff = ((long)b * (kvHeads * groupSize) + qHead) * perHead;
                    Buffer.MemoryCopy(inPtr + srcOff, outPtr + dstOff, perHead * sizeof(float), perHead * sizeof(float));
                }
            }
        }
    }
}
