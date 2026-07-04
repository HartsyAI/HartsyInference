using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Z-Image transformer block (Lumina2/NextDiT). Used for both <c>noise_refiner</c> blocks and the 30 main <c>layers</c> — they're structurally identical and only differ in which tokens they're called on. Uses AdaLN with 4 outputs (scale_msa, gate_msa, scale_mlp, gate_mlp — scale + gate, no shifts) and a fused QKV projection. The SwarmUI single-file checkpoint stores QKV as one big <c>[3*hidden, hidden]</c> tensor and the output projection as <c>attention.out</c>; QK-norm is <c>q_norm</c>/<c>k_norm</c> (not <c>norm_q</c>/<c>norm_k</c>). All attention linears have NO bias.</summary>
public sealed unsafe class ZImageBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnDim;
    private readonly float _eps;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // AdaLN: Linear(adaLNEmbedDim → 4 * hidden). t_emb is fed raw — Z-Image's t_embedder applies SiLU
    // internally between its two Linears, but the final output is the second Linear's output (not silu'd).
    // Z-Image's adaLN_modulation is `Sequential(Linear)` — index 0 — so no SiLU here.
    private Tensor? _adaLNWeight;
    private Tensor? _adaLNBias;

    // Fused QKV is split into separate Q/K/V weights at load (GPU-residency rewrite): 3 Linears write
    // directly into [B, S, H, D] so the head-split needs no host memcopy. The scalar fp8 weight_scale is
    // per-tensor, so the three splits share it (see LoadSplitQkv).
    private Tensor? _toQWeight, _toKWeight, _toVWeight;

    // Output projection: Linear(hidden → hidden), no bias.
    private Tensor? _attnOutWeight;

    // RMSNorm scales
    private Tensor? _attnNorm1Weight, _attnNorm2Weight;
    private Tensor? _ffnNorm1Weight, _ffnNorm2Weight;

    // SwiGLU FFN: w1 (gate), w3 (linear), w2 (output). No biases.
    private Tensor? _w1Weight;
    private Tensor? _w2Weight;
    private Tensor? _w3Weight;

    public ZImageBlock(int hiddenSize, int numHeads, int ffnDim, float eps = 1e-5f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _ffnDim = ffnDim;
        _eps = eps;

        _normQ = new QkNorm(_headDim, eps);
        _normK = new QkNorm(_headDim, eps);
    }

    /// <summary>Loads weights using Z-Image's Lumina-style naming. <paramref name="prefix"/> is e.g. "noise_refiner.0" or "layers.5".</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _adaLNWeight = weights[$"{prefix}.adaLN_modulation.0.weight"];
        weights.TryGetValue($"{prefix}.adaLN_modulation.0.bias", out _adaLNBias);

        LoadSplitQkv(weights[$"{prefix}.attention.qkv.weight"]);
        _attnOutWeight = weights[$"{prefix}.attention.out.weight"];

        _normQ.LoadWeights(weights[$"{prefix}.attention.q_norm.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attention.k_norm.weight"]);

        // CudaBackend.RmsNorm reads weight as float* directly, so RMSNorm scales MUST be F32.
        // BF16-stored norms (e.g., from a BF16 or nvfp8-mixed checkpoint) would otherwise be
        // bit-reinterpreted as garbage F32. Cheap one-time cast (each tensor is just [hidden]).
        _attnNorm1Weight = LoadAsF32(weights, $"{prefix}.attention_norm1.weight");
        _attnNorm2Weight = LoadAsF32(weights, $"{prefix}.attention_norm2.weight");
        _ffnNorm1Weight = LoadAsF32(weights, $"{prefix}.ffn_norm1.weight");
        _ffnNorm2Weight = LoadAsF32(weights, $"{prefix}.ffn_norm2.weight");

        _w1Weight = weights[$"{prefix}.feed_forward.w1.weight"];
        _w2Weight = weights[$"{prefix}.feed_forward.w2.weight"];
        _w3Weight = weights[$"{prefix}.feed_forward.w3.weight"];
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_adaLNWeight is not null) yield return _adaLNWeight;
        if (_adaLNBias is not null) yield return _adaLNBias;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_attnOutWeight is not null) yield return _attnOutWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_attnNorm1Weight is not null) yield return _attnNorm1Weight;
        if (_attnNorm2Weight is not null) yield return _attnNorm2Weight;
        if (_ffnNorm1Weight is not null) yield return _ffnNorm1Weight;
        if (_ffnNorm2Weight is not null) yield return _ffnNorm2Weight;
        if (_w1Weight is not null) yield return _w1Weight;
        if (_w2Weight is not null) yield return _w2Weight;
        if (_w3Weight is not null) yield return _w3Weight;
    }

    /// <summary>Forward pass with optional RoPE.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="x">Token sequence [B, seqLen, hidden].</param>
    /// <param name="tEmb">Timestep embedding [B, adaLNEmbedDim] — already through t_embedder.mlp (Linear → SiLU → Linear), output is NOT SiLU'd.</param>
    /// <param name="rope">Multi-axis RoPE precomputed for this seqLen, or null to skip.</param>
    public Tensor Forward(IBackend backend, Tensor x, Tensor tEmb, ZImageRope? rope, Tensor? attnBias = null)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        TensorShape headsShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // GPU-residency rewrite (mirrors the verified ErnieImageBlock / FluxSingleStreamBlock ports): the scale-only
        // AdaLN affines, the two gated residuals, the QK-norm and the head-split/merge reshapes all run as IBackend
        // ops so the activation chain stays device-resident — no per-op DataPointer D2H sync barriers around the
        // attention/FFN GEMMs. ApplyScale → DiTUtils.Modulate (shift null = pure broadcast multiply x·(1+scale));
        // ApplyGatedResidual → GatedResidualLastDim; the ReshapeToMultiHead/FromMultiHead host loops → Permute0213
        // (Q/K/V declared directly as [B, S, H, D], byte-identical to [B, S, hidden]); the fused-QKV host split →
        // three Linears against the load-time-split Q/K/V weights. The ONLY ops left on the CPU are the tiny AdaLN
        // chunk ([1, 15360]) and ZImageRope.Forward — both contained host excursions the current block already runs.

        // ── AdaLN: Linear(t_emb) → split into 4 along last dim (tiny; host) ──
        Tensor[] mod = ComputeAdaLN(backend, tEmb, batch);

        // ── Attention sub-block ──
        Tensor norm1 = new Tensor(shape, DType.F32);
        backend.RmsNorm(norm1, x, _attnNorm1Weight!, _eps);
        Tensor modulated = DiTUtils.Modulate(backend, norm1, null, mod[0], shape); // x·(1+scale_msa), no shift
        norm1.Dispose();

        // Separate Q/K/V projections declared directly as [B, S, H, D] so RmsNorm normalizes over headDim and
        // Permute0213 runs with no explicit reshape.
        Tensor qHeads = new Tensor(headsShape, DType.F32);
        Tensor kHeads = new Tensor(headsShape, DType.F32);
        Tensor v = new Tensor(headsShape, DType.F32);
        backend.Linear(qHeads, modulated, _toQWeight!, null);
        backend.Linear(kHeads, modulated, _toKWeight!, null);
        backend.Linear(v, modulated, _toVWeight!, null);
        modulated.Dispose();

        Tensor qNormed = new Tensor(headsShape, DType.F32);
        Tensor kNormed = new Tensor(headsShape, DType.F32);
        backend.RmsNorm(qNormed, qHeads, _normQ.Weight, _normQ.Eps);
        backend.RmsNorm(kNormed, kHeads, _normK.Weight, _normK.Eps);
        qHeads.Dispose();
        kHeads.Dispose();

        // [B, S, H, D] → [B, H, S, D] for SDPA. RoPE runs on the [B, H, S, D] layout (host, unchanged).
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(qMh, qNormed, seqLen, _numHeads, _headDim);
        backend.Permute0213(kMh, kNormed, seqLen, _numHeads, _headDim);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        v.Dispose();

        rope?.Forward(qMh, kMh, batch, _numHeads, seqLen);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attnBias, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // [B, H, S, D] → [B, S, hidden]
        Tensor attnFlat = new Tensor(shape, DType.F32);
        backend.Permute0213(attnFlat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, DType.F32);
        backend.Linear(projected, attnFlat, _attnOutWeight!, null);
        attnFlat.Dispose();

        Tensor postAttnNorm = new Tensor(shape, DType.F32);
        backend.RmsNorm(postAttnNorm, projected, _attnNorm2Weight!, _eps);
        projected.Dispose();

        Tensor afterAttn = new Tensor(shape, DType.F32);
        backend.GatedResidualLastDim(afterAttn, x, postAttnNorm, mod[1]);   // x + gate_msa·attn_out
        postAttnNorm.Dispose();

        // ── FFN sub-block ──
        Tensor normF1 = new Tensor(shape, DType.F32);
        backend.RmsNorm(normF1, afterAttn, _ffnNorm1Weight!, _eps);
        Tensor modulatedF = DiTUtils.Modulate(backend, normF1, null, mod[2], shape); // x·(1+scale_mlp), no shift
        normF1.Dispose();

        Tensor ffnOut = ForwardSwiGlu(backend, modulatedF, batch, seqLen);
        modulatedF.Dispose();

        Tensor postFfnNorm = new Tensor(shape, DType.F32);
        backend.RmsNorm(postFfnNorm, ffnOut, _ffnNorm2Weight!, _eps);
        ffnOut.Dispose();

        Tensor result = new Tensor(shape, DType.F32);
        backend.GatedResidualLastDim(result, afterAttn, postFfnNorm, mod[3]);  // afterAttn + gate_mlp·ffn_out
        afterAttn.Dispose();
        postFfnNorm.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return result;
    }

    /// <summary>Splits the fused <c>attention.qkv.weight</c> <c>[3*hidden, hidden]</c> into separate contiguous Q/K/V
    /// weights <c>[hidden, hidden]</c> at load. Rows [0,H)=Q, [H,2H)=K, [2H,3H)=V (matches the old feature-dim split).
    /// The per-tensor scalar <see cref="Tensor.Fp8ScaleFactor"/> is shared by all three splits (mirrors
    /// <c>CheckpointConvertUtils.SplitInProjWeight</c>). Dtype-agnostic byte copy — works for fp8/F16/F32.</summary>
    private void LoadSplitQkv(Tensor qkv)
    {
        int h = _hiddenSize;
        if (qkv.Shape[0] != 3L * h || qkv.Shape[1] != h)
            throw new ArgumentException($"Expected fused QKV weight [{3 * h}, {h}], got [{qkv.Shape[0]}, {qkv.Shape[1]}].");

        long chunkBytes = (long)h * h * qkv.DType.SizeInBytes;
        TensorShape splitShape = new TensorShape(h, h);

        _toQWeight = new Tensor(splitShape, qkv.DType) { Fp8ScaleFactor = qkv.Fp8ScaleFactor };
        _toKWeight = new Tensor(splitShape, qkv.DType) { Fp8ScaleFactor = qkv.Fp8ScaleFactor };
        _toVWeight = new Tensor(splitShape, qkv.DType) { Fp8ScaleFactor = qkv.Fp8ScaleFactor };

        byte* src = (byte*)qkv.DataPointer;
        Buffer.MemoryCopy(src, (void*)_toQWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)_toKWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)_toVWeight.DataPointer, chunkBytes, chunkBytes);
    }

    /// <summary>Loads a norm weight from the dict, casting to F32 if not already (RmsNorm requires F32 weight pointer).</summary>
    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    private Tensor[] ComputeAdaLN(IBackend backend, Tensor tEmb, int batch)
    {
        int outDim = 4 * _hiddenSize;
        TensorShape projShape = new TensorShape(batch, outDim);
        Tensor projected = new Tensor(projShape, tEmb.DType);
        backend.Linear(projected, tEmb, _adaLNWeight!, _adaLNBias);

        // Diffusers Z-Image applies tanh() to gate_msa (idx 1) and gate_mlp (idx 3) before use.
        // Without this, gates blow up at init and the network produces noise. See transformer_z_image.py:233.
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

    /// <summary>SwiGLU FFN with no biases: output = w2(silu(w1(x)) * w3(x)).</summary>
    private Tensor ForwardSwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnDim);

        Tensor gate = new Tensor(ffShape, input.DType);
        backend.Linear(gate, input, _w1Weight!, null);
        Tensor gateActivated = new Tensor(ffShape, input.DType);
        backend.Silu(gateActivated, gate);
        gate.Dispose();

        Tensor linear = new Tensor(ffShape, input.DType);
        backend.Linear(linear, input, _w3Weight!, null);

        Tensor gated = new Tensor(ffShape, input.DType);
        backend.Mul(gated, gateActivated, linear);
        gateActivated.Dispose();
        linear.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, gated, _w2Weight!, null);
        gated.Dispose();

        return output;
    }

}
