using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

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

    // Fused QKV: Linear(hidden → 3 * hidden), no bias.
    private Tensor? _qkvWeight;

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

        _qkvWeight = weights[$"{prefix}.attention.qkv.weight"];
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
        if (_qkvWeight is not null) yield return _qkvWeight;
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
    public Tensor Forward(IBackend backend, Tensor x, Tensor tEmb, ZImageRope? rope)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);

        // ── AdaLN: Linear(t_emb) → split into 4 along last dim ──
        Tensor[] mod = ComputeAdaLN(backend, tEmb, batch);

        // ── Attention sub-block ──
        Tensor norm1 = new Tensor(shape, x.DType);
        backend.RmsNorm(norm1, x, _attnNorm1Weight!, _eps);
        Tensor modulated = ApplyScale(norm1, mod[0], batch, seqLen, _hiddenSize);
        norm1.Dispose();

        // Fused QKV: [B, S, 3*hidden]
        TensorShape qkvShape = new TensorShape(batch, seqLen, 3 * _hiddenSize);
        Tensor qkv = new Tensor(qkvShape, x.DType);
        backend.Linear(qkv, modulated, _qkvWeight!, null);
        modulated.Dispose();

        Tensor q = new Tensor(shape, x.DType);
        Tensor k = new Tensor(shape, x.DType);
        Tensor v = new Tensor(shape, x.DType);
        SplitQkv(qkv, q, k, v, batch, seqLen, _hiddenSize);
        qkv.Dispose();

        int totalVecs = batch * seqLen * _numHeads;
        Tensor qN = new Tensor(q.Shape, DType.F32);
        Tensor kN = new Tensor(k.Shape, DType.F32);
        _normQ.Forward(qN, q, totalVecs);
        _normK.Forward(kN, k, totalVecs);
        q.Dispose();
        k.Dispose();

        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        ReshapeToMultiHead(qMh, qN, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(kMh, kN, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(vMh, v, batch, seqLen, _numHeads, _headDim);
        qN.Dispose();
        kN.Dispose();
        v.Dispose();

        rope?.Forward(qMh, kMh, batch, _numHeads, seqLen);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        Tensor attnFlat = new Tensor(shape, DType.F32);
        ReshapeFromMultiHead(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, x.DType);
        backend.Linear(projected, attnFlat, _attnOutWeight!, null);
        attnFlat.Dispose();

        Tensor postAttnNorm = new Tensor(shape, x.DType);
        backend.RmsNorm(postAttnNorm, projected, _attnNorm2Weight!, _eps);
        projected.Dispose();

        Tensor afterAttn = ApplyGatedResidual(x, postAttnNorm, mod[1], batch, seqLen, _hiddenSize);
        postAttnNorm.Dispose();

        // ── FFN sub-block ──
        Tensor normF1 = new Tensor(shape, x.DType);
        backend.RmsNorm(normF1, afterAttn, _ffnNorm1Weight!, _eps);
        Tensor modulatedF = ApplyScale(normF1, mod[2], batch, seqLen, _hiddenSize);
        normF1.Dispose();

        Tensor ffnOut = ForwardSwiGlu(backend, modulatedF, batch, seqLen);
        modulatedF.Dispose();

        Tensor postFfnNorm = new Tensor(shape, x.DType);
        backend.RmsNorm(postFfnNorm, ffnOut, _ffnNorm2Weight!, _eps);
        ffnOut.Dispose();

        Tensor result = ApplyGatedResidual(afterAttn, postFfnNorm, mod[3], batch, seqLen, _hiddenSize);
        afterAttn.Dispose();
        postFfnNorm.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return result;
    }

    /// <summary>Loads a norm weight from the dict, casting to F32 if not already (RmsNorm requires F32 weight pointer).</summary>
    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    /// <summary>Splits a fused QKV tensor [B, S, 3H] into three [B, S, H] tensors. Layout: feature dim is [Q | K | V].</summary>
    private static void SplitQkv(Tensor qkv, Tensor q, Tensor k, Tensor v, int batch, int seqLen, int hidden)
    {
        float* srcPtr = (float*)qkv.DataPointer;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* vPtr = (float*)v.DataPointer;

        long bytesPerSlice = (long)hidden * sizeof(float);
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long srcBase = ((long)b * seqLen + s) * 3 * hidden;
                long dstBase = ((long)b * seqLen + s) * hidden;
                Buffer.MemoryCopy(srcPtr + srcBase, qPtr + dstBase, bytesPerSlice, bytesPerSlice);
                Buffer.MemoryCopy(srcPtr + srcBase + hidden, kPtr + dstBase, bytesPerSlice, bytesPerSlice);
                Buffer.MemoryCopy(srcPtr + srcBase + 2 * hidden, vPtr + dstBase, bytesPerSlice, bytesPerSlice);
            }
        }
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
}
