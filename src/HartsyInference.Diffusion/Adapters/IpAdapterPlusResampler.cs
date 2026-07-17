using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>The Perceiver-style resampler used by IP-Adapter Plus / Plus-Face / Full-Face. A small transformer with N learnable query embeddings (latents) that cross-attend over the CLIP-Vision penultimate hidden states (concatenated with the latents themselves for K/V), distilled through <c>depth</c> alternating attention + feed-forward layers, and finally projected from the resampler's working dim to the base UNet's cross-attention dim.
///
/// <para>Canonical configs (verified against the released checkpoint tensor shapes): SDXL Plus is <c>depth=4, num_queries=16, dim=1280, dim_head=64, heads=20 (inner_dim=1280), embedding_dim=1280, output_dim=2048, ff_mult=4</c>; SD1.5 Plus is the same shape at <c>dim=768, heads=12 (inner_dim=768), output_dim=768</c>. In both, <c>to_q</c> is square (<c>dim == inner_dim</c>) and the FFN expands <c>dim → 4·dim → dim</c>.</para>
///
/// <para>Weight key layout matches tencent-ailab / diffusers IP-Adapter checkpoints: <c>image_proj.latents</c>, <c>image_proj.proj_in.{weight,bias}</c>, per-layer <c>image_proj.layers.{i}.0.{norm1,norm2,to_q,to_kv,to_out}.*</c> + <c>image_proj.layers.{i}.1.net.{0,1,3}.*</c>, and <c>image_proj.proj_out.{weight,bias}</c> + <c>image_proj.norm_out.{weight,bias}</c>.</para></summary>
public sealed unsafe class IpAdapterPlusResampler : IIpAdapterImageProjection
{
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _innerDim; // numHeads * headDim
    private readonly int _numTokens;
    private readonly int _outputDim;
    private readonly int _depth;
    private readonly int _ffMultiplier;

    public int NumTokens => _numTokens;

    private Tensor? _latents;        // [1, numTokens, hiddenDim]
    private Tensor? _projInWeight;   // [hiddenDim, embeddingDim]
    private Tensor? _projInBias;     // [hiddenDim]
    private Tensor? _projOutWeight;  // [outputDim, hiddenDim]
    private Tensor? _projOutBias;    // [outputDim]
    private Tensor? _normOutWeight;  // [outputDim]
    private Tensor? _normOutBias;    // [outputDim]
    private readonly ResamplerLayer[] _layers;

    public IpAdapterPlusResampler(
        int embeddingDim,
        int hiddenDim,
        int numHeads,
        int headDim,
        int numTokens,
        int outputDim,
        int depth,
        int ffMultiplier = 4)
    {
        _embeddingDim = embeddingDim;
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _headDim = headDim;
        _innerDim = numHeads * headDim;
        _numTokens = numTokens;
        _outputDim = outputDim;
        _depth = depth;
        _ffMultiplier = ffMultiplier;
        _layers = new ResamplerLayer[depth];
        for (int i = 0; i < depth; i++)
        {
            _layers[i] = new ResamplerLayer(_hiddenDim, _numHeads, _headDim, _innerDim, ffMultiplier);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "image_proj")
    {
        _latents = EnsureF32(weights[$"{prefix}.latents"]);
        _projInWeight = EnsureF32(weights[$"{prefix}.proj_in.weight"]);
        _projInBias = EnsureF32(weights[$"{prefix}.proj_in.bias"]);
        _projOutWeight = EnsureF32(weights[$"{prefix}.proj_out.weight"]);
        _projOutBias = EnsureF32(weights[$"{prefix}.proj_out.bias"]);
        _normOutWeight = EnsureF32(weights[$"{prefix}.norm_out.weight"]);
        _normOutBias = EnsureF32(weights[$"{prefix}.norm_out.bias"]);
        for (int i = 0; i < _depth; i++)
        {
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_latents is not null) yield return _latents;
        if (_projInWeight is not null) yield return _projInWeight;
        if (_projInBias is not null) yield return _projInBias;
        for (int i = 0; i < _depth; i++)
        {
            foreach (Tensor w in _layers[i].EnumerateWeights()) yield return w;
        }
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
    }

    /// <summary>Forward pass on penultimate CLIP-Vision hidden states <c>[B, seqLen=numPatches+1, embeddingDim]</c>. Returns image-prompt tokens <c>[B, numTokens, outputDim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor visionInput)
    {
        if (visionInput.Shape.Rank != 3 || visionInput.Shape[2] != _embeddingDim)
        {
            throw new ArgumentException(
                $"visionInput must be [B, seqLen, {_embeddingDim}]; got {visionInput.Shape}.", nameof(visionInput));
        }
        int batch = (int)visionInput.Shape[0];
        int seqLen = (int)visionInput.Shape[1];

        // 1. proj_in: [B, seqLen, embDim] → [B, seqLen, hiddenDim]
        Tensor x = IpaLinear.Apply(backend, visionInput, _projInWeight!, _projInBias, batch, seqLen, _embeddingDim, _hiddenDim);

        // 2. Initialize latents by broadcasting the learned [1, numTokens, hiddenDim] over the batch.
        Tensor latents = new Tensor(new TensorShape(batch, _numTokens, _hiddenDim), DType.F32);
        BroadcastLatents(latents, _latents!, batch, _numTokens, _hiddenDim);

        // 3. Run resampler layers.
        for (int i = 0; i < _depth; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, latents, batch, seqLen);
            latents.Dispose();
            latents = next;
        }
        x.Dispose();

        // 4. proj_out: [B, numTokens, hiddenDim] → [B, numTokens, outputDim]
        Tensor projected = IpaLinear.Apply(backend, latents, _projOutWeight!, _projOutBias, batch, _numTokens, _hiddenDim, _outputDim);
        latents.Dispose();

        // 5. norm_out (LayerNorm over outputDim).
        IpAdapterMath.LayerNormInPlace(projected, _normOutWeight!, _normOutBias!,
            rows: batch * _numTokens, dim: _outputDim, eps: 1e-5f);
        return projected;
    }

    /// <summary>Broadcasts the stored <c>[1, numTokens, hiddenDim]</c> latents to <c>[batch, numTokens, hiddenDim]</c> by repeating across the batch axis.</summary>
    private static void BroadcastLatents(Tensor output, Tensor latents, int batch, int numTokens, int hiddenDim)
    {
        float* sp = (float*)latents.DataPointer;
        float* dp = (float*)output.DataPointer;
        int perBatch = numTokens * hiddenDim;
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < perBatch; i++)
            {
                dp[b * perBatch + i] = sp[i];
            }
        }
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}

/// <summary>One resampler layer: PerceiverAttention(latents over [x, latents]) + residual; FeedForward(latents) + residual. PerceiverAttention is a cross-attention where Q comes from latents and K/V come from concatenating the input x with the latents themselves — letting the latents both gather information from the image patches and refine via self-attention in a single op.</summary>
internal sealed unsafe class ResamplerLayer
{
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _innerDim;
    private readonly int _ffInnerDim;

    private Tensor? _norm1Weight, _norm1Bias; // applied to x
    private Tensor? _norm2Weight, _norm2Bias; // applied to latents
    private Tensor? _toQWeight; // [innerDim, hiddenDim], no bias
    private Tensor? _toKvWeight; // [2*innerDim, hiddenDim], no bias
    private Tensor? _toOutWeight; // [hiddenDim, innerDim], no bias

    private Tensor? _ffNormWeight, _ffNormBias; // FFN's net.0 LayerNorm
    private Tensor? _ffLinear1Weight; // [ffInnerDim, hiddenDim], no bias
    private Tensor? _ffLinear2Weight; // [hiddenDim, ffInnerDim], no bias

    public ResamplerLayer(int hiddenDim, int numHeads, int headDim, int innerDim, int ffMultiplier)
    {
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _headDim = headDim;
        _innerDim = innerDim;
        _ffInnerDim = hiddenDim * ffMultiplier;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1Weight = EnsureF32(weights[$"{prefix}.0.norm1.weight"]);
        _norm1Bias = EnsureF32(weights[$"{prefix}.0.norm1.bias"]);
        _norm2Weight = EnsureF32(weights[$"{prefix}.0.norm2.weight"]);
        _norm2Bias = EnsureF32(weights[$"{prefix}.0.norm2.bias"]);
        _toQWeight = EnsureF32(weights[$"{prefix}.0.to_q.weight"]);
        _toKvWeight = EnsureF32(weights[$"{prefix}.0.to_kv.weight"]);
        _toOutWeight = EnsureF32(weights[$"{prefix}.0.to_out.weight"]);
        _ffNormWeight = EnsureF32(weights[$"{prefix}.1.0.weight"]);
        _ffNormBias = EnsureF32(weights[$"{prefix}.1.0.bias"]);
        _ffLinear1Weight = EnsureF32(weights[$"{prefix}.1.1.weight"]);
        _ffLinear2Weight = EnsureF32(weights[$"{prefix}.1.3.weight"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1Weight, _norm1Bias, _norm2Weight, _norm2Bias, _toQWeight, _toKvWeight, _toOutWeight, _ffNormWeight, _ffNormBias, _ffLinear1Weight, _ffLinear2Weight];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary>Forward pass. Applies attention residual then FFN residual to <paramref name="latents"/> and returns the new latents tensor (caller disposes the old one).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor latents, int batch, int seqLen)
    {
        // ── PerceiverAttention ──────────────────────────────────────────
        // Pre-norms
        TensorShape xShape = new TensorShape(batch, seqLen, _hiddenDim);
        TensorShape lShape = new TensorShape(batch, latents.Shape[1], _hiddenDim);
        int numLatents = (int)latents.Shape[1];

        Tensor xNormed = new Tensor(xShape, DType.F32);
        backend.LayerNorm(xNormed, x, _norm1Weight!, _norm1Bias!, 1e-5f);
        Tensor lNormed = new Tensor(lShape, DType.F32);
        backend.LayerNorm(lNormed, latents, _norm2Weight!, _norm2Bias!, 1e-5f);

        // Q from latents — [B, numLatents, innerDim]
        Tensor q = IpaLinear.Apply(backend, lNormed, _toQWeight!, null, batch, numLatents, _hiddenDim, _innerDim);

        // KV input = concat(xNormed, lNormed) along seq dim — [B, seqLen + numLatents, hiddenDim]
        int kvSeqLen = seqLen + numLatents;
        Tensor kvInput = ConcatSeqDim(xNormed, lNormed, batch, seqLen, numLatents, _hiddenDim);
        xNormed.Dispose();
        lNormed.Dispose();

        // KV projection — [B, kvSeqLen, 2*innerDim], split into K and V
        Tensor kv = IpaLinear.Apply(backend, kvInput, _toKvWeight!, null, batch, kvSeqLen, _hiddenDim, 2 * _innerDim);
        kvInput.Dispose();
        Tensor k = SplitFirstHalf(kv, batch, kvSeqLen, _innerDim);
        Tensor v = SplitSecondHalf(kv, batch, kvSeqLen, _innerDim);
        kv.Dispose();

        // Multi-head attention. Reshape Q [B, numLatents, innerDim] → [B, heads, numLatents, headDim].
        Tensor qMh = ToMultiHead(q, batch, numLatents, _numHeads, _headDim);
        q.Dispose();
        Tensor kMh = ToMultiHead(k, batch, kvSeqLen, _numHeads, _headDim);
        k.Dispose();
        Tensor vMh = ToMultiHead(v, batch, kvSeqLen, _numHeads, _headDim);
        v.Dispose();

        TensorShape attnOutShape = new TensorShape(batch, _numHeads, numLatents, _headDim);
        Tensor attnOut = new Tensor(attnOutShape, DType.F32);
        float scale = 1.0f / MathF.Sqrt(_headDim);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        // Reshape back: [B, heads, numLatents, headDim] → [B, numLatents, innerDim]
        Tensor merged = FromMultiHead(attnOut, batch, numLatents, _numHeads, _headDim);
        attnOut.Dispose();

        // to_out projection — [B, numLatents, hiddenDim]
        Tensor attnProj = IpaLinear.Apply(backend, merged, _toOutWeight!, null, batch, numLatents, _innerDim, _hiddenDim);
        merged.Dispose();

        // Residual: latents = latents + attn_out
        Tensor afterAttn = new Tensor(lShape, DType.F32);
        backend.Add(afterAttn, latents, attnProj);
        attnProj.Dispose();

        // ── FeedForward ─────────────────────────────────────────────────
        Tensor ffNormed = new Tensor(lShape, DType.F32);
        backend.LayerNorm(ffNormed, afterAttn, _ffNormWeight!, _ffNormBias!, 1e-5f);
        Tensor fc1 = IpaLinear.Apply(backend, ffNormed, _ffLinear1Weight!, null, batch, numLatents, _hiddenDim, _ffInnerDim);
        ffNormed.Dispose();
        TensorShape fc1Shape = new TensorShape(batch, numLatents, _ffInnerDim);
        Tensor activated = new Tensor(fc1Shape, DType.F32);
        // The reference FeedForward uses nn.GELU() (exact erf form) — the tanh approximation's ~3e-3
        // pointwise gap compounds over the 4 residual layers and breaks 0.9999-corr parity.
        backend.GeluErf(activated, fc1);
        fc1.Dispose();
        Tensor fc2 = IpaLinear.Apply(backend, activated, _ffLinear2Weight!, null, batch, numLatents, _ffInnerDim, _hiddenDim);
        activated.Dispose();

        Tensor result = new Tensor(lShape, DType.F32);
        backend.Add(result, afterAttn, fc2);
        afterAttn.Dispose();
        fc2.Dispose();
        return result;
    }

    /// <summary>Concatenates two <c>[B, sN, D]</c> + <c>[B, sM, D]</c> tensors along the seq dim → <c>[B, sN+sM, D]</c>.</summary>
    private static Tensor ConcatSeqDim(Tensor a, Tensor b, int batch, int sA, int sB, int dim)
    {
        Tensor output = new Tensor(new TensorShape(batch, sA + sB, dim), DType.F32);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float* op = (float*)output.DataPointer;
        int rowBytes = dim * sizeof(float);
        for (int batchIdx = 0; batchIdx < batch; batchIdx++)
        {
            int aRowsBytes = sA * rowBytes;
            int bRowsBytes = sB * rowBytes;
            int outRowsBytes = (sA + sB) * rowBytes;
            Buffer.MemoryCopy(ap + batchIdx * sA * dim, op + batchIdx * (sA + sB) * dim, aRowsBytes, aRowsBytes);
            Buffer.MemoryCopy(bp + batchIdx * sB * dim, op + batchIdx * (sA + sB) * dim + sA * dim, bRowsBytes, bRowsBytes);
        }
        return output;
    }

    /// <summary>Split a <c>[B, S, 2D]</c> tensor into the first <c>[B, S, D]</c> half (K). Second half goes to <see cref="SplitSecondHalf"/> (V).</summary>
    private static Tensor SplitFirstHalf(Tensor kv, int batch, int seqLen, int innerDim)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, innerDim), DType.F32);
        float* sp = (float*)kv.DataPointer;
        float* dp = (float*)output.DataPointer;
        int twoD = 2 * innerDim;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int srcOff = (b * seqLen + s) * twoD;
                int dstOff = (b * seqLen + s) * innerDim;
                for (int d = 0; d < innerDim; d++)
                {
                    dp[dstOff + d] = sp[srcOff + d];
                }
            }
        }
        return output;
    }

    private static Tensor SplitSecondHalf(Tensor kv, int batch, int seqLen, int innerDim)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, innerDim), DType.F32);
        float* sp = (float*)kv.DataPointer;
        float* dp = (float*)output.DataPointer;
        int twoD = 2 * innerDim;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int srcOff = (b * seqLen + s) * twoD + innerDim;
                int dstOff = (b * seqLen + s) * innerDim;
                for (int d = 0; d < innerDim; d++)
                {
                    dp[dstOff + d] = sp[srcOff + d];
                }
            }
        }
        return output;
    }

    /// <summary>Reshape <c>[B, S, H*D]</c> → <c>[B, H, S, D]</c> for SDPA.</summary>
    private static Tensor ToMultiHead(Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        Tensor output = new Tensor(new TensorShape(batch, numHeads, seqLen, headDim), DType.F32);
        float* sp = (float*)input.DataPointer;
        float* dp = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int sOff = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int dOff = ((b * numHeads + h) * seqLen + s) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        dp[dOff + d] = sp[sOff + d];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Reshape <c>[B, H, S, D]</c> → <c>[B, S, H*D]</c> back from SDPA output.</summary>
    private static Tensor FromMultiHead(Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, numHeads * headDim), DType.F32);
        float* sp = (float*)input.DataPointer;
        float* dp = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int sOff = ((b * numHeads + h) * seqLen + s) * headDim;
                    int dOff = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        dp[dOff + d] = sp[sOff + d];
                    }
                }
            }
        }
        return output;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}

/// <summary>Linear projection helper for the IPA components. Mirrors <c>ClipTransformerLayer.ProjectLinear</c>'s shape contract: input <c>[B, S, inDim]</c>, weight <c>[outDim, inDim]</c> (PyTorch standard), optional bias <c>[outDim]</c>; output <c>[B, S, outDim]</c>. Internally transposes weight to <c>[inDim, outDim]</c> for <see cref="IBackend.BatchedMatMul"/>.</summary>
internal static unsafe class IpaLinear
{
    public static Tensor Apply(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int batch, int seqLen, int inDim, int outDim)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, outDim), DType.F32);
        Tensor weightT = new Tensor(new TensorShape(inDim, outDim), DType.F32);
        TransposeMatrix(weight, weightT, outDim, inDim);
        backend.BatchedMatMul(output, input, weightT);
        weightT.Dispose();
        if (bias is not null)
        {
            AddBiasBroadcast(output, bias, batch, seqLen, outDim);
        }
        return output;
    }

    private static void TransposeMatrix(Tensor src, Tensor dst, int rows, int cols)
    {
        float* srcPtr = (float*)src.DataPointer;
        float* dstPtr = (float*)dst.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                dstPtr[c * rows + r] = srcPtr[r * cols + c];
            }
        }
    }

    private static void AddBiasBroadcast(Tensor output, Tensor bias, int batch, int seqLen, int outDim)
    {
        float* outPtr = (float*)output.DataPointer;
        float* biasPtr = (float*)bias.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * outDim;
                for (int d = 0; d < outDim; d++)
                {
                    outPtr[offset + d] += biasPtr[d];
                }
            }
        }
    }
}
