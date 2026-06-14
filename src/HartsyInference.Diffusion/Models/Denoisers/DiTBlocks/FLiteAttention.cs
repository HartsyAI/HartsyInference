using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>F-Lite attention block — implements both self-attention (with optional V-residual + RoPE) and cross-attention (Q from x, KV from context). Routes through <see cref="IBackend"/> for all linear projections + SDPA. Multi-head layout is <c>[B, H, S, D]</c> (head-major) matching DiTUtils convention.
///
/// <para><b>Self-attn weight keys</b> (relative to block prefix):</para>
/// <list type="bullet">
/// <item><c>self_attn.qkv.weight</c> [3*hidden, hidden] (no bias if <see cref="FLiteConfig.TrainBiasAndRms"/> false)</item>
/// <item><c>self_attn.proj.weight</c> [hidden, hidden] (always no bias — F-Lite always has <c>bias=False</c> on proj)</item>
/// <item><c>self_attn.qk_norm.{query,key}_norm.weight</c> [head_dim] — only if TrainBiasAndRms</item>
/// <item><c>self_attn.lambda_param</c> [1] — only if <see cref="FLiteConfig.ResidualV"/></item>
/// </list>
///
/// <para><b>Cross-attn weight keys</b>:</para>
/// <list type="bullet">
/// <item><c>cross_attn.q.weight</c> [hidden, hidden]</item>
/// <item><c>cross_attn.context_kv.weight</c> [2*hidden, cross_attn_input_size]</item>
/// <item><c>cross_attn.proj.weight</c> [hidden, hidden]</item>
/// <item><c>cross_attn.qk_norm.{query,key}_norm.weight</c> — only if TrainBiasAndRms</item>
/// </list></summary>
public sealed unsafe class FLiteAttention
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _crossInputSize;
    private readonly bool _isSelfAttn;
    private readonly bool _residualV;
    private readonly bool _hasBiasAndRms;
    private readonly float _scale;

    private Tensor? _qkvWeight, _qkvBias;
    private Tensor? _qWeight, _qBias;
    private Tensor? _ctxKvWeight, _ctxKvBias;
    private Tensor? _projWeight, _projBias;
    private Tensor? _qNormWeight, _kNormWeight;
    private Tensor? _lambdaParam;

    /// <summary>Self-attention constructor. Allocates fused QKV linear keys.</summary>
    public static FLiteAttention SelfAttn(FLiteConfig config) => new(config, isSelf: true);

    /// <summary>Cross-attention constructor. Allocates separate Q + context_kv linear keys.</summary>
    public static FLiteAttention CrossAttn(FLiteConfig config) => new(config, isSelf: false);

    private FLiteAttention(FLiteConfig config, bool isSelf)
    {
        _hidden = config.HiddenSize;
        _numHeads = config.NumHeads;
        _headDim = config.HeadDim;
        _crossInputSize = config.CrossAttnInputSize;
        _isSelfAttn = isSelf;
        _residualV = isSelf && config.ResidualV;
        _hasBiasAndRms = config.TrainBiasAndRms;
        _scale = 1.0f / MathF.Sqrt(_headDim);
    }

    /// <summary>Loads weights from a dict using <paramref name="prefix"/> (e.g. "blocks.0.self_attn"). Optional weights (qk_norm, lambda, biases) are silently skipped when not present.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        if (_isSelfAttn)
        {
            _qkvWeight = weights[$"{prefix}.qkv.weight"];
            weights.TryGetValue($"{prefix}.qkv.bias", out _qkvBias);
            if (_residualV && weights.TryGetValue($"{prefix}.lambda_param", out Tensor? lp))
                _lambdaParam = lp;
        }
        else
        {
            _qWeight = weights[$"{prefix}.q.weight"];
            weights.TryGetValue($"{prefix}.q.bias", out _qBias);
            _ctxKvWeight = weights[$"{prefix}.context_kv.weight"];
            weights.TryGetValue($"{prefix}.context_kv.bias", out _ctxKvBias);
        }

        _projWeight = weights[$"{prefix}.proj.weight"];
        weights.TryGetValue($"{prefix}.proj.bias", out _projBias);

        if (_hasBiasAndRms)
        {
            weights.TryGetValue($"{prefix}.qk_norm.query_norm.weight", out _qNormWeight);
            weights.TryGetValue($"{prefix}.qk_norm.key_norm.weight", out _kNormWeight);
        }
    }

    /// <summary>Yields all loaded weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_qkvWeight is not null) yield return _qkvWeight;
        if (_qkvBias is not null) yield return _qkvBias;
        if (_qWeight is not null) yield return _qWeight;
        if (_qBias is not null) yield return _qBias;
        if (_ctxKvWeight is not null) yield return _ctxKvWeight;
        if (_ctxKvBias is not null) yield return _ctxKvBias;
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
        if (_qNormWeight is not null) yield return _qNormWeight;
        if (_kNormWeight is not null) yield return _kNormWeight;
        if (_lambdaParam is not null) yield return _lambdaParam;
    }

    /// <summary>Self-attention forward. <paramref name="x"/> shape <c>[B, S, hidden]</c>. Optionally applies RoPE rotation to Q and K, optionally mixes <paramref name="vPrev"/> into V via the learned lambda. Returns (output, V) — caller captures V from block 0 and passes it as <paramref name="vPrev"/> to subsequent blocks.</summary>
    public (Tensor output, Tensor v) ForwardSelf(IBackend backend, Tensor x, Tensor? vPrev, Tensor? cosRope, Tensor? sinRope)
    {
        if (!_isSelfAttn) throw new InvalidOperationException("ForwardSelf called on a cross-attention instance.");
        if (_qkvWeight is null) throw new InvalidOperationException("LoadWeights has not been called.");

        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];

        Tensor qkvFlat = new Tensor(new TensorShape(batch, seqLen, 3 * _hidden), DType.F32);
        backend.Linear(qkvFlat, x, _qkvWeight, _qkvBias);

        Tensor q = SliceLastDim(qkvFlat, 0, _hidden, batch, seqLen);
        Tensor k = SliceLastDim(qkvFlat, 1, _hidden, batch, seqLen);
        Tensor v = SliceLastDim(qkvFlat, 2, _hidden, batch, seqLen);
        qkvFlat.Dispose();

        Tensor qHead = DiTUtils.ReshapeToMultiHead(q, batch, seqLen, _numHeads, _headDim);
        q.Dispose();
        Tensor kHead = DiTUtils.ReshapeToMultiHead(k, batch, seqLen, _numHeads, _headDim);
        k.Dispose();
        Tensor vHead = DiTUtils.ReshapeToMultiHead(v, batch, seqLen, _numHeads, _headDim);
        v.Dispose();

        Tensor vMixed = MixVResidual(backend, vHead, vPrev);

        if (cosRope is not null && sinRope is not null)
        {
            FLiteRope rope = new FLiteRope(_headDim, ropeBase: 10000);
            rope.ApplyRotation(qHead, cosRope, sinRope, batch, _numHeads, seqLen);
            rope.ApplyRotation(kHead, cosRope, sinRope, batch, _numHeads, seqLen);
        }

        ApplyHeadRmsNorm(qHead, _qNormWeight, batch, _numHeads, seqLen);
        ApplyHeadRmsNorm(kHead, _kNormWeight, batch, _numHeads, seqLen);

        Tensor attnOut = new Tensor(new TensorShape(batch, _numHeads, seqLen, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnOut, qHead, kHead, vMixed, mask: null, _scale);
        qHead.Dispose();
        kHead.Dispose();

        Tensor flat = DiTUtils.ReshapeFromMultiHead(attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Linear(output, flat, _projWeight!, _projBias);
        flat.Dispose();

        return (output, vMixed);
    }

    /// <summary>Cross-attention forward. <paramref name="x"/> shape <c>[B, S_x, hidden]</c>; <paramref name="context"/> shape <c>[B, S_ctx, cross_input]</c>. No RoPE, no V-residual.</summary>
    public Tensor ForwardCross(IBackend backend, Tensor x, Tensor context)
    {
        if (_isSelfAttn) throw new InvalidOperationException("ForwardCross called on a self-attention instance.");
        if (_qWeight is null || _ctxKvWeight is null) throw new InvalidOperationException("LoadWeights has not been called.");

        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int ctxLen = (int)context.Shape[1];

        Tensor qFlat = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Linear(qFlat, x, _qWeight, _qBias);

        Tensor kvFlat = new Tensor(new TensorShape(batch, ctxLen, 2 * _hidden), DType.F32);
        backend.Linear(kvFlat, context, _ctxKvWeight, _ctxKvBias);
        Tensor k = SliceLastDim(kvFlat, 0, _hidden, batch, ctxLen);
        Tensor v = SliceLastDim(kvFlat, 1, _hidden, batch, ctxLen);
        kvFlat.Dispose();

        Tensor qHead = DiTUtils.ReshapeToMultiHead(qFlat, batch, seqLen, _numHeads, _headDim);
        qFlat.Dispose();
        Tensor kHead = DiTUtils.ReshapeToMultiHead(k, batch, ctxLen, _numHeads, _headDim);
        k.Dispose();
        Tensor vHead = DiTUtils.ReshapeToMultiHead(v, batch, ctxLen, _numHeads, _headDim);
        v.Dispose();

        ApplyHeadRmsNorm(qHead, _qNormWeight, batch, _numHeads, seqLen);
        ApplyHeadRmsNorm(kHead, _kNormWeight, batch, _numHeads, ctxLen);

        Tensor attnOut = new Tensor(new TensorShape(batch, _numHeads, seqLen, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnOut, qHead, kHead, vHead, mask: null, _scale);
        qHead.Dispose();
        kHead.Dispose();
        vHead.Dispose();

        Tensor flat = DiTUtils.ReshapeFromMultiHead(attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Linear(output, flat, _projWeight!, _projBias);
        flat.Dispose();
        return output;
    }

    private static Tensor SliceLastDim(Tensor source, int chunkIdx, int chunkSize, int batch, int seqLen)
    {
        Tensor slice = new Tensor(new TensorShape(batch, seqLen, chunkSize), DType.F32);
        float* srcPtr = (float*)source.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        long stridePerToken = source.Shape[2];
        long offset = (long)chunkIdx * chunkSize;
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                long srcRow = ((long)b * seqLen + t) * stridePerToken + offset;
                long dstRow = ((long)b * seqLen + t) * chunkSize;
                Buffer.MemoryCopy(srcPtr + srcRow, dstPtr + dstRow, chunkSize * sizeof(float), chunkSize * sizeof(float));
            }
        }
        return slice;
    }

    private Tensor MixVResidual(IBackend backend, Tensor vCur, Tensor? vPrev)
    {
        if (!_residualV || vPrev is null || _lambdaParam is null) return vCur;

        float lambda = ((float*)_lambdaParam.DataPointer)[0];
        float oneMinusLambda = 1.0f - lambda;

        Tensor scaledCur = new Tensor(vCur.Shape, DType.F32);
        backend.Scale(scaledCur, vCur, lambda);
        Tensor scaledPrev = new Tensor(vCur.Shape, DType.F32);
        backend.Scale(scaledPrev, vPrev, oneMinusLambda);
        Tensor mixed = new Tensor(vCur.Shape, DType.F32);
        backend.Add(mixed, scaledCur, scaledPrev);
        scaledCur.Dispose();
        scaledPrev.Dispose();
        vCur.Dispose();
        return mixed;
    }

    private static void ApplyHeadRmsNorm(Tensor x, Tensor? scale, int batch, int numHeads, int seqLen)
    {
        if (x.DType != DType.F32) throw new ArgumentException("FLiteAttention head-RMSNorm expects F32.");
        int headDim = (int)x.Shape[3];
        float* xPtr = (float*)x.DataPointer;
        float* scalePtr = scale is not null ? (float*)scale.DataPointer : null;
        const float Eps = 1e-6f;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    long baseIdx = (((long)b * numHeads + h) * seqLen + t) * headDim;
                    float sumSq = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        float v = xPtr[baseIdx + d];
                        sumSq += v * v;
                    }
                    float invRms = 1.0f / MathF.Sqrt(sumSq / headDim + Eps);
                    if (scalePtr is null)
                    {
                        for (int d = 0; d < headDim; d++)
                            xPtr[baseIdx + d] *= invRms;
                    }
                    else
                    {
                        for (int d = 0; d < headDim; d++)
                            xPtr[baseIdx + d] = xPtr[baseIdx + d] * invRms * scalePtr[d];
                    }
                }
            }
        }
    }
}
