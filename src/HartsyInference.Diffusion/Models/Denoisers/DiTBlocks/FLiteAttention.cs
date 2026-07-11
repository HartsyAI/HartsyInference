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

        // Device slice straight into [B, S, H, D] (hidden = H·D is already token-major head-then-dim).
        TensorShape bshd = new TensorShape(batch, seqLen, _numHeads, _headDim);
        Tensor q = new Tensor(bshd, DType.F32);
        Tensor k = new Tensor(bshd, DType.F32);
        Tensor v = new Tensor(bshd, DType.F32);
        backend.SliceLastDim(q, qkvFlat, 0);
        backend.SliceLastDim(k, qkvFlat, _hidden);
        backend.SliceLastDim(v, qkvFlat, 2 * _hidden);
        qkvFlat.Dispose();

        // Device RoPE PRE-permute (llama split-half convention; F-Lite's sign lives in the negated
        // sin table from FLiteRope.BuildDeviceTables).
        if (cosRope is not null && sinRope is not null)
            backend.ApplyRope(q, k, cosRope, sinRope);

        TensorShape bhsd = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor qHead = new Tensor(bhsd, DType.F32);
        Tensor kHead = new Tensor(bhsd, DType.F32);
        Tensor vHead = new Tensor(bhsd, DType.F32);
        backend.Permute0213(qHead, q, seqLen, _numHeads, _headDim);
        backend.Permute0213(kHead, k, seqLen, _numHeads, _headDim);
        backend.Permute0213(vHead, v, seqLen, _numHeads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor vMixed = MixVResidual(backend, vHead, vPrev);

        Tensor qNormed = ApplyHeadRmsNorm(backend, qHead, _qNormWeight);
        Tensor kNormed = ApplyHeadRmsNorm(backend, kHead, _kNormWeight);
        qHead.Dispose(); kHead.Dispose();

        Tensor attnOut = new Tensor(bhsd, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qNormed, kNormed, vMixed, mask: null, _scale, allowF16: true);
        qNormed.Dispose();
        kNormed.Dispose();

        Tensor flat = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Permute0213(flat, attnOut, _numHeads, seqLen, _headDim);
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

        Tensor qFlat = new Tensor(new TensorShape(batch, seqLen, _numHeads, _headDim), DType.F32);
        backend.Linear(qFlat, x, _qWeight, _qBias);

        Tensor kvFlat = new Tensor(new TensorShape(batch, ctxLen, 2 * _hidden), DType.F32);
        backend.Linear(kvFlat, context, _ctxKvWeight, _ctxKvBias);
        TensorShape ctxBshd = new TensorShape(batch, ctxLen, _numHeads, _headDim);
        Tensor k = new Tensor(ctxBshd, DType.F32);
        Tensor v = new Tensor(ctxBshd, DType.F32);
        backend.SliceLastDim(k, kvFlat, 0);
        backend.SliceLastDim(v, kvFlat, _hidden);
        kvFlat.Dispose();

        Tensor qHead = new Tensor(new TensorShape(batch, _numHeads, seqLen, _headDim), DType.F32);
        Tensor kHead = new Tensor(new TensorShape(batch, _numHeads, ctxLen, _headDim), DType.F32);
        Tensor vHead = new Tensor(new TensorShape(batch, _numHeads, ctxLen, _headDim), DType.F32);
        backend.Permute0213(qHead, qFlat, seqLen, _numHeads, _headDim);
        backend.Permute0213(kHead, k, ctxLen, _numHeads, _headDim);
        backend.Permute0213(vHead, v, ctxLen, _numHeads, _headDim);
        qFlat.Dispose(); k.Dispose(); v.Dispose();

        Tensor qN = ApplyHeadRmsNorm(backend, qHead, _qNormWeight);
        Tensor kN = ApplyHeadRmsNorm(backend, kHead, _kNormWeight);
        qHead.Dispose(); kHead.Dispose();
        qHead = qN; kHead = kN;

        Tensor attnOut = new Tensor(new TensorShape(batch, _numHeads, seqLen, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnOut, qHead, kHead, vHead, mask: null, _scale, allowF16: true);
        qHead.Dispose();
        kHead.Dispose();
        vHead.Dispose();

        Tensor flat = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Permute0213(flat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        Tensor output = new Tensor(new TensorShape(batch, seqLen, _hidden), DType.F32);
        backend.Linear(output, flat, _projWeight!, _projBias);
        flat.Dispose();
        return output;
    }

    private Tensor MixVResidual(IBackend backend, Tensor vCur, Tensor? vPrev)
    {
        if (!_residualV || vPrev is null || _lambdaParam is null) return vCur;

        // Read the scalar once and cache. lambda_param ships BF16 — reading its bytes as float was
        // garbage (the same reinterpretation bug as register_tokens; poisoned V in every block >= 1).
        if (_lambdaCached is null)
        {
            if (_lambdaParam.DType == DType.F32)
            {
                _lambdaCached = ((float*)_lambdaParam.DataPointer)[0];
            }
            else
            {
                using Tensor lamF32 = _lambdaParam.CastTo(DType.F32);
                _lambdaCached = ((float*)lamF32.DataPointer)[0];
            }
        }
        float lambda = _lambdaCached.Value;
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

    /// <summary>Per-head RMSNorm over headDim on the device. <paramref name="scale"/> null → all-ones (the public 10B ships affine-free QK norms).</summary>
    private Tensor ApplyHeadRmsNorm(IBackend backend, Tensor x, Tensor? scale)
    {
        Tensor result = new Tensor(x.Shape, DType.F32);
        backend.RmsNorm(result, x, scale ?? OnesHeadDim(backend), 1e-6f);
        return result;
    }

    private float? _lambdaCached;
    private Tensor? _onesHeadDim;
    private Tensor OnesHeadDim(IBackend backend)
    {
        if (_onesHeadDim is null)
        {
            _onesHeadDim = new Tensor(new TensorShape(_headDim), DType.F32);
            backend.Fill(_onesHeadDim, 1.0f);
        }
        return _onesHeadDim;
    }
}
