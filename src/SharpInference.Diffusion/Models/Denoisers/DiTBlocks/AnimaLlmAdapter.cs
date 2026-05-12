using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Anima-specific <c>llm_adapter</c> module — a 6-block transformer that refines Qwen-3 0.6B hidden states
/// before they reach the main DiT cross-attention. Geometry confirmed by direct inspection of the Anima checkpoint:
/// <list type="bullet">
///   <item>Per block (<c>blocks.{0..5}</c>): three RMSNorm pre-norms (<c>norm_self_attn</c>, <c>norm_cross_attn</c>,
///        <c>norm_mlp</c>) each <c>[1024]</c>; self+cross attention with <c>{q,k,v,o}_proj.weight [1024, 1024]</c>
///        (note <c>o_proj</c>, not <c>output_proj</c>) and per-head <c>{q,k}_norm.weight [64]</c> (16 heads × 64 = 1024);
///        MLP with <c>mlp.0 [4096, 1024]</c> + bias and <c>mlp.2 [1024, 4096]</c> + bias (GELU between).</item>
///   <item>Top-level: <c>embed.weight [32128, 1024]</c> (a codebook of T5-vocab size — semantic role TBD; not used in
///        the v1 forward pass), <c>norm.weight [1024]</c>, <c>out_proj.weight [1024, 1024]</c> + <c>bias [1024]</c>.</item>
/// </list>
///
/// <para><b>v1 forward strategy:</b> The adapter forward pass treats the Qwen-3 hidden states as the sole input stream:</para>
/// <code>
/// for block in 0..5:
///     x = x + self_attn(norm_self_attn(x))           # within-stream attention
///     x = x + cross_attn(norm_cross_attn(x), kv=x)   # second self-attention pass with separate weights
///     x = x + mlp(norm_mlp(x))                       # GELU FFN
/// return out_proj(norm(x))                           # final RMSNorm + linear → [B, T, 1024]
/// </code>
/// <para>The cross-attention's K/V come from the same refined hidden stream as Q. The <c>embed</c> codebook
/// is loaded for round-trip fidelity but not consumed at inference. If empirical results show structural
/// degradation versus the Comfy reference, revisit by indexing <c>embed</c> with token positions (mod 32128)
/// or by tokenizing the prompt with the T5 tokenizer to produce a parallel stream — see <c>AnimaConfig</c> remarks.</para>
///
/// No RoPE — the adapter checkpoint has no rotary tables (the Qwen-3 encoder already applies its own RoPE before
/// emitting hidden states, and the adapter operates on those already-positioned features).</summary>
public sealed unsafe class AnimaLlmAdapter : IDisposable
{
    private readonly AnimaLlmAdapterConfig _config;
    private readonly AnimaLlmAdapterBlock[] _blocks;
    private int _disposed;

    // Top-level weights.
    private Tensor? _embedWeight;       // [vocab=32128, hidden=1024] — loaded but unused in v1.
    private Tensor? _normWeight;        // [hidden] final RMSNorm scale.
    private Tensor? _outProjWeight;     // [hidden, hidden] final projection.
    private Tensor? _outProjBias;       // [hidden] final projection bias.

    public AnimaLlmAdapter(AnimaLlmAdapterConfig config)
    {
        _config = config;
        _blocks = new AnimaLlmAdapterBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _blocks[i] = new AnimaLlmAdapterBlock(
                config.HiddenSize, config.NumHeads, config.HeadDim, config.FfnHiddenSize,
                config.RmsNormEps, config.QkNormEps);
        }
    }

    public AnimaLlmAdapterConfig Config => _config;

    /// <summary>Loads weights from the converter-bucketed <c>LlmAdapter</c> dict (keys with <c>net.llm_adapter.</c>
    /// already stripped — e.g. <c>blocks.0.self_attn.q_proj.weight</c>, <c>norm.weight</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        weights.TryGetValue("embed.weight", out _embedWeight);
        _normWeight = LoadAsF32(weights, "norm.weight");
        _outProjWeight = weights["out_proj.weight"];
        weights.TryGetValue("out_proj.bias", out _outProjBias);

        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"blocks.{i}");
    }

    /// <summary>Enumerates every weight tensor for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        if (_normWeight is not null) yield return _normWeight;
        if (_outProjWeight is not null) yield return _outProjWeight;
        if (_outProjBias is not null) yield return _outProjBias;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass on Qwen-3 0.6B hidden states <c>[B, T, 1024]</c>. Returns refined features
    /// <c>[B, T, 1024]</c> consumed by the DiT's cross-attention K/V projections.</summary>
    public Tensor Forward(IBackend backend, Tensor qwenHidden)
    {
        if (qwenHidden.Shape.Rank != 3)
            throw new ArgumentException($"qwenHidden must be 3D [B, T, hidden], got {qwenHidden.Shape}.", nameof(qwenHidden));
        if ((int)qwenHidden.Shape[2] != _config.HiddenSize)
            throw new ArgumentException(
                $"qwenHidden last dim {qwenHidden.Shape[2]} != adapter hidden {_config.HiddenSize}.", nameof(qwenHidden));

        int batch = (int)qwenHidden.Shape[0];
        int seqLen = (int)qwenHidden.Shape[1];
        int hidden = _config.HiddenSize;
        TensorShape shape = new TensorShape(batch, seqLen, hidden);

        // Start from a copy of the input (in F32 — the blocks operate in F32).
        Tensor x = new Tensor(shape, DType.F32);
        if (qwenHidden.DType == DType.F32)
        {
            Buffer.MemoryCopy(qwenHidden.DataPointer, x.DataPointer,
                batch * seqLen * hidden * sizeof(float), batch * seqLen * hidden * sizeof(float));
        }
        else
        {
            using Tensor f32 = qwenHidden.CastTo(DType.F32);
            Buffer.MemoryCopy(f32.DataPointer, x.DataPointer,
                batch * seqLen * hidden * sizeof(float), batch * seqLen * hidden * sizeof(float));
        }

        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x);
            x.Dispose();
            x = next;
        }

        // ── Final RMSNorm + projection ──
        Tensor normed = new Tensor(shape, DType.F32);
        backend.RmsNorm(normed, x, _normWeight!, _config.RmsNormEps);
        x.Dispose();

        Tensor output = new Tensor(shape, DType.F32);
        backend.Linear(output, normed, _outProjWeight!, _outProjBias);
        normed.Dispose();
        return output;
    }

    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _embedWeight = null;
            _normWeight = null;
            _outProjWeight = null;
            _outProjBias = null;
        }
        GC.SuppressFinalize(this);
    }
}

/// <summary>One block of the <see cref="AnimaLlmAdapter"/>. Pre-norm pattern: each sub-block is
/// <c>x = x + sublayer(rmsnorm(x))</c>. Three sub-blocks: self-attn, cross-attn (K/V from the same stream
/// in v1), MLP (GELU). No RoPE — Qwen-3 hidden states already carry positional information.</summary>
public sealed unsafe class AnimaLlmAdapterBlock
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnHidden;
    private readonly float _eps;

    private readonly QkNorm _selfQNorm;
    private readonly QkNorm _selfKNorm;
    private readonly QkNorm _crossQNorm;
    private readonly QkNorm _crossKNorm;

    // RMSNorm pre-norms.
    private Tensor? _normSelfAttnWeight;
    private Tensor? _normCrossAttnWeight;
    private Tensor? _normMlpWeight;

    // Self-attention.
    private Tensor? _selfQ, _selfK, _selfV, _selfO;
    // Cross-attention.
    private Tensor? _crossQ, _crossK, _crossV, _crossO;

    // MLP (Linear → GELU → Linear, with biases).
    private Tensor? _mlp0Weight, _mlp0Bias;
    private Tensor? _mlp2Weight, _mlp2Bias;

    public AnimaLlmAdapterBlock(int hidden, int numHeads, int headDim, int ffnHidden, float eps, float qkEps)
    {
        if (hidden != numHeads * headDim)
            throw new ArgumentException(
                $"hidden {hidden} != numHeads {numHeads} × headDim {headDim} ({numHeads * headDim}).", nameof(hidden));
        _hidden = hidden;
        _numHeads = numHeads;
        _headDim = headDim;
        _ffnHidden = ffnHidden;
        _eps = eps;
        _selfQNorm = new QkNorm(headDim, qkEps);
        _selfKNorm = new QkNorm(headDim, qkEps);
        _crossQNorm = new QkNorm(headDim, qkEps);
        _crossKNorm = new QkNorm(headDim, qkEps);
    }

    /// <summary>Loads block weights. <paramref name="prefix"/> is e.g. <c>blocks.0</c> (relative to the
    /// already-stripped <c>llm_adapter</c> bucket).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _normSelfAttnWeight = LoadAsF32(weights, $"{prefix}.norm_self_attn.weight");
        _normCrossAttnWeight = LoadAsF32(weights, $"{prefix}.norm_cross_attn.weight");
        _normMlpWeight = LoadAsF32(weights, $"{prefix}.norm_mlp.weight");

        _selfQ = weights[$"{prefix}.self_attn.q_proj.weight"];
        _selfK = weights[$"{prefix}.self_attn.k_proj.weight"];
        _selfV = weights[$"{prefix}.self_attn.v_proj.weight"];
        _selfO = weights[$"{prefix}.self_attn.o_proj.weight"];
        _selfQNorm.LoadWeights(weights[$"{prefix}.self_attn.q_norm.weight"]);
        _selfKNorm.LoadWeights(weights[$"{prefix}.self_attn.k_norm.weight"]);

        _crossQ = weights[$"{prefix}.cross_attn.q_proj.weight"];
        _crossK = weights[$"{prefix}.cross_attn.k_proj.weight"];
        _crossV = weights[$"{prefix}.cross_attn.v_proj.weight"];
        _crossO = weights[$"{prefix}.cross_attn.o_proj.weight"];
        _crossQNorm.LoadWeights(weights[$"{prefix}.cross_attn.q_norm.weight"]);
        _crossKNorm.LoadWeights(weights[$"{prefix}.cross_attn.k_norm.weight"]);

        _mlp0Weight = weights[$"{prefix}.mlp.0.weight"];
        weights.TryGetValue($"{prefix}.mlp.0.bias", out _mlp0Bias);
        _mlp2Weight = weights[$"{prefix}.mlp.2.weight"];
        weights.TryGetValue($"{prefix}.mlp.2.bias", out _mlp2Bias);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normSelfAttnWeight is not null) yield return _normSelfAttnWeight;
        if (_normCrossAttnWeight is not null) yield return _normCrossAttnWeight;
        if (_normMlpWeight is not null) yield return _normMlpWeight;
        if (_selfQ is not null) yield return _selfQ;
        if (_selfK is not null) yield return _selfK;
        if (_selfV is not null) yield return _selfV;
        if (_selfO is not null) yield return _selfO;
        foreach (Tensor w in _selfQNorm.EnumerateWeights()) yield return w;
        foreach (Tensor w in _selfKNorm.EnumerateWeights()) yield return w;
        if (_crossQ is not null) yield return _crossQ;
        if (_crossK is not null) yield return _crossK;
        if (_crossV is not null) yield return _crossV;
        if (_crossO is not null) yield return _crossO;
        foreach (Tensor w in _crossQNorm.EnumerateWeights()) yield return w;
        foreach (Tensor w in _crossKNorm.EnumerateWeights()) yield return w;
        if (_mlp0Weight is not null) yield return _mlp0Weight;
        if (_mlp0Bias is not null) yield return _mlp0Bias;
        if (_mlp2Weight is not null) yield return _mlp2Weight;
        if (_mlp2Bias is not null) yield return _mlp2Bias;
    }

    /// <summary>Forward pass. <paramref name="x"/> is the running stream <c>[B, T, hidden]</c>.
    /// Returns the residual sum after self-attn, cross-attn, and MLP.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hidden);

        // ── 1. Self-attention sub-block: x = x + self_attn(rms(x)) ──
        Tensor preSelf = new Tensor(shape, DType.F32);
        backend.RmsNorm(preSelf, x, _normSelfAttnWeight!, _eps);
        Tensor selfOut = Attention(backend, preSelf, preSelf,
            _selfQ!, _selfK!, _selfV!, _selfO!, _selfQNorm, _selfKNorm,
            batch, seqLen, seqLen);
        preSelf.Dispose();
        Tensor afterSelf = new Tensor(shape, DType.F32);
        backend.Add(afterSelf, x, selfOut);
        selfOut.Dispose();

        // ── 2. Cross-attention sub-block (K/V = same stream in v1): x = x + cross_attn(rms(x), kv=rms(x)) ──
        Tensor preCross = new Tensor(shape, DType.F32);
        backend.RmsNorm(preCross, afterSelf, _normCrossAttnWeight!, _eps);
        // Q and K/V both derive from preCross — separate projections, same source tensor.
        Tensor crossOut = Attention(backend, preCross, preCross,
            _crossQ!, _crossK!, _crossV!, _crossO!, _crossQNorm, _crossKNorm,
            batch, seqLen, seqLen);
        preCross.Dispose();
        Tensor afterCross = new Tensor(shape, DType.F32);
        backend.Add(afterCross, afterSelf, crossOut);
        afterSelf.Dispose();
        crossOut.Dispose();

        // ── 3. MLP sub-block: x = x + mlp(rms(x)) ──
        Tensor preMlp = new Tensor(shape, DType.F32);
        backend.RmsNorm(preMlp, afterCross, _normMlpWeight!, _eps);
        Tensor mlpOut = Mlp(backend, preMlp, batch, seqLen);
        preMlp.Dispose();
        Tensor result = new Tensor(shape, DType.F32);
        backend.Add(result, afterCross, mlpOut);
        afterCross.Dispose();
        mlpOut.Dispose();

        return result;
    }

    /// <summary>Standard multi-head attention with QK-norm. Q comes from <paramref name="qSource"/>,
    /// K/V from <paramref name="kvSource"/>. Output is <c>[B, qLen, hidden]</c>.</summary>
    private Tensor Attention(IBackend backend, Tensor qSource, Tensor kvSource,
        Tensor qWeight, Tensor kWeight, Tensor vWeight, Tensor oWeight,
        QkNorm qNorm, QkNorm kNorm, int batch, int qLen, int kvLen)
    {
        TensorShape qShape = new TensorShape(batch, qLen, _hidden);
        TensorShape kvShape = new TensorShape(batch, kvLen, _hidden);

        Tensor q = new Tensor(qShape, DType.F32);
        Tensor k = new Tensor(kvShape, DType.F32);
        Tensor v = new Tensor(kvShape, DType.F32);
        backend.Linear(q, qSource, qWeight, null);
        backend.Linear(k, kvSource, kWeight, null);
        backend.Linear(v, kvSource, vWeight, null);

        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, qLen, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, kvLen, _numHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, kvLen, _numHeads, _headDim);
        q.Dispose();
        k.Dispose();
        v.Dispose();

        // Per-head QK-RMSNorm.
        Tensor qNormed = new Tensor(qMh.Shape, DType.F32);
        Tensor kNormed = new Tensor(kMh.Shape, DType.F32);
        qNorm.Forward(qNormed, qMh, batch * _numHeads * qLen);
        kNorm.Forward(kNormed, kMh, batch * _numHeads * kvLen);
        qMh.Dispose();
        kMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMh.Shape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qNormed, kNormed, vMh, null, scale);
        qNormed.Dispose();
        kNormed.Dispose();
        vMh.Dispose();

        Tensor flat = DiTUtils.ReshapeFromMultiHead(attnOut, batch, qLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor output = new Tensor(qShape, DType.F32);
        backend.Linear(output, flat, oWeight, null);
        flat.Dispose();
        return output;
    }

    /// <summary>Two-layer FFN with GELU activation and biases. <c>Linear(hidden, ffn) → GELU → Linear(ffn, hidden)</c>.</summary>
    private Tensor Mlp(IBackend backend, Tensor x, int batch, int seqLen)
    {
        TensorShape ffnShape = new TensorShape(batch, seqLen, _ffnHidden);
        TensorShape outShape = new TensorShape(batch, seqLen, _hidden);

        Tensor h = new Tensor(ffnShape, DType.F32);
        backend.Linear(h, x, _mlp0Weight!, _mlp0Bias);

        Tensor activated = new Tensor(ffnShape, DType.F32);
        backend.Gelu(activated, h);
        h.Dispose();

        Tensor result = new Tensor(outShape, DType.F32);
        backend.Linear(result, activated, _mlp2Weight!, _mlp2Bias);
        activated.Dispose();
        return result;
    }

    private static Tensor LoadAsF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor t = weights[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
