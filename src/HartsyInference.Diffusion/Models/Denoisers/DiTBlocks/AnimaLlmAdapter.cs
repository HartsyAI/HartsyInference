using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Anima's <c>net.llm_adapter</c> — a 6-block transformer that converts the prompt into the
/// 1024-dim conditioning the Cosmos-Predict2 DiT cross-attention expects. Source of truth:
/// <c>tdrussell/diffusion-pipe/models/llm_adapter.py</c>.
///
/// <para><b>Two input streams (this is the key bit v1 got wrong):</b></para>
/// <list type="bullet">
///   <item><c>target_input_ids</c>: a <b>T5-tokenized</b> version of the same prompt, indexed into the learned
///        <c>embed [32128, 1024]</c> table to produce <c>x</c> of shape <c>[B, S_t5, 1024]</c>. This is the
///        adapter's main running stream — what self-attn and the MLP operate on. Output sequence length
///        equals <c>S_t5</c>.</item>
///   <item><c>source_hidden_states</c>: the Qwen-3 0.6B last-layer hidden states <c>[B, S_qwen3, 1024]</c>.
///        Used ONLY as the K/V source of cross-attention.</item>
/// </list>
///
/// <para><b>Per block:</b> three pre-norm RMSNorms (<c>norm_self_attn</c>, <c>norm_cross_attn</c>, <c>norm_mlp</c>),
/// self-attn (Q,K,V all from x with RoPE on Q+K), cross-attn (Q from x with RoPE on target positions, K+V from
/// context with RoPE on source positions), MLP <c>Linear(1024,4096) -> GELU -> Linear(4096,1024)</c>.</para>
///
/// <para><b>Final:</b> <c>out_proj(x)</c> THEN <c>RMSNorm</c> (Python: <c>return self.norm(self.out_proj(x))</c>).
/// v1 had these reversed.</para>
///
/// <para><b>RoPE convention:</b> LLaMA-style "rotate-half, concat-duplicate". <c>theta = 10000</c>,
/// <c>head_dim = 64</c>. <c>cos/sin</c> tables are computed per stream from its own
/// <c>arange(seq_len)</c> position ids; cross-attn uses target positions for Q and source positions for K.</para>
///
/// <para>RoPE math: <c>rotate_half(x) = cat(-x[halfDim:], x[:halfDim])</c>;
/// <c>out = x * cos + rotate_half(x) * sin</c>, where <c>cos = [freq, freq]</c> (concat-duplicated, NOT
/// interleave-duplicated like ERNIE's). Implementation here applies the rotation in-place on <c>[B, H, S, D]</c>
/// layout (heads-first), since that's what <see cref="DiTUtils.ReshapeToMultiHead(Tensor, int, int, int, int)"/>
/// produces.</para></summary>
public sealed unsafe class AnimaLlmAdapter : IDisposable
{
    private readonly AnimaLlmAdapterConfig _config;
    private readonly AnimaLlmAdapterBlock[] _blocks;
    private int _disposed;

    // Top-level weights.
    private Tensor? _embedWeight;       // [vocab=32128, hidden=1024] — main stream input via T5 token IDs.
    private Tensor? _normWeight;        // [hidden] final RMSNorm scale (applied AFTER out_proj).
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
    /// already stripped — e.g. <c>blocks.0.self_attn.q_proj.weight</c>, <c>embed.weight</c>, <c>norm.weight</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _embedWeight = TensorCasts.LoadF32(weights, "embed.weight");
        _normWeight = TensorCasts.LoadF32(weights, "norm.weight");
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

    /// <summary>Forward pass.
    /// <para><paramref name="qwenHidden"/>: Qwen-3 last-layer hidden states <c>[1, S_qwen3, 1024]</c> — used as
    /// the K/V source of every block's cross-attention.</para>
    /// <para><paramref name="t5TokenIds"/>: T5 tokenization of the same prompt — indexed into <c>embed</c> to
    /// produce the main stream <c>x</c> <c>[1, S_t5, 1024]</c>.</para>
    /// <para>Output: refined features <c>[1, S_t5, 1024]</c> consumed by the DiT's cross-attention K/V. Note
    /// the output sequence length is the T5-token count, NOT the Qwen-3 token count.</para></summary>
    public Tensor Forward(IBackend backend, Tensor qwenHidden, int[] t5TokenIds)
    {
        if (qwenHidden.Shape.Rank != 3)
            throw new ArgumentException($"qwenHidden must be 3D [B, T, hidden], got {qwenHidden.Shape}.", nameof(qwenHidden));
        if ((int)qwenHidden.Shape[2] != _config.HiddenSize)
            throw new ArgumentException(
                $"qwenHidden last dim {qwenHidden.Shape[2]} != adapter hidden {_config.HiddenSize}.", nameof(qwenHidden));
        if (t5TokenIds.Length == 0)
            throw new ArgumentException("t5TokenIds must not be empty.", nameof(t5TokenIds));

        int batch = (int)qwenHidden.Shape[0];
        int sQwen = (int)qwenHidden.Shape[1];
        int sT5 = t5TokenIds.Length;
        int hidden = _config.HiddenSize;
        if (batch != 1) throw new ArgumentException("batch != 1 not supported in v2.", nameof(qwenHidden));

        // ── 1. Embed lookup: x = embed[t5_ids] ──
        TensorShape xShape = new TensorShape(batch, sT5, hidden);
        Tensor x = new Tensor(xShape, DType.F32);
        EmbedLookup(x, _embedWeight!, t5TokenIds, _config.CodebookVocab, hidden);
        AnimaLlmAdapterDebugDump.Dump("x_post_embed", x);

        // ── 2. Build RoPE cos/sin tables for both streams ──
        // Per the canonical Python, position_ids = arange(S) per stream. theta = 10000, head_dim = 64.
        (Tensor cosTgt, Tensor sinTgt) = BuildRopeTables(sT5, _config.HeadDim, theta: 10000.0f);
        (Tensor cosSrc, Tensor sinSrc) = BuildRopeTables(sQwen, _config.HeadDim, theta: 10000.0f);

        // ── 3. Make a working F32 copy of qwenHidden (context never changes across blocks) ──
        Tensor context;
        if (qwenHidden.DType == DType.F32)
        {
            context = new Tensor(qwenHidden.Shape, DType.F32);
            long bytes = batch * sQwen * hidden * sizeof(float);
            Buffer.MemoryCopy(qwenHidden.DataPointer, context.DataPointer, bytes, bytes);
        }
        else
        {
            context = qwenHidden.CastTo(DType.F32);
        }

        // ── 4. Run 6 blocks ──
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x, context, cosTgt, sinTgt, cosSrc, sinSrc);
            x.Dispose();
            x = next;
            AnimaLlmAdapterDebugDump.Dump($"block_{i}", x);
        }

        context.Dispose();
        cosTgt.Dispose(); sinTgt.Dispose();
        cosSrc.Dispose(); sinSrc.Dispose();

        // ── 5. Final: out_proj(x) THEN RMSNorm ──
        Tensor projected = new Tensor(xShape, DType.F32);
        backend.Linear(projected, x, _outProjWeight!, _outProjBias);
        x.Dispose();
        AnimaLlmAdapterDebugDump.Dump("post_out_proj", projected);

        Tensor output = new Tensor(xShape, DType.F32);
        backend.RmsNorm(output, projected, _normWeight!, _config.RmsNormEps);
        projected.Dispose();
        AnimaLlmAdapterDebugDump.Dump("final_output", output);
        return output;
    }

    /// <summary>Builds the LLaMA-style RoPE cos/sin tables. Output layout: each is <c>[seqLen, headDim]</c>
    /// with the "concat-duplicate" pattern <c>[f0, f1, ..., f_{halfDim-1}, f0, f1, ..., f_{halfDim-1}]</c>
    /// per position — i.e. positions <c>i</c> and <c>i+halfDim</c> share the same angle. Mirrors
    /// <c>diffusion-pipe/models/llm_adapter.py:RotaryEmbedding.forward</c>.</summary>
    private static (Tensor cos, Tensor sin) BuildRopeTables(int seqLen, int headDim, float theta)
    {
        int halfDim = headDim / 2;
        TensorShape shape = new TensorShape(seqLen, headDim);
        Tensor cos = new Tensor(shape, DType.F32);
        Tensor sin = new Tensor(shape, DType.F32);
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;

        // inv_freq[k] = 1 / theta^(2k / head_dim) for k in [0, halfDim).
        Span<double> invFreq = stackalloc double[halfDim];
        for (int k = 0; k < halfDim; k++)
            invFreq[k] = 1.0 / Math.Pow(theta, (2.0 * k) / headDim);

        for (int s = 0; s < seqLen; s++)
        {
            for (int k = 0; k < halfDim; k++)
            {
                double angle = s * invFreq[k];
                float c = (float)Math.Cos(angle);
                float si = (float)Math.Sin(angle);
                // Concat-duplicate: cos[s, k] = cos[s, k + halfDim] = cos(angle).
                cosPtr[s * headDim + k] = c;
                cosPtr[s * headDim + k + halfDim] = c;
                sinPtr[s * headDim + k] = si;
                sinPtr[s * headDim + k + halfDim] = si;
            }
        }
        return (cos, sin);
    }

    /// <summary>Indexes <paramref name="embedWeight"/> <c>[vocab, hidden]</c> by <paramref name="tokenIds"/> to
    /// produce <c>[1, seqLen, hidden]</c> in <paramref name="output"/>.</summary>
    private static void EmbedLookup(Tensor output, Tensor embedWeight, int[] tokenIds, int vocab, int hidden)
    {
        float* outPtr = (float*)output.DataPointer;
        float* embedPtr = (float*)embedWeight.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int id = tokenIds[s];
            if (id < 0 || id >= vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token id {id} at position {s} is out of range [0, {vocab}).");
            long src = (long)id * hidden;
            long dst = (long)s * hidden;
            Buffer.MemoryCopy(embedPtr + src, outPtr + dst, hidden * sizeof(float), hidden * sizeof(float));
        }
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

/// <summary>One block of the <see cref="AnimaLlmAdapter"/>. Pre-norm sub-block pattern with two attention
/// passes (self then cross) and a GELU MLP.</summary>
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

    private Tensor? _normSelfAttnWeight;
    private Tensor? _normCrossAttnWeight;
    private Tensor? _normMlpWeight;

    private Tensor? _selfQ, _selfK, _selfV, _selfO;
    private Tensor? _crossQ, _crossK, _crossV, _crossO;

    private Tensor? _mlp0Weight, _mlp0Bias;
    private Tensor? _mlp2Weight, _mlp2Bias;

    public AnimaLlmAdapterBlock(int hidden, int numHeads, int headDim, int ffnHidden, float eps, float qkEps)
    {
        if (hidden != numHeads * headDim)
            throw new ArgumentException(
                $"hidden {hidden} != numHeads {numHeads} × headDim {headDim} ({numHeads * headDim}).", nameof(hidden));
        if (headDim % 2 != 0)
            throw new ArgumentException($"headDim {headDim} must be even for RoPE.", nameof(headDim));
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

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _normSelfAttnWeight = TensorCasts.LoadF32(weights, $"{prefix}.norm_self_attn.weight");
        _normCrossAttnWeight = TensorCasts.LoadF32(weights, $"{prefix}.norm_cross_attn.weight");
        _normMlpWeight = TensorCasts.LoadF32(weights, $"{prefix}.norm_mlp.weight");

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

    /// <summary>Pre-norm self-attn → cross-attn → MLP, each a residual addition (no gating, unlike the outer DiT).</summary>
    /// <param name="x">Main stream <c>[1, S_t5, hidden]</c>. Self-attn Q,K,V and cross-attn Q all derive from this.</param>
    /// <param name="context">Qwen-3 hidden states <c>[1, S_qwen3, hidden]</c>. Cross-attn K,V derive from this. Caller owns disposal.</param>
    /// <param name="cosTgt">Self-attn / cross-attn-Q RoPE cos <c>[S_t5, headDim]</c>. Caller owns.</param>
    /// <param name="sinTgt">Self-attn / cross-attn-Q RoPE sin <c>[S_t5, headDim]</c>. Caller owns.</param>
    /// <param name="cosSrc">Cross-attn-K RoPE cos <c>[S_qwen3, headDim]</c>. Caller owns.</param>
    /// <param name="sinSrc">Cross-attn-K RoPE sin <c>[S_qwen3, headDim]</c>. Caller owns.</param>
    public Tensor Forward(IBackend backend, Tensor x, Tensor context,
        Tensor cosTgt, Tensor sinTgt, Tensor cosSrc, Tensor sinSrc)
    {
        int batch = (int)x.Shape[0];
        int sT5 = (int)x.Shape[1];
        int sQwen = (int)context.Shape[1];
        TensorShape xShape = new TensorShape(batch, sT5, _hidden);

        // ── 1. Self-attn: x = x + self_attn(norm(x)) ──
        Tensor preSelf = new Tensor(xShape, DType.F32);
        backend.RmsNorm(preSelf, x, _normSelfAttnWeight!, _eps);
        Tensor selfOut = Attention(backend, preSelf, preSelf,
            _selfQ!, _selfK!, _selfV!, _selfO!, _selfQNorm, _selfKNorm,
            cosTgt, sinTgt, cosTgt, sinTgt,                  // self-attn: both Q and K use target positions
            batch, sT5, sT5);
        preSelf.Dispose();
        Tensor afterSelf = new Tensor(xShape, DType.F32);
        backend.Add(afterSelf, x, selfOut);
        selfOut.Dispose();

        // ── 2. Cross-attn: x = x + cross_attn(norm(x), kv=context) ──
        Tensor preCross = new Tensor(xShape, DType.F32);
        backend.RmsNorm(preCross, afterSelf, _normCrossAttnWeight!, _eps);
        Tensor crossOut = Attention(backend, preCross, context,
            _crossQ!, _crossK!, _crossV!, _crossO!, _crossQNorm, _crossKNorm,
            cosTgt, sinTgt, cosSrc, sinSrc,                  // cross-attn: Q uses target positions, K uses source positions
            batch, sT5, sQwen);
        preCross.Dispose();
        Tensor afterCross = new Tensor(xShape, DType.F32);
        backend.Add(afterCross, afterSelf, crossOut);
        afterSelf.Dispose();
        crossOut.Dispose();

        // ── 3. MLP: x = x + mlp(norm(x)) ──
        Tensor preMlp = new Tensor(xShape, DType.F32);
        backend.RmsNorm(preMlp, afterCross, _normMlpWeight!, _eps);
        Tensor mlpOut = Mlp(backend, preMlp, batch, sT5);
        preMlp.Dispose();
        Tensor result = new Tensor(xShape, DType.F32);
        backend.Add(result, afterCross, mlpOut);
        afterCross.Dispose();
        mlpOut.Dispose();
        return result;
    }

    /// <summary>Multi-head attention with QK-RMSNorm and RoPE on Q+K (separate position tables per stream).
    /// Q comes from <paramref name="qSource"/> <c>[B, qLen, hidden]</c>, K/V from <paramref name="kvSource"/>
    /// <c>[B, kvLen, hidden]</c>.</summary>
    private Tensor Attention(IBackend backend, Tensor qSource, Tensor kvSource,
        Tensor qWeight, Tensor kWeight, Tensor vWeight, Tensor oWeight,
        QkNorm qNorm, QkNorm kNorm,
        Tensor cosQ, Tensor sinQ, Tensor cosK, Tensor sinK,
        int batch, int qLen, int kvLen)
    {
        TensorShape qFlatShape = new TensorShape(batch, qLen, _hidden);
        TensorShape kvFlatShape = new TensorShape(batch, kvLen, _hidden);

        // Project to Q/K/V (no biases).
        Tensor q = new Tensor(qFlatShape, DType.F32);
        Tensor k = new Tensor(kvFlatShape, DType.F32);
        Tensor v = new Tensor(kvFlatShape, DType.F32);
        backend.Linear(q, qSource, qWeight, null);
        backend.Linear(k, kvSource, kWeight, null);
        backend.Linear(v, kvSource, vWeight, null);

        // Reshape to multi-head [B, H, S, D].
        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, qLen, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, kvLen, _numHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, kvLen, _numHeads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        // Per-head QK-RMSNorm. NB: applied BEFORE RoPE — matches diffusion-pipe (the .view(...) + q_norm() comes
        // before apply_rotary_pos_emb()).
        Tensor qNormed = new Tensor(qMh.Shape, DType.F32);
        Tensor kNormed = new Tensor(kMh.Shape, DType.F32);
        qNorm.Forward(qNormed, qMh, batch * _numHeads * qLen);
        kNorm.Forward(kNormed, kMh, batch * _numHeads * kvLen);
        qMh.Dispose(); kMh.Dispose();

        // RoPE rotation. Q gets cosQ/sinQ (target stream's position table); K gets cosK/sinK (source's).
        // Layout: [B, H, S, D] heads-first, rotation is over the last dim D.
        ApplyRotaryHalf(qNormed, cosQ, sinQ, batch, _numHeads, qLen, _headDim);
        ApplyRotaryHalf(kNormed, cosK, sinK, batch, _numHeads, kvLen, _headDim);

        // Scaled dot-product attention.
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qNormed.Shape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qNormed, kNormed, vMh, null, scale);
        qNormed.Dispose(); kNormed.Dispose(); vMh.Dispose();

        // Reshape back + output projection.
        Tensor flat = DiTUtils.ReshapeFromMultiHead(attnOut, batch, qLen, _numHeads, _headDim);
        attnOut.Dispose();
        Tensor output = new Tensor(qFlatShape, DType.F32);
        backend.Linear(output, flat, oWeight, null);
        flat.Dispose();
        return output;
    }

    /// <summary>Applies LLaMA-style "rotate-half + concat-duplicate" RoPE in-place to a <c>[B, H, S, D]</c>
    /// tensor using the precomputed <paramref name="cos"/>/<paramref name="sin"/> <c>[S, D]</c> tables.
    /// The pairing is <c>(i, i + halfDim)</c> with shared <c>cos[i] == cos[i + halfDim]</c> values.</summary>
    private static void ApplyRotaryHalf(Tensor qOrK, Tensor cos, Tensor sin,
        int batch, int numHeads, int seqLen, int headDim)
    {
        int halfDim = headDim / 2;
        float* xPtr = (float*)qOrK.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    long vecOff = (((long)b * numHeads + h) * seqLen + s) * headDim;
                    long tabOff = (long)s * headDim;
                    for (int i = 0; i < halfDim; i++)
                    {
                        float lo = xPtr[vecOff + i];
                        float hi = xPtr[vecOff + i + halfDim];
                        float c = cosPtr[tabOff + i];   // == cosPtr[tabOff + i + halfDim] by construction
                        float si = sinPtr[tabOff + i];  // == sinPtr[tabOff + i + halfDim]
                        // rotate_half([lo, hi]) = [-hi, lo], so:
                        //   out[i]         = lo * cos + (-hi) * sin
                        //   out[i+halfDim] = hi * cos + lo    * sin
                        xPtr[vecOff + i] = lo * c - hi * si;
                        xPtr[vecOff + i + halfDim] = hi * c + lo * si;
                    }
                }
            }
        }
    }

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
}
