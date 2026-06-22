using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>StyleTTS 2's diffusion network — a small 1-D transformer over the 256-d style latent (treated
/// as a length-1 sequence) conditioned on the diffusion timestep, the PLBERT text features (cross-attention),
/// and an optional reference style vector (AdaLayerNorm modulation). Mirrors <c>StyleTransformer1d</c> from
/// <c>Modules/diffusion/modules.py</c>: <c>channels=256, context_embedding_features=768, num_layers=3,
/// num_heads=8, head_features=64, multiplier=2</c>.
///
/// <para>Per layer: <c>x = SelfAttention(AdaLN(x, style), time)</c> → <c>x = CrossAttention(x, text)</c> →
/// <c>x = FeedForward(x, time)</c>, all residual. The style latent is length-1 so self-attention is a
/// learned per-token projection that the time embedding biases; cross-attention is the real read over the
/// variable-length PLBERT context.</para>
///
/// <para>Weights are pre-cast to F32 (<see cref="WhisperOps.EnsureF32"/>) at load. With no weights present
/// the network returns a zero velocity so the EDM/CFG wrapper + sampler still run deterministically.</para></summary>
public sealed unsafe class StyleTransformer1d
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _headFeatures;
    private readonly int _innerDim;
    private readonly int _contextFeatures;
    private readonly int _ffDim;
    private readonly float _attnScale;
    private readonly StyleTransformerLayer[] _layers;

    // Timestep MLP: sinusoidal(t) → Linear → SiLU → Linear (channels-wide time features).
    private readonly Tensor? _timeW1, _timeB1, _timeW2, _timeB2;

    // Input/output 1×1 conv (== Linear over channels): [1, in=1, len] is projected to [channels] then back.
    private readonly Tensor? _inW, _inB, _outW, _outB;

    /// <summary>True once the projection weights are present — otherwise the forward is a zero fallback.</summary>
    public bool HasWeights => _inW is not null;

    public StyleTransformer1d(
        int channels,
        int numHeads,
        int headFeatures,
        int contextFeatures,
        int multiplier,
        StyleTransformerLayer[] layers,
        Tensor? timeW1, Tensor? timeB1, Tensor? timeW2, Tensor? timeB2,
        Tensor? inW, Tensor? inB, Tensor? outW, Tensor? outB)
    {
        if (channels <= 0) throw new ArgumentException("channels must be positive.", nameof(channels));
        if (numHeads <= 0) throw new ArgumentException("numHeads must be positive.", nameof(numHeads));
        _channels = channels;
        _numHeads = numHeads;
        _headFeatures = headFeatures;
        _innerDim = numHeads * headFeatures;
        _contextFeatures = contextFeatures;
        _ffDim = channels * multiplier;
        _attnScale = 1f / MathF.Sqrt(headFeatures);
        _layers = layers ?? throw new ArgumentNullException(nameof(layers));
        _timeW1 = timeW1; _timeB1 = timeB1; _timeW2 = timeW2; _timeB2 = timeB2;
        _inW = inW; _inB = inB; _outW = outW; _outB = outB;
    }

    /// <summary>Predicts the denoised style. <paramref name="noisyStyle"/> is <c>[1, 1, 256]</c> (the EDM
    /// <c>c_in·x</c> input), <paramref name="sigma"/> is the <c>c_noise = ln(σ)/4</c> timestep scalar, and
    /// <paramref name="cond"/> packs the PLBERT text features <c>[1, seq, 768]</c>; the optional reference
    /// style is supplied via <see cref="SetReferenceStyle"/>. Returns a <c>[1, 1, 256]</c> tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor noisyStyle, float sigma, Tensor? cond)
    {
        if (!HasWeights)
            return new Tensor(noisyStyle.Shape, DType.F32);

        Tensor x3 = noisyStyle.Reshape(new TensorShape(1, 1, _channels));

        // Input projection (1×1 conv over channels) → working width _channels.
        Tensor h = WhisperOps.ProjectLinear(backend, x3, _inW!, _inB, 1, 1, _channels, _channels);

        Tensor timeEmb = TimeEmbedding(backend, sigma);

        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = ForwardLayer(backend, _layers[i], h, timeEmb, cond);
            h.Dispose();
            h = next;
        }
        timeEmb.Dispose();

        Tensor outT = WhisperOps.ProjectLinear(backend, h, _outW!, _outB, 1, 1, _channels, _channels);
        h.Dispose();
        return outT.Reshape(noisyStyle.Shape);
    }

    /// <summary>One transformer layer: AdaLN(style) → self-attn(+time) → cross-attn(text) → FF(+time), all residual.</summary>
    private Tensor ForwardLayer(IBackend backend, StyleTransformerLayer layer, Tensor x, Tensor timeEmb, Tensor? cond)
    {
        // ── Self-attention with AdaLN (reference-style) modulation + time bias. ──
        Tensor normed = AdaLayerNorm(backend, x, layer);
        AddBroadcast(normed, timeEmb);
        Tensor attn = SelfAttention(backend, layer, normed);
        normed.Dispose();
        Tensor x1 = Residual(x, attn);
        attn.Dispose();

        // ── Cross-attention over the PLBERT text context. ──
        Tensor x2;
        if (cond is not null && layer.CrossQW is not null)
        {
            Tensor cnorm = LayerNormAffine(backend, x1, layer.CrossNormW, layer.CrossNormB);
            Tensor cross = CrossAttention(backend, layer, cnorm, cond);
            cnorm.Dispose();
            x2 = Residual(x1, cross);
            cross.Dispose();
            x1.Dispose();
        }
        else
        {
            x2 = x1;
        }

        // ── Feed-forward with a time bias. ──
        Tensor fnorm = LayerNormAffine(backend, x2, layer.FfNormW, layer.FfNormB);
        AddBroadcast(fnorm, timeEmb);
        Tensor ff = FeedForward(backend, layer, fnorm);
        fnorm.Dispose();
        Tensor outT = Residual(x2, ff);
        ff.Dispose();
        x2.Dispose();
        return outT;
    }

    /// <summary>AdaLayerNorm: layer-norm then scale/shift by <c>γ,β</c> regressed from the reference style.
    /// Without a style vector (or AdaLN weights) this degrades to a plain affine layer-norm.</summary>
    private Tensor AdaLayerNorm(IBackend backend, Tensor x, StyleTransformerLayer layer)
    {
        if (layer.AdaLnW is null || _refStyle is null)
            return LayerNormAffine(backend, x, layer.SelfNormW, layer.SelfNormB);

        // γ,β = Linear(style) → 2·channels; modulate the non-affine layer-norm of x.
        Tensor gb = WhisperOps.ProjectLinear(backend, _refStyle, layer.AdaLnW!, layer.AdaLnB, 1, 1, _contextFeatures, 2 * _channels);
        Tensor normed = new(new TensorShape(1, 1, _channels), DType.F32);
        backend.LayerNormNoAffine(normed, x, 1e-5f);
        float* np = (float*)normed.DataPointer;
        float* gbp = (float*)gb.DataPointer;
        for (int d = 0; d < _channels; d++)
        {
            float gamma = 1f + gbp[d];
            float beta = gbp[_channels + d];
            np[d] = np[d] * gamma + beta;
        }
        gb.Dispose();
        return normed;
    }

    private Tensor LayerNormAffine(IBackend backend, Tensor x, Tensor? weight, Tensor? bias)
    {
        Tensor outT = new(new TensorShape(1, 1, _channels), DType.F32);
        if (weight is not null && bias is not null)
            backend.LayerNorm(outT, x, weight, bias, 1e-5f);
        else
            backend.LayerNormNoAffine(outT, x, 1e-5f);
        return outT;
    }

    /// <summary>Self-attention over the length-1 style sequence. With one token the softmax is a no-op, so
    /// this reduces to <c>outProj(V)</c> with V the value projection — but we run the full QKV path so the
    /// learned projections (and their interaction with the AdaLN/time-biased input) are exercised faithfully.</summary>
    private Tensor SelfAttention(IBackend backend, StyleTransformerLayer layer, Tensor normed)
    {
        Tensor q = WhisperOps.ProjectLinear(backend, normed, layer.SelfQW!, layer.SelfQB, 1, 1, _channels, _innerDim);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, layer.SelfKW!, layer.SelfKB, 1, 1, _channels, _innerDim);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, layer.SelfVW!, layer.SelfVB, 1, 1, _channels, _innerDim);
        Tensor ctx = Attend(backend, q, k, v, seqQ: 1, seqKv: 1);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor outT = WhisperOps.ProjectLinear(backend, ctx, layer.SelfOW!, layer.SelfOB, 1, 1, _innerDim, _channels);
        ctx.Dispose();
        return outT;
    }

    /// <summary>Cross-attention: Q from the style token, K/V from the variable-length PLBERT text context.</summary>
    private Tensor CrossAttention(IBackend backend, StyleTransformerLayer layer, Tensor normed, Tensor context)
    {
        int seqKv = (int)context.Shape[1];
        int ctxDim = (int)context.Shape[2];
        Tensor q = WhisperOps.ProjectLinear(backend, normed, layer.CrossQW!, layer.CrossQB, 1, 1, _channels, _innerDim);
        Tensor k = WhisperOps.ProjectLinear(backend, context, layer.CrossKW!, layer.CrossKB, 1, seqKv, ctxDim, _innerDim);
        Tensor v = WhisperOps.ProjectLinear(backend, context, layer.CrossVW!, layer.CrossVB, 1, seqKv, ctxDim, _innerDim);
        Tensor ctx = Attend(backend, q, k, v, seqQ: 1, seqKv: seqKv);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor outT = WhisperOps.ProjectLinear(backend, ctx, layer.CrossOW!, layer.CrossOB, 1, 1, _innerDim, _channels);
        ctx.Dispose();
        return outT;
    }

    /// <summary>Multi-head SDPA: reshapes [1, S, inner] → [1, H, S, headDim], attends, reshapes back.</summary>
    private Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, int seqQ, int seqKv)
    {
        Tensor q4 = new(new TensorShape(1, _numHeads, seqQ, _headFeatures), DType.F32);
        Tensor k4 = new(new TensorShape(1, _numHeads, seqKv, _headFeatures), DType.F32);
        Tensor v4 = new(new TensorShape(1, _numHeads, seqKv, _headFeatures), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(q4, q, 1, seqQ, _numHeads, _headFeatures);
        WhisperOps.ReshapeToMultiHead4D(k4, k, 1, seqKv, _numHeads, _headFeatures);
        WhisperOps.ReshapeToMultiHead4D(v4, v, 1, seqKv, _numHeads, _headFeatures);

        Tensor attn4 = new(new TensorShape(1, _numHeads, seqQ, _headFeatures), DType.F32);
        backend.ScaledDotProductAttention(attn4, q4, k4, v4, null, _attnScale);
        q4.Dispose(); k4.Dispose(); v4.Dispose();

        Tensor ctx = new(new TensorShape(1, seqQ, _innerDim), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(ctx, attn4, 1, seqQ, _numHeads, _headFeatures);
        attn4.Dispose();
        return ctx;
    }

    /// <summary>GEGLU-style feed-forward: <c>Linear → GELU → Linear</c> over the length-1 token.</summary>
    private Tensor FeedForward(IBackend backend, StyleTransformerLayer layer, Tensor x)
    {
        Tensor up = WhisperOps.ProjectLinear(backend, x, layer.FfW1!, layer.FfB1, 1, 1, _channels, _ffDim);
        backend.Gelu(up, up);
        Tensor down = WhisperOps.ProjectLinear(backend, up, layer.FfW2!, layer.FfB2, 1, 1, _ffDim, _channels);
        up.Dispose();
        return down;
    }

    /// <summary>Sinusoidal timestep embedding → 2-layer SiLU MLP → channels-wide time features <c>[1, 1, C]</c>.</summary>
    private Tensor TimeEmbedding(IBackend backend, float t)
    {
        int half = _channels / 2;
        Tensor sin = new(new TensorShape(1, 1, _channels), DType.F32);
        float* sp = (float*)sin.DataPointer;
        double logBase = Math.Log(10000.0) / Math.Max(1, half - 1);
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-logBase * i);
            sp[i] = (float)Math.Sin(t * freq);
            sp[half + i] = (float)Math.Cos(t * freq);
        }
        if (_timeW1 is null)
            return sin;
        Tensor h1 = WhisperOps.ProjectLinear(backend, sin, _timeW1!, _timeB1, 1, 1, _channels, _channels);
        sin.Dispose();
        backend.Silu(h1, h1);
        Tensor h2 = WhisperOps.ProjectLinear(backend, h1, _timeW2!, _timeB2, 1, 1, _channels, _channels);
        h1.Dispose();
        return h2;
    }

    private static Tensor Residual(Tensor a, Tensor b)
    {
        Tensor outT = new(a.Shape, DType.F32);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float* op = (float*)outT.DataPointer;
        long n = a.ElementCount;
        for (long i = 0; i < n; i++) op[i] = ap[i] + bp[i];
        return outT;
    }

    /// <summary>Adds the length-1 time feature <c>[1, 1, C]</c> into <paramref name="target"/> in place.</summary>
    private void AddBroadcast(Tensor target, Tensor timeEmb)
    {
        float* tp = (float*)target.DataPointer;
        float* ep = (float*)timeEmb.DataPointer;
        for (int d = 0; d < _channels; d++) tp[d] += ep[d];
    }

    private Tensor? _refStyle;

    /// <summary>Sets the optional reference-style vector <c>[1, 1, 256]</c> used by the AdaLayerNorm path.
    /// Pass null to disable style modulation (single-speaker / unconditional-style behaviour).</summary>
    public void SetReferenceStyle(Tensor? refStyle) => _refStyle = refStyle;

    /// <summary>All non-null weight tensors, for GPU preload/free by the pipeline.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_timeW1, _timeB1, _timeW2, _timeB2, _inW, _inB, _outW, _outB];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (StyleTransformerLayer layer in _layers)
        {
            Tensor?[] lw =
            [
                layer.AdaLnW, layer.AdaLnB, layer.SelfNormW, layer.SelfNormB,
                layer.SelfQW, layer.SelfQB, layer.SelfKW, layer.SelfKB, layer.SelfVW, layer.SelfVB, layer.SelfOW, layer.SelfOB,
                layer.CrossNormW, layer.CrossNormB,
                layer.CrossQW, layer.CrossQB, layer.CrossKW, layer.CrossKB, layer.CrossVW, layer.CrossVB, layer.CrossOW, layer.CrossOB,
                layer.FfNormW, layer.FfNormB, layer.FfW1, layer.FfB1, layer.FfW2, layer.FfB2,
            ];
            foreach (Tensor? t in lw) if (t is not null) yield return t;
        }
    }
}
