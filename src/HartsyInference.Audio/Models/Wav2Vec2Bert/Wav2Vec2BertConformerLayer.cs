using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Wav2Vec2Bert;

/// <summary>One <c>Wav2Vec2BertModel</c> Conformer block (real <c>modeling_wav2vec2_bert.py</c>'s
/// <c>Wav2Vec2BertEncoderLayer</c>): macaron feed-forward (half-step residual) on both sides of a relative-key
/// self-attention and a causal-padded convolution module — <c>x += 0.5·FFN1(norm(x)); x += Attn(norm(x)); x +=
/// Conv(norm(x)); x += 0.5·FFN2(norm(x)); x = finalNorm(x)</c>.</summary>
/// <remarks>Structurally related to but distinct from IndexTTS's own <c>IndexTtsConformerLayer</c>: that one is
/// non-macaron (single FFN, no half-step) with Transformer-XL relative attention (<c>pos_bias_u/v</c> + skew) and
/// symmetric conv padding; this one is macaron with Shaw-style "relative_key" attention (a per-distance embedding
/// bucket, no skew) and a strictly causal (left-only) depthwise conv pad — confirmed from the real
/// <c>modeling_wav2vec2_bert.py</c> source, not assumed from the similar name.</remarks>
internal sealed unsafe class Wav2Vec2BertConformerLayer
{
    private readonly Wav2Vec2BertConfig _cfg;
    private readonly int _channels, _numHeads, _headDim;

    private Tensor? _ffn1NormW, _ffn1NormB, _ffn1W1, _ffn1B1, _ffn1W2, _ffn1B2;
    private Tensor? _attnNormW, _attnNormB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _distanceEmbedding;
    private Tensor? _convNormW, _convNormB, _pw1W, _dwW, _dwLnW, _dwLnB, _pw2W;
    private Tensor? _ffn2NormW, _ffn2NormB, _ffn2W1, _ffn2B1, _ffn2W2, _ffn2B2;
    private Tensor? _finalNormW, _finalNormB;

    public Wav2Vec2BertConformerLayer(Wav2Vec2BertConfig cfg)
    {
        _cfg = cfg;
        _channels = cfg.HiddenSize;
        _numHeads = cfg.NumAttentionHeads;
        _headDim = _channels / _numHeads;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _ffn1NormW = WhisperOps.EnsureF32(w[$"{prefix}.ffn1_layer_norm.weight"]);
        _ffn1NormB = WhisperOps.EnsureF32(w[$"{prefix}.ffn1_layer_norm.bias"]);
        _ffn1W1 = WhisperOps.EnsureF32(w[$"{prefix}.ffn1.intermediate_dense.weight"]); _ffn1B1 = WhisperOps.EnsureF32(w[$"{prefix}.ffn1.intermediate_dense.bias"]);
        _ffn1W2 = WhisperOps.EnsureF32(w[$"{prefix}.ffn1.output_dense.weight"]); _ffn1B2 = WhisperOps.EnsureF32(w[$"{prefix}.ffn1.output_dense.bias"]);

        _attnNormW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn_layer_norm.weight"]);
        _attnNormB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn_layer_norm.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_q.weight"]); _qB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_q.bias"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_k.weight"]); _kB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_k.bias"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_v.weight"]); _vB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_v.bias"]);
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_out.weight"]); _oB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_out.bias"]);
        _distanceEmbedding = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.distance_embedding.weight"]);

        _convNormW = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.layer_norm.weight"]);
        _convNormB = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.layer_norm.bias"]);
        _pw1W = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.pointwise_conv1.weight"]);       // bias=False
        _dwW = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.depthwise_conv.weight"]);          // bias=False
        _dwLnW = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.depthwise_layer_norm.weight"]);
        _dwLnB = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.depthwise_layer_norm.bias"]);
        _pw2W = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.pointwise_conv2.weight"]);        // bias=False

        _ffn2NormW = WhisperOps.EnsureF32(w[$"{prefix}.ffn2_layer_norm.weight"]);
        _ffn2NormB = WhisperOps.EnsureF32(w[$"{prefix}.ffn2_layer_norm.bias"]);
        _ffn2W1 = WhisperOps.EnsureF32(w[$"{prefix}.ffn2.intermediate_dense.weight"]); _ffn2B1 = WhisperOps.EnsureF32(w[$"{prefix}.ffn2.intermediate_dense.bias"]);
        _ffn2W2 = WhisperOps.EnsureF32(w[$"{prefix}.ffn2.output_dense.weight"]); _ffn2B2 = WhisperOps.EnsureF32(w[$"{prefix}.ffn2.output_dense.bias"]);

        _finalNormW = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.bias"]);
    }

    /// <summary>Runs the block over <paramref name="seq"/> <c>[1, T, C]</c> (channels-last); returns a new tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor seq, int t)
    {
        float eps = _cfg.LayerNormEps;
        Tensor x = new(seq.Shape, DType.F32);
        backend.CopyTo(x, seq);

        // 1. Feed-forward 1 (half-step).
        Tensor ffn1Normed = new(x.Shape, DType.F32);
        backend.LayerNorm(ffn1Normed, x, _ffn1NormW!, _ffn1NormB!, eps);
        Tensor ffn1 = FeedForward(backend, ffn1Normed, t, _ffn1W1!, _ffn1B1, _ffn1W2!, _ffn1B2);
        ffn1Normed.Dispose();
        AddScaledInPlace(x, ffn1, 0.5f); ffn1.Dispose();

        // 2. Self-attention (relative_key bias).
        Tensor attnNormed = new(x.Shape, DType.F32);
        backend.LayerNorm(attnNormed, x, _attnNormW!, _attnNormB!, eps);
        Tensor attn = RelativeKeyAttention(backend, attnNormed, t);
        attnNormed.Dispose();
        AddInPlace(x, attn); attn.Dispose();

        // 3. Convolution module (causal left-pad).
        Tensor convNormed = new(x.Shape, DType.F32);
        backend.LayerNorm(convNormed, x, _convNormW!, _convNormB!, eps);
        Tensor conv = ConvModule(backend, convNormed, t);
        convNormed.Dispose();
        AddInPlace(x, conv); conv.Dispose();

        // 4. Feed-forward 2 (half-step).
        Tensor ffn2Normed = new(x.Shape, DType.F32);
        backend.LayerNorm(ffn2Normed, x, _ffn2NormW!, _ffn2NormB!, eps);
        Tensor ffn2 = FeedForward(backend, ffn2Normed, t, _ffn2W1!, _ffn2B1, _ffn2W2!, _ffn2B2);
        ffn2Normed.Dispose();
        AddScaledInPlace(x, ffn2, 0.5f); ffn2.Dispose();

        Tensor outT = new(x.Shape, DType.F32);
        backend.LayerNorm(outT, x, _finalNormW!, _finalNormB!, eps);
        x.Dispose();
        return outT;
    }

    /// <summary>SiLU-gated two-layer MLP (<c>Wav2Vec2BertFeedForward</c>): <c>Linear → SiLU → Linear</c>.</summary>
    private Tensor FeedForward(IBackend backend, Tensor normed, int t, Tensor w1, Tensor? b1, Tensor w2, Tensor? b2)
    {
        int inner = (int)w1.Shape[0];
        Tensor h = WhisperOps.ProjectLinear(backend, normed, w1, b1, 1, t, _channels, inner);
        backend.Silu(h, h);
        Tensor o = WhisperOps.ProjectLinear(backend, h, w2, b2, 1, t, inner, _channels);
        h.Dispose();
        return o;
    }

    /// <summary>Shaw-style "relative_key" attention (real <c>_apply_relative_key_position_encoding</c> +
    /// <c>eager_attention_forward</c>): content score <c>q·kᵀ</c> plus a position bias from a learned embedding
    /// indexed by the clamped relative distance <c>clamp(j-i, -left, right)</c> — no skew trick, no
    /// <c>pos_bias_u/v</c> (those belong to the different "relative" Transformer-XL variant).</summary>
    private Tensor RelativeKeyAttention(IBackend backend, Tensor x, int t)
    {
        int h = _numHeads, d = _headDim, c = _channels;
        int left = _cfg.LeftMaxPositionEmbeddings, right = _cfg.RightMaxPositionEmbeddings;
        float scaling = 1f / MathF.Sqrt(d);

        Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, _qB, 1, t, c, c);
        Tensor k = WhisperOps.ProjectLinear(backend, x, _kW!, _kB, 1, t, c, c);
        Tensor v = WhisperOps.ProjectLinear(backend, x, _vW!, _vB, 1, t, c, c);

        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        float* distEmb = (float*)_distanceEmbedding!.DataPointer;   // [numPositions, headDim]

        Tensor outMerged = new(new TensorShape(1, t, c), DType.F32);
        float* om = (float*)outMerged.DataPointer;
        float[] scores = new float[t];

        for (int head = 0; head < h; head++)
        {
            int hOff = head * d;
            for (int i = 0; i < t; i++)
            {
                float* qi = qp + (long)i * c + hOff;
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    float* kj = kp + (long)j * c + hOff;
                    int distance = Math.Clamp(j - i, -left, right);
                    float* pos = distEmb + (long)(distance + left) * d;
                    float content = 0f, posScore = 0f;
                    for (int e = 0; e < d; e++) { content += qi[e] * kj[e]; posScore += qi[e] * pos[e]; }
                    float s = (content + posScore) * scaling;
                    scores[j] = s;
                    if (s > maxS) maxS = s;
                }
                float sum = 0f;
                for (int j = 0; j < t; j++) { float e = MathF.Exp(scores[j] - maxS); scores[j] = e; sum += e; }
                float invSum = 1f / sum;
                float* oi = om + (long)i * c + hOff;
                for (int e = 0; e < d; e++) oi[e] = 0f;
                for (int j = 0; j < t; j++)
                {
                    float a = scores[j] * invSum;
                    float* vj = vp + (long)j * c + hOff;
                    for (int e = 0; e < d; e++) oi[e] += a * vj[e];
                }
            }
        }
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor o = WhisperOps.ProjectLinear(backend, outMerged, _oW!, _oB, 1, t, c, c);
        outMerged.Dispose();
        return o;
    }

    /// <summary>Real <c>Wav2Vec2BertConvolutionModule</c>: pointwise expand (no bias) → GLU → STRICTLY CAUSAL
    /// depthwise conv (left-pad <c>kernel-1</c>, no right pad — confirmed from the real source's explicit
    /// <c>F.pad(x, (kernel_size-1, 0))</c> then zero-padding Conv1d) → LayerNorm → SiLU → pointwise contract (no
    /// bias). Channels-last in/out, channels-first internally for the convs.</summary>
    private Tensor ConvModule(IBackend backend, Tensor seqChLast, int t)
    {
        int c = _channels, kernel = _cfg.ConvDepthwiseKernelSize;
        Tensor chFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(chFirst, seqChLast, t, c);

        Tensor gateChFirst = new(new TensorShape(1, 2 * c, t), DType.F32);
        backend.Conv1d(gateChFirst, chFirst, _pw1W!, null, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        chFirst.Dispose();

        Tensor glu = new(new TensorShape(1, c, t), DType.F32);
        GluChannelSplit(glu, gateChFirst, c, t);
        gateChFirst.Dispose();

        Tensor dw = new(new TensorShape(1, c, t), DType.F32);
        backend.Conv1d(dw, glu, _dwW!, null, stride: 1, padLeft: kernel - 1, padRight: 0, dilation: 1, groups: c);
        glu.Dispose();

        Tensor dwChLast = new(new TensorShape(t, c), DType.F32);
        backend.Transpose2D(dwChLast, dw, c, t);
        dw.Dispose();
        Tensor normed = new(dwChLast.Shape, DType.F32);
        backend.LayerNorm(normed, dwChLast, _dwLnW!, _dwLnB!, _cfg.LayerNormEps);
        dwChLast.Dispose();
        backend.Silu(normed, normed);

        Tensor normedChFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(normedChFirst, normed, t, c);
        normed.Dispose();
        Tensor pw2 = new(new TensorShape(1, c, t), DType.F32);
        backend.Conv1d(pw2, normedChFirst, _pw2W!, null, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        normedChFirst.Dispose();

        Tensor outChLast = new(new TensorShape(1, t, c), DType.F32);
        backend.Transpose2D(outChLast, pw2, c, t);
        pw2.Dispose();
        return outChLast;
    }

    private static void GluChannelSplit(Tensor output, Tensor gate, int c, int t)
    {
        float* gp = (float*)gate.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            float* firstRow = gp + (long)ch * t;
            float* secondRow = gp + (long)(c + ch) * t;
            float* outRow = op + (long)ch * t;
            for (int i = 0; i < t; i++)
            {
                float gate2 = 1f / (1f + MathF.Exp(-secondRow[i]));
                outRow[i] = firstRow[i] * gate2;
            }
        }
    }

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] += sp[i];
    }

    private static void AddScaledInPlace(Tensor dst, Tensor src, float scale)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] += sp[i] * scale;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _ffn1NormW, _ffn1NormB, _ffn1W1, _ffn1B1, _ffn1W2, _ffn1B2,
            _attnNormW, _attnNormB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _distanceEmbedding,
            _convNormW, _convNormB, _pw1W, _dwW, _dwLnW, _dwLnB, _pw2W,
            _ffn2NormW, _ffn2NormB, _ffn2W1, _ffn2B1, _ffn2W2, _ffn2B2,
            _finalNormW, _finalNormB,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
