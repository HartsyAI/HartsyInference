using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>One IndexTTS-1.5 conditioning-encoder Conformer block (ESPnet <c>ConformerEncoderLayer</c>, non-macaron,
/// with a convolution module): <c>x += RelPosMHSA(norm_mha(x), pos); x += ConvModule(norm_conv(x)); x +=
/// FFN(norm_ff(x)); x = norm_final(x)</c>.</summary>
/// <remarks>The attention keeps CosyVoice's <c>pos_bias_u</c>/<c>pos_bias_v</c> + <c>linear_pos</c> structure
/// (<see cref="HartsyInference.Audio.Models.CosyVoice.UpsampleConformerEncoder"/>) but, exactly like IndexTTS's own
/// <c>RelPositionMultiHeadedAttention</c>, has <c>rel_shift</c> removed and is fed the plain forward sinusoid for
/// positions <c>0..T-1</c> (see <see cref="IndexTtsConformerEncoder"/>) rather than a <c>2T-1</c> relative table. The
/// block adds the conv module and closing norm CosyVoice's <c>use_cnn_module=False</c> variant omits, and uses eps
/// 1e-5 (vs. CosyVoice's 1e-12) per the real checkpoint's LayerNorm construction.</remarks>
internal sealed unsafe class IndexTtsConformerLayer
{
    private readonly IndexTtsConformerConfig _cfg;
    private readonly int _channels, _numHeads, _headDim;

    private Tensor? _mhaNormW, _mhaNormB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _posW, _posBiasU, _posBiasV;
    private Tensor? _convNormW, _convNormB, _pw1W, _pw1B, _dwW, _dwB, _convLnW, _convLnB, _pw2W, _pw2B;
    private Tensor? _ffNormW, _ffNormB, _ffW1, _ffB1, _ffW2, _ffB2;
    private Tensor? _finalNormW, _finalNormB;

    public IndexTtsConformerLayer(IndexTtsConformerConfig cfg)
    {
        _cfg = cfg;
        _channels = cfg.OutputSize;
        _numHeads = cfg.AttentionHeads;
        _headDim = _channels / _numHeads;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _mhaNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_mha.weight"]);
        _mhaNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_mha.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_q.weight"]); _qB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_q.bias"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_k.weight"]); _kB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_k.bias"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_v.weight"]); _vB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_v.bias"]);
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_out.weight"]); _oB = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_out.bias"]);
        _posW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_pos.weight"]);
        _posBiasU = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.pos_bias_u"]);
        _posBiasV = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.pos_bias_v"]);

        _convNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_conv.weight"]);
        _convNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_conv.bias"]);
        _pw1W = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.pointwise_conv1.weight"]); _pw1B = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.pointwise_conv1.bias"]);
        _dwW = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.depthwise_conv.weight"]); _dwB = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.depthwise_conv.bias"]);
        _convLnW = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.norm.weight"]); _convLnB = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.norm.bias"]);
        _pw2W = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.pointwise_conv2.weight"]); _pw2B = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.pointwise_conv2.bias"]);

        _ffNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_ff.weight"]);
        _ffNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_ff.bias"]);
        _ffW1 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_1.weight"]); _ffB1 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_1.bias"]);
        _ffW2 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_2.weight"]); _ffB2 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_2.bias"]);

        _finalNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_final.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_final.bias"]);
    }

    /// <summary>Runs the block over <paramref name="seq"/> <c>[1, T, C]</c> (channels-last) with position table
    /// <paramref name="posEmb"/> <c>[1, T, C]</c>; returns a new tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor seq, Tensor posEmb, int t)
    {
        float eps = _cfg.LayerNormEps;
        Tensor x = new(seq.Shape, DType.F32);
        backend.CopyTo(x, seq);

        Tensor mhaNormed = new(x.Shape, DType.F32);
        backend.LayerNorm(mhaNormed, x, _mhaNormW!, _mhaNormB!, eps);
        Tensor att = RelPosAttention(backend, mhaNormed, posEmb, t);
        mhaNormed.Dispose();
        AddInPlace(x, att); att.Dispose();

        Tensor convNormed = new(x.Shape, DType.F32);
        backend.LayerNorm(convNormed, x, _convNormW!, _convNormB!, eps);
        Tensor conv = ConvModule(backend, convNormed, t);
        convNormed.Dispose();
        AddInPlace(x, conv); conv.Dispose();

        Tensor ffNormed = new(x.Shape, DType.F32);
        backend.LayerNorm(ffNormed, x, _ffNormW!, _ffNormB!, eps);
        int ffDim = (int)_ffW1!.Shape[0];
        Tensor h1 = WhisperOps.ProjectLinear(backend, ffNormed, _ffW1!, _ffB1, 1, t, _channels, ffDim);
        ffNormed.Dispose();
        backend.Silu(h1, h1);
        Tensor ff = WhisperOps.ProjectLinear(backend, h1, _ffW2!, _ffB2, 1, t, ffDim, _channels);
        h1.Dispose();
        AddInPlace(x, ff); ff.Dispose();

        Tensor outT = new(x.Shape, DType.F32);
        backend.LayerNorm(outT, x, _finalNormW!, _finalNormB!, eps);
        x.Dispose();
        return outT;
    }

    /// <summary>Multi-head attention with IndexTTS's position term: per head, <c>scores = ((q+u)·kᵀ + (q+v)·pᵀ) / √d</c>
    /// with <c>p = linear_pos(pe[0:T])</c> indexed by the key position and no <c>rel_shift</c>.</summary>
    private Tensor RelPosAttention(IBackend backend, Tensor x, Tensor posEmb, int t)
    {
        int h = _numHeads, d = _headDim, c = _channels, posLen = t;
        Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, _qB, 1, t, c, c);
        Tensor k = WhisperOps.ProjectLinear(backend, x, _kW!, _kB, 1, t, c, c);
        Tensor v = WhisperOps.ProjectLinear(backend, x, _vW!, _vB, 1, t, c, c);
        Tensor p = WhisperOps.ProjectLinear(backend, posEmb, _posW!, null, 1, posLen, c, c);

        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer, pp = (float*)p.DataPointer;
        float* bu = (float*)_posBiasU!.DataPointer, bv = (float*)_posBiasV!.DataPointer;

        Tensor outMerged = new(new TensorShape(1, t, c), DType.F32);
        float* om = (float*)outMerged.DataPointer;
        float invSqrtD = 1f / MathF.Sqrt(d);
        float[] scores = new float[t];

        for (int head = 0; head < h; head++)
        {
            int hOff = head * d;
            float* buH = bu + (long)head * d;
            float* bvH = bv + (long)head * d;
            for (int i = 0; i < t; i++)
            {
                float* qi = qp + (long)i * c + hOff;
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    float* kj = kp + (long)j * c + hOff;
                    float* pj = pp + (long)j * c + hOff;   // position of KEY j (no rel_shift in the reference)
                    float ac = 0f, bd = 0f;
                    for (int e = 0; e < d; e++)
                    {
                        ac += (qi[e] + buH[e]) * kj[e];
                        bd += (qi[e] + bvH[e]) * pj[e];
                    }
                    float s = (ac + bd) * invSqrtD;
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
        q.Dispose(); k.Dispose(); v.Dispose(); p.Dispose();

        Tensor o = WhisperOps.ProjectLinear(backend, outMerged, _oW!, _oB, 1, t, c, c);
        outMerged.Dispose();
        return o;
    }

    /// <summary>ESPnet <c>ConvolutionModule</c>: pointwise expand → GLU → depthwise conv (kernel <see cref="IndexTtsConformerConfig.ConvKernel"/>, groups=channels) → LayerNorm → SiLU → pointwise contract. Channels-last in/out, channels-first internally for the convs.</summary>
    private Tensor ConvModule(IBackend backend, Tensor seqChLast, int t)
    {
        int pad = _cfg.ConvKernel / 2;
        int c = _channels;
        Tensor chFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(chFirst, seqChLast, t, c);

        Tensor gateChFirst = new(new TensorShape(1, 2 * c, t), DType.F32);
        backend.Conv1d(gateChFirst, chFirst, _pw1W!, _pw1B, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        chFirst.Dispose();

        // GLU over the channel axis (dim=1 of [1,2c,t]): first half * sigmoid(second half).
        Tensor glu = new(new TensorShape(1, c, t), DType.F32);
        GluChannelSplit(glu, gateChFirst, c, t);
        gateChFirst.Dispose();

        Tensor dw = new(new TensorShape(1, c, t), DType.F32);
        backend.Conv1d(dw, glu, _dwW!, _dwB, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: c);
        glu.Dispose();

        Tensor dwChLast = new(new TensorShape(t, c), DType.F32);
        backend.Transpose2D(dwChLast, dw, c, t);
        dw.Dispose();
        Tensor normed = new(dwChLast.Shape, DType.F32);
        backend.LayerNorm(normed, dwChLast, _convLnW!, _convLnB!, _cfg.LayerNormEps);
        dwChLast.Dispose();
        backend.Silu(normed, normed);

        Tensor normedChFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(normedChFirst, normed, t, c);
        normed.Dispose();
        Tensor pw2 = new(new TensorShape(1, c, t), DType.F32);
        backend.Conv1d(pw2, normedChFirst, _pw2W!, _pw2B, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        normedChFirst.Dispose();

        Tensor outChLast = new(new TensorShape(1, t, c), DType.F32);
        backend.Transpose2D(outChLast, pw2, c, t);
        pw2.Dispose();
        return outChLast;
    }

    /// <summary>Splits a <c>[1, 2c, t]</c> channels-first tensor on the channel axis into <c>[1, c, t]</c> halves and applies GLU: <c>out = first * sigmoid(second)</c>.</summary>
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

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _mhaNormW, _mhaNormB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _posW, _posBiasU, _posBiasV,
            _convNormW, _convNormB, _pw1W, _pw1B, _dwW, _dwB, _convLnW, _convLnB, _pw2W, _pw2B,
            _ffNormW, _ffNormB, _ffW1, _ffB1, _ffW2, _ffB2, _finalNormW, _finalNormB,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
