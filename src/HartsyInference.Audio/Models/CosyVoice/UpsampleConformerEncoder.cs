using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>The CosyVoice 2 flow-matching token encoder (<c>cosyvoice/transformer/upsample_encoder.py:UpsampleConformerEncoder</c>), reimplemented to match the real checkpoint exactly.</summary>
/// <remarks>Pipeline: <c>embed</c> (Linear → LayerNorm → ×√d scale, plus a relative-position
/// table) → <c>pre_lookahead_layer</c> (two causal-ish convs + residual) → 6 relative-position Transformer
/// blocks → <c>up_layer</c> (<see cref="Upsample1D"/>: nearest ×2 + left-pad + conv) → <c>up_embed</c>
/// (Linear → LayerNorm → ×√d) → 4 relative-position Transformer blocks → <c>after_norm</c>. Output feeds
/// <c>encoder_proj</c> for the mel conditioning μ.
///
/// <para><b>Block</b> (ESPnet <c>ConformerEncoderLayer</c> with <c>macaron_style=False</c>,
/// <c>use_cnn_module=False</c>): <c>x += RelPosMHSA(norm_mha(x)); x += FFN(norm_ff(x))</c>. Sub-layer norms
/// use eps 1e-12; the encoder-level <c>after_norm</c> and the embed LayerNorms use eps 1e-5. Attention is
/// Transformer-XL relative position (linear_pos + pos_bias_u/v + rel_shift), NOT plain dot-product.</para></remarks>
public sealed unsafe class UpsampleConformerEncoder : IDisposable
{
    public const int TokenMelRatio = 2;
    private const float BlockNormEps = 1e-12f;
    private const float EmbedNormEps = 1e-5f;

    private readonly int _outputSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly RelPosBlock[] _preBlocks;
    private readonly RelPosBlock[] _postBlocks;
    private int _disposed;

    // embed: Linear(input→output) + LayerNorm.
    private Tensor? _embLinW, _embLinB, _embLnW, _embLnB;
    // pre_lookahead_layer: conv1 (kernel = k1, right-pad k1-1) + conv2 (kernel = k2, left-pad k2-1).
    private Tensor? _plConv1W, _plConv1B, _plConv2W, _plConv2B;
    // up_layer (Upsample1D): conv (kernel = k, stride 1) over a nearest-×ratio + left-pad signal.
    private Tensor? _upConvW, _upConvB;
    // up_embed: Linear + LayerNorm.
    private Tensor? _upEmbLinW, _upEmbLinB, _upEmbLnW, _upEmbLnB;
    // after_norm.
    private Tensor? _afterNormW, _afterNormB;

    public UpsampleConformerEncoder(int outputSize, int numHeads, int numPreBlocks, int numPostBlocks)
    {
        if (outputSize % numHeads != 0)
            throw new ArgumentException($"outputSize {outputSize} must be divisible by numHeads {numHeads}.");
        _outputSize = outputSize;
        _numHeads = numHeads;
        _headDim = outputSize / numHeads;
        _preBlocks = new RelPosBlock[numPreBlocks];
        _postBlocks = new RelPosBlock[numPostBlocks];
        for (int i = 0; i < numPreBlocks; i++) _preBlocks[i] = new RelPosBlock(outputSize, numHeads);
        for (int i = 0; i < numPostBlocks; i++) _postBlocks[i] = new RelPosBlock(outputSize, numHeads);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _embLinW = WhisperOps.EnsureF32(w[$"{p}embed.out.0.weight"]);
        _embLinB = TryGet(w, $"{p}embed.out.0.bias");
        _embLnW = WhisperOps.EnsureF32(w[$"{p}embed.out.1.weight"]);
        _embLnB = WhisperOps.EnsureF32(w[$"{p}embed.out.1.bias"]);

        _plConv1W = WhisperOps.EnsureF32(w[$"{p}pre_lookahead_layer.conv1.weight"]);
        _plConv1B = TryGet(w, $"{p}pre_lookahead_layer.conv1.bias");
        _plConv2W = WhisperOps.EnsureF32(w[$"{p}pre_lookahead_layer.conv2.weight"]);
        _plConv2B = TryGet(w, $"{p}pre_lookahead_layer.conv2.bias");

        for (int i = 0; i < _preBlocks.Length; i++) _preBlocks[i].LoadWeights(w, $"{p}encoders.{i}");

        _upConvW = WhisperOps.EnsureF32(w[$"{p}up_layer.conv.weight"]);
        _upConvB = TryGet(w, $"{p}up_layer.conv.bias");

        _upEmbLinW = WhisperOps.EnsureF32(w[$"{p}up_embed.out.0.weight"]);
        _upEmbLinB = TryGet(w, $"{p}up_embed.out.0.bias");
        _upEmbLnW = WhisperOps.EnsureF32(w[$"{p}up_embed.out.1.weight"]);
        _upEmbLnB = WhisperOps.EnsureF32(w[$"{p}up_embed.out.1.bias"]);

        for (int i = 0; i < _postBlocks.Length; i++) _postBlocks[i].LoadWeights(w, $"{p}up_encoders.{i}");

        _afterNormW = WhisperOps.EnsureF32(w[$"{p}after_norm.weight"]);
        _afterNormB = WhisperOps.EnsureF32(w[$"{p}after_norm.bias"]);
    }

    // TODO(gpu-residency): Embed (xscale), PreLookahead (residual), Upsample (nearest), BuildRelPos, and the
    // RelPosBlock rel-pos attention (scores/softmax/bias) all use host `(float*)DataPointer` loops, which force
    // device→host syncs on CUDA and break GPU residency. Port these to PTX kernels / backend ops for a fully
    // on-device flow encoder.
    /// <summary>Forwards embedded speech tokens <c>[1, T, inputSize]</c> → upsampled features <c>[1, T·<see cref="TokenMelRatio"/>, outputSize]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor tokenEmb, int inputSize)
    {
        if (_embLinW is null) throw new InvalidOperationException("UpsampleConformerEncoder weights not loaded.");
        int t = (int)tokenEmb.Shape[1];

        // embed: Linear → LayerNorm(1e-5) → ×√d, plus the relative-position table for length t.
        Tensor x = Embed(backend, tokenEmb, inputSize, _embLinW!, _embLinB, _embLnW!, _embLnB!);
        Tensor posPre = BuildRelPos(t);
        x = PreLookahead(backend, x, t);
        for (int i = 0; i < _preBlocks.Length; i++)
        {
            Tensor next = _preBlocks[i].Forward(backend, x, posPre, t);
            x.Dispose();
            x = next;
        }
        posPre.Dispose();

        // up_layer (Upsample1D): nearest ×ratio → left-pad → conv.
        Tensor up = Upsample(backend, x, t);
        x.Dispose();
        int tUp = t * TokenMelRatio;

        // up_embed: Linear → LayerNorm → ×√d, plus the relative-position table for length tUp.
        Tensor xu = Embed(backend, up, _outputSize, _upEmbLinW!, _upEmbLinB, _upEmbLnW!, _upEmbLnB!);
        up.Dispose();
        Tensor posUp = BuildRelPos(tUp);
        for (int i = 0; i < _postBlocks.Length; i++)
        {
            Tensor next = _postBlocks[i].Forward(backend, xu, posUp, tUp);
            xu.Dispose();
            xu = next;
        }
        posUp.Dispose();

        Tensor outT = new(xu.Shape, DType.F32);
        backend.LayerNorm(outT, xu, _afterNormW!, _afterNormB!, EmbedNormEps);
        xu.Dispose();
        return outT;
    }

    /// <summary>Linear(in→out) → LayerNorm(eps 1e-5) → scale by √outputSize (the ESPnet rel-pos xscale).</summary>
    private Tensor Embed(IBackend backend, Tensor input, int inDim, Tensor linW, Tensor? linB, Tensor lnW, Tensor lnB)
    {
        int t = (int)input.Shape[1];
        Tensor lin = WhisperOps.ProjectLinear(backend, input, linW, linB, 1, t, inDim, _outputSize);
        Tensor normed = new(lin.Shape, DType.F32);
        backend.LayerNorm(normed, lin, lnW, lnB, EmbedNormEps);
        lin.Dispose();
        float scale = MathF.Sqrt(_outputSize);
        float* np = (float*)normed.DataPointer;
        long n = normed.ElementCount;
        for (long i = 0; i < n; i++) np[i] *= scale;
        return normed;
    }

    /// <summary>PreLookaheadLayer: transpose → right-pad(k1-1) → conv1 → LeakyReLU → left-pad(k2-1) → conv2 → transpose, added to the input residual. All on a channels-first <c>[1, C, T]</c> view.</summary>
    private Tensor PreLookahead(IBackend backend, Tensor seq, int t)
    {
        int c = _outputSize;
        int k1 = (int)_plConv1W!.Shape[2];
        int k2 = (int)_plConv2W!.Shape[2];

        Tensor chFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(chFirst, seq, t, c);

        // conv1 (kernel k1) with right pad k1-1 keeps length t.
        Tensor c1 = new(new TensorShape(1, c, t), DType.F32);
        backend.Conv1d(c1, chFirst, _plConv1W!, _plConv1B, stride: 1, padLeft: 0, padRight: k1 - 1, dilation: 1, groups: 1);
        chFirst.Dispose();
        backend.LeakyRelu(c1, c1, 0.01f);

        // conv2 (kernel k2) with left pad k2-1 keeps length t.
        Tensor c2 = new(new TensorShape(1, c, t), DType.F32);
        backend.Conv1d(c2, c1, _plConv2W!, _plConv2B, stride: 1, padLeft: k2 - 1, padRight: 0, dilation: 1, groups: 1);
        c1.Dispose();

        Tensor outSeq = new(new TensorShape(1, t, c), DType.F32);
        backend.Transpose2D(outSeq, c2, c, t);
        c2.Dispose();

        // Residual add (channels-last).
        float* op = (float*)outSeq.DataPointer;
        float* sp = (float*)seq.DataPointer;
        long n = outSeq.ElementCount;
        for (long i = 0; i < n; i++) op[i] += sp[i];
        return outSeq;
    }

    /// <summary>Upsample1D: nearest-neighbour ×ratio upsample, left-pad by ratio·2, then conv (kernel k, stride 1, no pad) → exactly ratio·T frames. Channels-last in/out.</summary>
    private Tensor Upsample(IBackend backend, Tensor seq, int t)
    {
        int c = _outputSize;
        int ratio = TokenMelRatio;
        int k = (int)_upConvW!.Shape[2];

        // Channels-first [1, C, T].
        Tensor chFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(chFirst, seq, t, c);

        // Nearest upsample ×ratio then left-pad by ratio·2 → length ratio·T + ratio·2.
        int padL = ratio * 2;
        int upLen = ratio * t;
        int paddedLen = upLen + padL;
        Tensor padded = new(new TensorShape(1, c, paddedLen), DType.F32);
        float* pp = (float*)padded.DataPointer;
        float* cf = (float*)chFirst.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            float* dst = pp + (long)ch * paddedLen;
            float* src = cf + (long)ch * t;
            for (int j = 0; j < padL; j++) dst[j] = 0f;
            for (int j = 0; j < upLen; j++) dst[padL + j] = src[j / ratio];   // nearest
        }
        chFirst.Dispose();

        // conv (kernel k, stride 1, no pad) → paddedLen - k + 1 = ratio·T (since padL = ratio·2 = k - 1).
        int outLen = paddedLen - k + 1;
        Tensor convOut = new(new TensorShape(1, c, outLen), DType.F32);
        backend.Conv1d(convOut, padded, _upConvW!, _upConvB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        padded.Dispose();

        Tensor outSeq = new(new TensorShape(1, outLen, c), DType.F32);
        backend.Transpose2D(outSeq, convOut, c, outLen);
        convOut.Dispose();
        return outSeq;
    }

    /// <summary>Builds the ESPnet relative-position table <c>[1, 2T-1, outputSize]</c>: index 0 is relative position +(T-1), descending through 0 to -(T-1). Even dims sin, odd dims cos, with the standard 1/10000^(2k/d) frequencies.</summary>
    private Tensor BuildRelPos(int t)
    {
        int d = _outputSize;
        int len = 2 * t - 1;
        Tensor pe = new(new TensorShape(1, len, d), DType.F32);
        float* p = (float*)pe.DataPointer;
        int half = d / 2;
        for (int i = 0; i < len; i++)
        {
            // i in [0, T-1] → position (T-1-i) (positive, flipped); i in [T, 2T-2] → position -(i-T+1).
            int pos = i < t ? (t - 1 - i) : -(i - t + 1);
            float* row = p + (long)i * d;
            for (int k = 0; k < half; k++)
            {
                double inv = Math.Exp(-(2.0 * k / d) * Math.Log(10000.0));
                double angle = pos * inv;
                row[2 * k] = (float)Math.Sin(angle);
                row[2 * k + 1] = (float)Math.Cos(angle);
            }
        }
        return pe;
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core =
        [
            _embLinW, _embLinB, _embLnW, _embLnB, _plConv1W, _plConv1B, _plConv2W, _plConv2B,
            _upConvW, _upConvB, _upEmbLinW, _upEmbLinB, _upEmbLnW, _upEmbLnB, _afterNormW, _afterNormB,
        ];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (RelPosBlock b in _preBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (RelPosBlock b in _postBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>One CosyVoice 2 relative-position Transformer encoder block (ESPnet ConformerEncoderLayer with no macaron FFN and no conv module): <c>x += RelPosMHSA(norm_mha(x), pos); x += FFN(norm_ff(x))</c>. Sub-layer LayerNorms use eps 1e-12. The FFN is <c>Linear → SiLU → Linear</c> with scale 1.0.</summary>
internal sealed unsafe class RelPosBlock
{
    private const float NormEps = 1e-12f;
    private readonly int _channels, _numHeads, _headDim;

    private Tensor? _mhaNormW, _mhaNormB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _posW, _posBiasU, _posBiasV;
    private Tensor? _ffNormW, _ffNormB, _ffW1, _ffB1, _ffW2, _ffB2;

    public RelPosBlock(int channels, int numHeads)
    {
        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _mhaNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_mha.weight"]);
        _mhaNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_mha.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_q.weight"]); _qB = TryGet(w, $"{prefix}.self_attn.linear_q.bias");
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_k.weight"]); _kB = TryGet(w, $"{prefix}.self_attn.linear_k.bias");
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_v.weight"]); _vB = TryGet(w, $"{prefix}.self_attn.linear_v.bias");
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_out.weight"]); _oB = TryGet(w, $"{prefix}.self_attn.linear_out.bias");
        _posW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_pos.weight"]);   // no bias
        _posBiasU = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.pos_bias_u"]);       // [H, d]
        _posBiasV = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.pos_bias_v"]);       // [H, d]

        _ffNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_ff.weight"]);
        _ffNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_ff.bias"]);
        _ffW1 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_1.weight"]); _ffB1 = TryGet(w, $"{prefix}.feed_forward.w_1.bias");
        _ffW2 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_2.weight"]); _ffB2 = TryGet(w, $"{prefix}.feed_forward.w_2.bias");
    }

    public Tensor Forward(IBackend backend, Tensor seq, Tensor posEmb, int t)
    {
        int c = _channels;
        Tensor x = new(seq.Shape, DType.F32);
        backend.CopyTo(x, seq);

        // Self-attention sub-layer.
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _mhaNormW!, _mhaNormB!, NormEps);
        Tensor att = RelPosAttention(backend, normed, posEmb, t, c);
        normed.Dispose();
        Add(x, att); att.Dispose();

        // Feed-forward sub-layer (scale 1.0).
        Tensor normed2 = new(x.Shape, DType.F32);
        backend.LayerNorm(normed2, x, _ffNormW!, _ffNormB!, NormEps);
        int ffDim = (int)_ffW1!.Shape[0];
        Tensor h = WhisperOps.ProjectLinear(backend, normed2, _ffW1!, _ffB1, 1, t, c, ffDim);
        normed2.Dispose();
        backend.Silu(h, h);
        Tensor ff = WhisperOps.ProjectLinear(backend, h, _ffW2!, _ffB2, 1, t, ffDim, c);
        h.Dispose();
        Add(x, ff); ff.Dispose();
        return x;
    }

    /// <summary>Transformer-XL relative-position multi-head attention (ESPnet RelPositionMultiHeadedAttention): per head, <c>scores = ((q+u)·kᵀ + rel_shift((q+v)·pᵀ)) / √d</c>, softmax, then <c>·v</c>. <paramref name="posEmb"/> is <c>[1, 2T-1, C]</c>.</summary>
    private Tensor RelPosAttention(IBackend backend, Tensor x, Tensor posEmb, int t, int c)
    {
        int h = _numHeads, d = _headDim, posLen = 2 * t - 1;
        Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, _qB, 1, t, c, c);
        Tensor k = WhisperOps.ProjectLinear(backend, x, _kW!, _kB, 1, t, c, c);
        Tensor v = WhisperOps.ProjectLinear(backend, x, _vW!, _vB, 1, t, c, c);
        Tensor p = WhisperOps.ProjectLinear(backend, posEmb, _posW!, null, 1, posLen, c, c);

        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer, pp = (float*)p.DataPointer;
        float* bu = (float*)_posBiasU!.DataPointer, bv = (float*)_posBiasV!.DataPointer;

        Tensor outMerged = new(new TensorShape(1, t, c), DType.F32);
        float* om = (float*)outMerged.DataPointer;
        float invSqrtD = 1f / MathF.Sqrt(d);
        float[] scores = new float[t];   // reused per (head, query) row

        for (int head = 0; head < h; head++)
        {
            int hOff = head * d;
            float* buH = bu + (long)head * d;
            float* bvH = bv + (long)head * d;
            for (int i = 0; i < t; i++)
            {
                float* qi = qp + (long)i * c + hOff;   // q[i, head, :]
                // matrix_ac[i,j] = (q_i + u)·k_j ; matrix_bd[i,j] = (q_i + v)·p_(T-1-i+j)  (rel_shift).
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    float* kj = kp + (long)j * c + hOff;
                    int posIdx = (t - 1) - i + j;       // rel_shift gather
                    float* pj = pp + (long)posIdx * c + hOff;
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
                // softmax over j.
                float sum = 0f;
                for (int j = 0; j < t; j++) { float e = MathF.Exp(scores[j] - maxS); scores[j] = e; sum += e; }
                float invSum = 1f / sum;
                // out[i, head, :] = Σ_j softmax_j · v_j.
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

    private static void Add(Tensor dst, Tensor src)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] += sp[i];
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _mhaNormW, _mhaNormB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _posW, _posBiasU, _posBiasV,
            _ffNormW, _ffNormB, _ffW1, _ffB1, _ffW2, _ffB2,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
