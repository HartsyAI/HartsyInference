using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>The CosyVoice 2 flow-matching token encoder
/// (<c>cosyvoice/transformer/upsample_encoder.py:UpsampleConformerEncoder</c>). Embedded speech tokens
/// enter at 25 Hz; an <c>input_layer</c> linear lifts them to the model width, a stack of pre-up conformer
/// blocks contextualizes them, a <c>ConvTranspose1d</c> (stride = <see cref="TokenMelRatio"/>, default 2)
/// upsamples 25 Hz → 50 Hz, and a second stack of post-up conformer blocks refines the upsampled stream.
/// The output feeds <c>encoder_proj</c> to produce the mel conditioning <c>μ</c>.
///
/// <para>Each conformer block is the macaron-FFN sandwich:
/// <c>x += ½·FFN(x); x += MHSA(x); x += Conv(x); x += ½·FFN(x); x = LayerNorm(x)</c>, matching the
/// WeNet/ESPnet conformer the FunAudioLLM repo derives from. Self-attention here uses plain scaled
/// dot-product (the v2 flow encoder dropped the learned relative-position bias of v1); the conv module is
/// the standard pointwise→GLU→depthwise→norm→SiLU→pointwise stack.</para></summary>
public sealed unsafe class UpsampleConformerEncoder : IDisposable
{
    public const int TokenMelRatio = 2;

    private readonly int _outputSize;
    private readonly int _numHeads;
    private readonly ConformerBlock[] _preBlocks;
    private readonly ConformerBlock[] _postBlocks;
    private int _disposed;

    private Tensor? _inLinearW, _inLinearB;          // input_size → outputSize
    private Tensor? _upConvW, _upConvB;              // ConvTranspose1d [outputSize, outputSize, k]
    private Tensor? _afterNormW, _afterNormB;        // final LayerNorm (optional)

    public UpsampleConformerEncoder(int outputSize, int numHeads, int numPreBlocks, int numPostBlocks)
    {
        if (outputSize % numHeads != 0)
            throw new ArgumentException($"outputSize {outputSize} must be divisible by numHeads {numHeads}.");
        _outputSize = outputSize;
        _numHeads = numHeads;
        _preBlocks = new ConformerBlock[numPreBlocks];
        _postBlocks = new ConformerBlock[numPostBlocks];
        for (int i = 0; i < numPreBlocks; i++) _preBlocks[i] = new ConformerBlock(outputSize, numHeads);
        for (int i = 0; i < numPostBlocks; i++) _postBlocks[i] = new ConformerBlock(outputSize, numHeads);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _inLinearW = WhisperOps.EnsureF32(w[$"{p}embed.out.0.weight"]);
        _inLinearB = w.TryGetValue($"{p}embed.out.0.bias", out Tensor? ib) ? WhisperOps.EnsureF32(ib) : null;
        for (int i = 0; i < _preBlocks.Length; i++)
            _preBlocks[i].LoadWeights(w, $"{p}encoders.{i}");
        _upConvW = WhisperOps.EnsureF32(w[$"{p}up_layer.weight"]);
        _upConvB = w.TryGetValue($"{p}up_layer.bias", out Tensor? ub) ? WhisperOps.EnsureF32(ub) : null;
        for (int i = 0; i < _postBlocks.Length; i++)
            _postBlocks[i].LoadWeights(w, $"{p}up_encoders.{i}");
        if (w.TryGetValue($"{p}after_norm.weight", out Tensor? anw))
        {
            _afterNormW = WhisperOps.EnsureF32(anw);
            _afterNormB = WhisperOps.EnsureF32(w[$"{p}after_norm.bias"]);
        }
    }

    /// <summary>Forwards embedded speech tokens <c>[1, T, inputSize]</c> → upsampled features
    /// <c>[1, T·<see cref="TokenMelRatio"/>, outputSize]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor tokenEmb, int inputSize)
    {
        if (_inLinearW is null) throw new InvalidOperationException("UpsampleConformerEncoder weights not loaded.");
        int t = (int)tokenEmb.Shape[1];

        Tensor x = WhisperOps.ProjectLinear(backend, tokenEmb, _inLinearW!, _inLinearB, 1, t, inputSize, _outputSize);
        for (int i = 0; i < _preBlocks.Length; i++)
        {
            Tensor next = _preBlocks[i].Forward(backend, x, t, _outputSize);
            x.Dispose();
            x = next;
        }

        Tensor up = Upsample(backend, x, t);
        x.Dispose();
        int tUp = (int)up.Shape[1];
        for (int i = 0; i < _postBlocks.Length; i++)
        {
            Tensor next = _postBlocks[i].Forward(backend, up, tUp, _outputSize);
            up.Dispose();
            up = next;
        }

        if (_afterNormW is not null)
        {
            Tensor normed = new(up.Shape, DType.F32);
            backend.LayerNorm(normed, up, _afterNormW!, _afterNormB!, 1e-5f);
            up.Dispose();
            up = normed;
        }
        return up;
    }

    /// <summary>ConvTranspose1d time-upsample (stride = ratio) on a channels-last <c>[1, T, C]</c>
    /// sequence; transposes to channels-first for the conv and back.</summary>
    private Tensor Upsample(IBackend backend, Tensor seq, int t)
    {
        int c = _outputSize;
        Tensor chFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(chFirst, seq, t, c);

        int k = (int)_upConvW!.Shape[2];
        // ConvTranspose1d output length: (T-1)*stride - 2*pad + (k-1) + 1. Pad to keep ratio·T frames.
        int pad = (k - TokenMelRatio) / 2;
        int outLen = (t - 1) * TokenMelRatio - 2 * pad + (k - 1) + 1;
        Tensor upCh = new(new TensorShape(1, c, outLen), DType.F32);
        backend.ConvTranspose1d(upCh, chFirst, _upConvW!, _upConvB, stride: TokenMelRatio, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
        chFirst.Dispose();

        int tUp = t * TokenMelRatio;
        Tensor up = new(new TensorShape(1, tUp, c), DType.F32);
        // Transpose back, cropping/clamping to exactly ratio·T frames.
        Tensor cropped = upCh;
        if (outLen != tUp)
        {
            cropped = new(new TensorShape(1, c, tUp), DType.F32);
            float* sp = (float*)upCh.DataPointer;
            float* dp = (float*)cropped.DataPointer;
            int copy = Math.Min(outLen, tUp);
            for (int ch = 0; ch < c; ch++)
                Buffer.MemoryCopy(sp + (long)ch * outLen, dp + (long)ch * tUp, (long)tUp * 4, (long)copy * 4);
            upCh.Dispose();
        }
        backend.Transpose2D(up, cropped, c, tUp);
        cropped.Dispose();
        return up;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_inLinearW, _inLinearB, _upConvW, _upConvB, _afterNormW, _afterNormB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (ConformerBlock b in _preBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (ConformerBlock b in _postBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>One macaron-FFN conformer block operating on a channels-last <c>[1, T, C]</c> sequence:
/// half-step FFN → MHSA → conv module → half-step FFN → final LayerNorm, each sub-layer residual.</summary>
internal sealed unsafe class ConformerBlock
{
    private const float FfnScale = 0.5f;

    private readonly int _channels, _numHeads, _headDim;

    private Tensor? _ff1NormW, _ff1NormB, _ff1W1, _ff1B1, _ff1W2, _ff1B2;
    private Tensor? _attnNormW, _attnNormB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
    private Tensor? _convNormW, _convNormB, _pw1W, _pw1B, _dwW, _dwB, _convLnW, _convLnB, _pw2W, _pw2B;
    private Tensor? _ff2NormW, _ff2NormB, _ff2W1, _ff2B1, _ff2W2, _ff2B2;
    private Tensor? _finalNormW, _finalNormB;

    public ConformerBlock(int channels, int numHeads)
    {
        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _ff1NormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_ff_macaron.weight"]);
        _ff1NormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_ff_macaron.bias"]);
        _ff1W1 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward_macaron.w_1.weight"]); _ff1B1 = TryBias(w, $"{prefix}.feed_forward_macaron.w_1.bias");
        _ff1W2 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward_macaron.w_2.weight"]); _ff1B2 = TryBias(w, $"{prefix}.feed_forward_macaron.w_2.bias");

        _attnNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_mha.weight"]);
        _attnNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_mha.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_q.weight"]); _qB = TryBias(w, $"{prefix}.self_attn.linear_q.bias");
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_k.weight"]); _kB = TryBias(w, $"{prefix}.self_attn.linear_k.bias");
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_v.weight"]); _vB = TryBias(w, $"{prefix}.self_attn.linear_v.bias");
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_out.weight"]); _oB = TryBias(w, $"{prefix}.self_attn.linear_out.bias");

        _convNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_conv.weight"]);
        _convNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_conv.bias"]);
        _pw1W = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.pointwise_conv1.weight"]); _pw1B = TryBias(w, $"{prefix}.conv_module.pointwise_conv1.bias");
        _dwW = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.depthwise_conv.weight"]); _dwB = TryBias(w, $"{prefix}.conv_module.depthwise_conv.bias");
        _convLnW = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.norm.weight"]);
        _convLnB = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.norm.bias"]);
        _pw2W = WhisperOps.EnsureF32(w[$"{prefix}.conv_module.pointwise_conv2.weight"]); _pw2B = TryBias(w, $"{prefix}.conv_module.pointwise_conv2.bias");

        _ff2NormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_ff.weight"]);
        _ff2NormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_ff.bias"]);
        _ff2W1 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_1.weight"]); _ff2B1 = TryBias(w, $"{prefix}.feed_forward.w_1.bias");
        _ff2W2 = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_2.weight"]); _ff2B2 = TryBias(w, $"{prefix}.feed_forward.w_2.bias");

        _finalNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_final.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_final.bias"]);
    }

    private static Tensor? TryBias(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;

    /// <summary>Forwards a <c>[1, T, C]</c> sequence; returns a new owned <c>[1, T, C]</c> tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor seq, int t, int c)
    {
        Tensor x = new(seq.Shape, DType.F32);
        backend.CopyTo(x, seq);

        ApplyFfn(backend, x, t, c, _ff1NormW!, _ff1NormB!, _ff1W1!, _ff1B1, _ff1W2!, _ff1B2, FfnScale);
        ApplyAttention(backend, x, t, c);
        ApplyConvModule(backend, x, t, c);
        ApplyFfn(backend, x, t, c, _ff2NormW!, _ff2NormB!, _ff2W1!, _ff2B1, _ff2W2!, _ff2B2, FfnScale);

        Tensor outT = new(x.Shape, DType.F32);
        backend.LayerNorm(outT, x, _finalNormW!, _finalNormB!, 1e-5f);
        x.Dispose();
        return outT;
    }

    private static void ApplyFfn(IBackend backend, Tensor x, int t, int c, Tensor normW, Tensor normB,
        Tensor w1, Tensor? b1, Tensor w2, Tensor? b2, float scale)
    {
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, normW, normB, 1e-5f);
        int ffDim = (int)w1.Shape[0];
        Tensor h = WhisperOps.ProjectLinear(backend, normed, w1, b1, 1, t, c, ffDim);
        normed.Dispose();
        backend.Silu(h, h);
        Tensor o = WhisperOps.ProjectLinear(backend, h, w2, b2, 1, t, ffDim, c);
        h.Dispose();
        AddScaled(x, o, scale);
        o.Dispose();
    }

    private void ApplyAttention(IBackend backend, Tensor x, int t, int c)
    {
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _attnNormW!, _attnNormB!, 1e-5f);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW!, _qB, 1, t, c, c);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW!, _kB, 1, t, c, c);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW!, _vB, 1, t, c, c);
        normed.Dispose();

        Tensor qH = new(new TensorShape(1, _numHeads, t, _headDim), DType.F32);
        Tensor kH = new(new TensorShape(1, _numHeads, t, _headDim), DType.F32);
        Tensor vH = new(new TensorShape(1, _numHeads, t, _headDim), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qH, q, 1, t, _numHeads, _headDim);
        WhisperOps.ReshapeToMultiHead4D(kH, k, 1, t, _numHeads, _headDim);
        WhisperOps.ReshapeToMultiHead4D(vH, v, 1, t, _numHeads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn = new(new TensorShape(1, _numHeads, t, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, mask: null, 1f / MathF.Sqrt(_headDim));
        qH.Dispose(); kH.Dispose(); vH.Dispose();

        Tensor merged = new(new TensorShape(1, t, c), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(merged, attn, 1, t, _numHeads, _headDim);
        attn.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, merged, _oW!, _oB, 1, t, c, c);
        merged.Dispose();
        AddScaled(x, o, 1f);
        o.Dispose();
    }

    /// <summary>Conformer conv module: LN → pointwise(2C) → GLU → depthwise(k) → LayerNorm → SiLU →
    /// pointwise(C), residual into <paramref name="x"/>. Runs on a channels-first view internally.</summary>
    private void ApplyConvModule(IBackend backend, Tensor x, int t, int c)
    {
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _convNormW!, _convNormB!, 1e-5f);

        // [1, T, C] → [1, C, T].
        Tensor chFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(chFirst, normed, t, c);
        normed.Dispose();

        // pointwise_conv1: C → 2C, then GLU halves back to C.
        Tensor pw1 = new(new TensorShape(1, 2 * c, t), DType.F32);
        backend.Conv1d(pw1, chFirst, _pw1W!, _pw1B, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        chFirst.Dispose();
        Tensor glu = new(new TensorShape(1, c, t), DType.F32);
        Glu(glu, pw1, c, t);
        pw1.Dispose();

        // depthwise_conv: groups = C, kernel k, "same" padding.
        int k = (int)_dwW!.Shape[2];
        int pad = (k - 1) / 2;
        Tensor dw = new(new TensorShape(1, c, t), DType.F32);
        backend.Conv1d(dw, glu, _dwW!, _dwB, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: c);
        glu.Dispose();

        // norm (LayerNorm over channels) → SiLU. Transpose to channels-last for the LN, then back.
        Tensor dwSeq = new(new TensorShape(1, t, c), DType.F32);
        backend.Transpose2D(dwSeq, dw, c, t);
        dw.Dispose();
        Tensor dwNorm = new(dwSeq.Shape, DType.F32);
        backend.LayerNorm(dwNorm, dwSeq, _convLnW!, _convLnB!, 1e-5f);
        dwSeq.Dispose();
        backend.Silu(dwNorm, dwNorm);
        Tensor dwChFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(dwChFirst, dwNorm, t, c);
        dwNorm.Dispose();

        // pointwise_conv2: C → C.
        Tensor pw2 = new(new TensorShape(1, c, t), DType.F32);
        backend.Conv1d(pw2, dwChFirst, _pw2W!, _pw2B, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        dwChFirst.Dispose();

        // Back to channels-last and residual add.
        Tensor outSeq = new(new TensorShape(1, t, c), DType.F32);
        backend.Transpose2D(outSeq, pw2, c, t);
        pw2.Dispose();
        AddScaled(x, outSeq, 1f);
        outSeq.Dispose();
    }

    /// <summary>GLU over a channels-first <c>[1, 2C, T]</c>: out[c] = in[c] · sigmoid(in[c+C]).</summary>
    private static void Glu(Tensor dst, Tensor src, int c, int t)
    {
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)dst.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            long aOff = (long)ch * t;
            long bOff = (long)(ch + c) * t;
            for (int j = 0; j < t; j++)
            {
                float gate = 1f / (1f + MathF.Exp(-sp[bOff + j]));
                dp[aOff + j] = sp[aOff + j] * gate;
            }
        }
    }

    private static void AddScaled(Tensor dst, Tensor src, float scale)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        if (scale == 1f)
            for (long i = 0; i < n; i++) dp[i] += sp[i];
        else
            for (long i = 0; i < n; i++) dp[i] += scale * sp[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _ff1NormW, _ff1NormB, _ff1W1, _ff1B1, _ff1W2, _ff1B2,
            _attnNormW, _attnNormB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB,
            _convNormW, _convNormB, _pw1W, _pw1B, _dwW, _dwB, _convLnW, _convLnB, _pw2W, _pw2B,
            _ff2NormW, _ff2NormB, _ff2W1, _ff2B1, _ff2W2, _ff2B2,
            _finalNormW, _finalNormB,
        ];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
