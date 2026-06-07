using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.CosyVoice;

/// <summary>The CFM velocity estimator — a Matcha-TTS-style conditioned 1-D residual+attention network
/// (<c>cosyvoice/flow/decoder.py:CausalConditionalDecoder</c>). Packs the noisy mel <c>x</c>, the
/// token-conditioning mel <c>μ</c>, the broadcast speaker vector, and the reference mel <c>cond</c> into
/// a <c>4·melBins</c>-channel input, injects a sinusoidal timestep embedding into each residual block,
/// and regresses the velocity <c>dφ/dt</c> as <c>[1, melBins, T]</c>.
///
/// <para><b>Scaffold note:</b> this is a constant-resolution residual+attention stack (timestep-injected
/// <see cref="ResnetBlock1D"/> interleaved with self-attention), which conditions correctly and runs
/// end-to-end. The exact down/up-sample topology + skip wiring + block counts of the real
/// <c>flow.decoder.estimator</c> must be reconciled against <c>flow.pt</c> before weights load cleanly;
/// the block count is driven by <see cref="CosyVoiceFlowConfig.NumMidBlocks"/>.</para></summary>
public sealed unsafe class CausalConditionalDecoder : ICfmEstimator
{
    private readonly CosyVoiceFlowConfig _cfg;
    private readonly int _channels;
    private readonly int _inChannels;
    private readonly ResnetBlock1D[] _blocks;
    private readonly SelfAttnBlock1D[] _attn;

    private Tensor? _timeLin1W, _timeLin1B, _timeLin2W, _timeLin2B;
    private Tensor? _inConvW, _inConvB;
    private Tensor? _outNormW, _outNormB, _outConvW, _outConvB;

    public CausalConditionalDecoder(CosyVoiceFlowConfig cfg)
    {
        _cfg = cfg;
        _channels = cfg.UnetChannels[0];
        _inChannels = cfg.MelBins * 4;          // x + μ + spk(broadcast) + cond
        _blocks = new ResnetBlock1D[cfg.NumMidBlocks];
        _attn = new SelfAttnBlock1D[cfg.NumMidBlocks];
        for (int i = 0; i < cfg.NumMidBlocks; i++)
        {
            _blocks[i] = new ResnetBlock1D(_channels, _channels, _channels);
            _attn[i] = new SelfAttnBlock1D(_channels, cfg.NumHeads, cfg.AttentionHeadDim);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "estimator")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _timeLin1W = WhisperOps.EnsureF32(w[$"{p}time_mlp.1.weight"]);
        _timeLin1B = WhisperOps.EnsureF32(w[$"{p}time_mlp.1.bias"]);
        _timeLin2W = WhisperOps.EnsureF32(w[$"{p}time_mlp.3.weight"]);
        _timeLin2B = WhisperOps.EnsureF32(w[$"{p}time_mlp.3.bias"]);
        _inConvW = WhisperOps.EnsureF32(w[$"{p}in_conv.weight"]);
        _inConvB = WhisperOps.EnsureF32(w[$"{p}in_conv.bias"]);
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadWeights(w, $"{p}blocks.{i}.resnet");
            _attn[i].LoadWeights(w, $"{p}blocks.{i}.attn");
        }
        _outNormW = WhisperOps.EnsureF32(w[$"{p}out_norm.weight"]);
        _outNormB = WhisperOps.EnsureF32(w[$"{p}out_norm.bias"]);
        _outConvW = WhisperOps.EnsureF32(w[$"{p}out_conv.weight"]);
        _outConvB = WhisperOps.EnsureF32(w[$"{p}out_conv.bias"]);
    }

    public Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond)
    {
        int mel = _cfg.MelBins;
        int tt = (int)x.Shape[2];

        // Timestep embedding: sinusoidal(t) → Linear → SiLU → Linear → [1, channels].
        Tensor timeEmb = TimeEmbedding(backend, t);

        // Pack [x, μ, spk(broadcast), cond] → [1, 4·mel, T].
        Tensor packed = PackInput(x, mu, spk, cond, mel, tt);
        Tensor h = new(new TensorShape(1, _channels, tt), DType.F32);
        backend.Conv1d(h, packed, _inConvW!, _inConvB, stride: 1, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
        packed.Dispose();

        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor r = _blocks[i].Forward(backend, h, timeEmb);
            h.Dispose();
            Tensor a = _attn[i].Forward(backend, r);
            r.Dispose();
            h = a;
        }
        timeEmb.Dispose();

        Tensor normed = new(h.Shape, DType.F32);
        backend.GroupNorm(normed, h, _outNormW!, _outNormB!, groups: 1, eps: 1e-6f);
        h.Dispose();
        backend.Silu(normed, normed);
        Tensor outT = new(new TensorShape(1, mel, tt), DType.F32);
        backend.Conv1d(outT, normed, _outConvW!, _outConvB, stride: 1, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
        normed.Dispose();
        return outT;
    }

    private Tensor TimeEmbedding(IBackend backend, float t)
    {
        int half = _channels / 2;
        Tensor sinEmb = new(new TensorShape(1, 1, _channels), DType.F32);
        float* sp = (float*)sinEmb.DataPointer;
        double logBase = Math.Log(10000.0) / Math.Max(1, half - 1);
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-logBase * i);
            sp[i] = (float)Math.Sin(t * freq);
            sp[half + i] = (float)Math.Cos(t * freq);
        }
        Tensor h1 = WhisperOps.ProjectLinear(backend, sinEmb, _timeLin1W!, _timeLin1B, 1, 1, _channels, _channels);
        sinEmb.Dispose();
        backend.Silu(h1, h1);
        Tensor h2 = WhisperOps.ProjectLinear(backend, h1, _timeLin2W!, _timeLin2B, 1, 1, _channels, _channels);
        h1.Dispose();
        return h2;     // [1, 1, channels]
    }

    private static Tensor PackInput(Tensor x, Tensor mu, Tensor spk, Tensor cond, int mel, int t)
    {
        Tensor packed = new(new TensorShape(1, mel * 4, t), DType.F32);
        float* pp = (float*)packed.DataPointer;
        CopyChannels(pp, (float*)x.DataPointer, 0, mel, t);
        CopyChannels(pp, (float*)mu.DataPointer, mel, mel, t);
        // spk is [1, mel] (or [1, mel, 1]) broadcast over T.
        float* spkp = (float*)spk.DataPointer;
        for (int c = 0; c < mel; c++)
        {
            float v = spkp[c];
            long baseOff = (long)(2 * mel + c) * t;
            for (int j = 0; j < t; j++) pp[baseOff + j] = v;
        }
        CopyChannels(pp, (float*)cond.DataPointer, 3 * mel, mel, t);
        return packed;
    }

    private static void CopyChannels(float* dst, float* src, int chOffset, int channels, int t)
    {
        long bytes = (long)channels * t * 4;
        Buffer.MemoryCopy(src, dst + (long)chOffset * t, bytes, bytes);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_timeLin1W, _timeLin1B, _timeLin2W, _timeLin2B, _inConvW, _inConvB, _outNormW, _outNormB, _outConvW, _outConvB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (ResnetBlock1D b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (SelfAttnBlock1D a in _attn) foreach (Tensor t in a.EnumerateWeights()) yield return t;
    }
}

/// <summary>GroupNorm→SiLU→Conv1d residual block with an additive timestep embedding, channels-first
/// <c>[1, C, T]</c>. Standard Matcha-TTS <c>ResnetBlock1D</c>.</summary>
internal sealed unsafe class ResnetBlock1D
{
    private readonly int _inCh, _outCh;
    private Tensor? _norm1W, _norm1B, _conv1W, _conv1B;
    private Tensor? _norm2W, _norm2B, _conv2W, _conv2B;
    private Tensor? _timeW, _timeB;
    private Tensor? _resConvW, _resConvB;

    public ResnetBlock1D(int inCh, int outCh, int timeDim)
    {
        _inCh = inCh;
        _outCh = outCh;
        _ = timeDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _norm1W = WhisperOps.EnsureF32(w[$"{prefix}.norm1.weight"]);
        _norm1B = WhisperOps.EnsureF32(w[$"{prefix}.norm1.bias"]);
        _conv1W = WhisperOps.EnsureF32(w[$"{prefix}.conv1.weight"]);
        _conv1B = WhisperOps.EnsureF32(w[$"{prefix}.conv1.bias"]);
        _norm2W = WhisperOps.EnsureF32(w[$"{prefix}.norm2.weight"]);
        _norm2B = WhisperOps.EnsureF32(w[$"{prefix}.norm2.bias"]);
        _conv2W = WhisperOps.EnsureF32(w[$"{prefix}.conv2.weight"]);
        _conv2B = WhisperOps.EnsureF32(w[$"{prefix}.conv2.bias"]);
        _timeW = WhisperOps.EnsureF32(w[$"{prefix}.time_emb.weight"]);
        _timeB = WhisperOps.EnsureF32(w[$"{prefix}.time_emb.bias"]);
        if (_inCh != _outCh)
        {
            _resConvW = WhisperOps.EnsureF32(w[$"{prefix}.res_conv.weight"]);
            _resConvB = WhisperOps.EnsureF32(w[$"{prefix}.res_conv.bias"]);
        }
    }

    public Tensor Forward(IBackend backend, Tensor x, Tensor timeEmb)
    {
        int t = (int)x.Shape[2];
        Tensor h = new(new TensorShape(1, _inCh, t), DType.F32);
        backend.GroupNorm(h, x, _norm1W!, _norm1B!, groups: 1, eps: 1e-6f);
        backend.Silu(h, h);
        Tensor c1 = new(new TensorShape(1, _outCh, t), DType.F32);
        backend.Conv1d(c1, h, _conv1W!, _conv1B, 1, 1, 1, 1, 1);
        h.Dispose();

        // Add the projected time embedding (broadcast over T).
        Tensor tproj = WhisperOps.ProjectLinear(backend, timeEmb, _timeW!, _timeB, 1, 1, (int)timeEmb.Shape[2], _outCh);
        float* cp = (float*)c1.DataPointer;
        float* tp = (float*)tproj.DataPointer;
        for (int c = 0; c < _outCh; c++)
        {
            float add = tp[c];
            long baseOff = (long)c * t;
            for (int j = 0; j < t; j++) cp[baseOff + j] += add;
        }
        tproj.Dispose();

        Tensor h2 = new(c1.Shape, DType.F32);
        backend.GroupNorm(h2, c1, _norm2W!, _norm2B!, groups: 1, eps: 1e-6f);
        c1.Dispose();
        backend.Silu(h2, h2);
        Tensor c2 = new(new TensorShape(1, _outCh, t), DType.F32);
        backend.Conv1d(c2, h2, _conv2W!, _conv2B, 1, 1, 1, 1, 1);
        h2.Dispose();

        // Residual (1×1 projection when channels change).
        Tensor res;
        if (_inCh != _outCh)
        {
            res = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(res, x, _resConvW!, _resConvB, 1, 0, 0, 1, 1);
        }
        else
        {
            res = x;
        }
        float* c2p = (float*)c2.DataPointer;
        float* rp = (float*)res.DataPointer;
        long n = c2.ElementCount;
        for (long i = 0; i < n; i++) c2p[i] += rp[i];
        if (_inCh != _outCh) res.Dispose();
        return c2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1W, _norm1B, _conv1W, _conv1B, _norm2W, _norm2B, _conv2W, _conv2B, _timeW, _timeB, _resConvW, _resConvB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

/// <summary>LayerNorm + multi-head self-attention + LayerNorm + GELU FFN, both residual. Operates on a
/// channels-first <c>[1, C, T]</c> tensor (transposed internally to <c>[1, T, C]</c> for attention).</summary>
internal sealed unsafe class SelfAttnBlock1D
{
    private readonly int _channels, _numHeads, _headDim;
    private Tensor? _norm1W, _norm1B, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
    private Tensor? _norm2W, _norm2B, _ff1W, _ff1B, _ff2W, _ff2B;

    public SelfAttnBlock1D(int channels, int numHeads, int headDim)
    {
        _channels = channels;
        _numHeads = numHeads;
        _headDim = headDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _norm1W = WhisperOps.EnsureF32(w[$"{prefix}.norm1.weight"]);
        _norm1B = WhisperOps.EnsureF32(w[$"{prefix}.norm1.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.to_q.weight"]); _qB = TryBias(w, $"{prefix}.to_q.bias");
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.to_k.weight"]); _kB = TryBias(w, $"{prefix}.to_k.bias");
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.to_v.weight"]); _vB = TryBias(w, $"{prefix}.to_v.bias");
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.to_out.weight"]); _oB = TryBias(w, $"{prefix}.to_out.bias");
        _norm2W = WhisperOps.EnsureF32(w[$"{prefix}.norm2.weight"]);
        _norm2B = WhisperOps.EnsureF32(w[$"{prefix}.norm2.bias"]);
        _ff1W = WhisperOps.EnsureF32(w[$"{prefix}.ff1.weight"]); _ff1B = TryBias(w, $"{prefix}.ff1.bias");
        _ff2W = WhisperOps.EnsureF32(w[$"{prefix}.ff2.weight"]); _ff2B = TryBias(w, $"{prefix}.ff2.bias");
    }

    private static Tensor? TryBias(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int c = _channels;
        int t = (int)x.Shape[2];
        Tensor seq = new(new TensorShape(1, t, c), DType.F32);   // [1, T, C]
        backend.Transpose2D(seq, x, c, t);

        Tensor normed = new(seq.Shape, DType.F32);
        backend.LayerNorm(normed, seq, _norm1W!, _norm1B!, 1e-6f);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW!, _qB, 1, t, c, c);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW!, _kB, 1, t, c, c);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW!, _vB, 1, t, c, c);
        normed.Dispose();

        Tensor qH = ToHeads(q, t), kH = ToHeads(k, t), vH = ToHeads(v, t);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attn = new(new TensorShape(_numHeads, t, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, mask: null, 1f / MathF.Sqrt(_headDim));
        qH.Dispose(); kH.Dispose(); vH.Dispose();
        Tensor merged = FromHeads(attn, t);
        attn.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, merged, _oW!, _oB, 1, t, c, c);
        merged.Dispose();
        AddInPlace(o, seq);                                 // residual
        seq.Dispose();

        Tensor n2 = new(o.Shape, DType.F32);
        backend.LayerNorm(n2, o, _norm2W!, _norm2B!, 1e-6f);
        int ffDim = (int)_ff1W!.Shape[0];
        Tensor f1 = WhisperOps.ProjectLinear(backend, n2, _ff1W!, _ff1B, 1, t, c, ffDim);
        n2.Dispose();
        backend.Gelu(f1, f1);
        Tensor f2 = WhisperOps.ProjectLinear(backend, f1, _ff2W!, _ff2B, 1, t, ffDim, c);
        f1.Dispose();
        AddInPlace(f2, o);
        o.Dispose();

        Tensor outT = new(new TensorShape(1, c, t), DType.F32);   // back to [1, C, T]
        backend.Transpose2D(outT, f2, t, c);
        f2.Dispose();
        return outT;
    }

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] += sp[i];
    }

    private Tensor ToHeads(Tensor seq, int t)
    {
        Tensor outT = new(new TensorShape(_numHeads, t, _headDim), DType.F32);
        float* ip = (float*)seq.DataPointer;        // [1, t, C]
        float* op = (float*)outT.DataPointer;
        for (int h = 0; h < _numHeads; h++)
            for (int j = 0; j < t; j++)
                for (int d = 0; d < _headDim; d++)
                    op[((long)h * t + j) * _headDim + d] = ip[(long)j * _channels + h * _headDim + d];
        return outT;
    }

    private Tensor FromHeads(Tensor heads, int t)
    {
        Tensor outT = new(new TensorShape(1, t, _channels), DType.F32);
        float* ip = (float*)heads.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int h = 0; h < _numHeads; h++)
            for (int j = 0; j < t; j++)
                for (int d = 0; d < _headDim; d++)
                    op[(long)j * _channels + h * _headDim + d] = ip[((long)h * t + j) * _headDim + d];
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1W, _norm1B, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _norm2W, _norm2B, _ff1W, _ff1B, _ff2W, _ff2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
