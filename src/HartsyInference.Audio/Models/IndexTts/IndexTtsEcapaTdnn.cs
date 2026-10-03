using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>ECAPA-TDNN speaker embedding (Desplanques et al. 2020), as embedded inside IndexTTS-1.5's BigVGAN
/// generator (<c>generator.speaker_encoder</c>) to compute the d-vector that conditions every upsampling stage —
/// a SEPARATE network from the GPT's Conformer-Perceiver speech-conditioning path
/// (<see cref="IndexTtsSpeakerEncoder"/>), confirmed by both the checkpoint's distinct key prefixes and the real
/// <c>BigVGAN.forward(x, mel_ref)</c> source.</summary>
/// <remarks>SpeechBrain's standard topology: an initial TDNN block, three SE-Res2Net blocks, multi-layer feature
/// aggregation, attentive statistics pooling, and a final 1×1 conv — exactly the shapes in IndexTTS's real
/// <c>bigvgan_generator.pth</c> (input 100 mel bands, channels <c>[512,512,512,512,1536]</c>, output 512-dim).
/// All convolutions and the pooling run host-side: this runs once per reference clip, not per generated frame.</remarks>
internal sealed unsafe class IndexTtsEcapaTdnn : IDisposable
{
    private static readonly int[] Channels = [512, 512, 512, 512, 1536];
    private static readonly int[] KernelSizes = [5, 3, 3, 3, 1];
    private static readonly int[] Dilations = [1, 2, 3, 4, 1];
    private const int Res2NetScale = 8;
    private const int SeChannels = 128;
    private const int AttentionChannels = 128;
    private const int InputMels = 100;
    private const int LinNeurons = 512;

    private TdnnBlock? _block0;
    private SeRes2NetBlock? _block1, _block2, _block3;
    private TdnnBlock? _mfa;
    private AttentiveStatsPooling? _asp;
    private BatchNorm1d? _aspBn;
    private Tensor? _fcW, _fcB;
    private int _disposed;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _block0 = TdnnBlock.Load(w, $"{prefix}.blocks.0", InputMels, Channels[0]);
        _block1 = SeRes2NetBlock.Load(w, $"{prefix}.blocks.1", Channels[0], Channels[1], KernelSizes[1], Dilations[1]);
        _block2 = SeRes2NetBlock.Load(w, $"{prefix}.blocks.2", Channels[1], Channels[2], KernelSizes[2], Dilations[2]);
        _block3 = SeRes2NetBlock.Load(w, $"{prefix}.blocks.3", Channels[2], Channels[3], KernelSizes[3], Dilations[3]);
        _mfa = TdnnBlock.Load(w, $"{prefix}.mfa", Channels[0] * 3, Channels[4]);
        _asp = AttentiveStatsPooling.Load(w, $"{prefix}.asp", Channels[4], AttentionChannels);
        _aspBn = BatchNorm1d.Load(w, $"{prefix}.asp_bn", Channels[4] * 2);
        _fcW = WhisperOps.EnsureF32(w[$"{prefix}.fc.conv.weight"]);
        _fcB = WhisperOps.EnsureF32(w[$"{prefix}.fc.conv.bias"]);
    }

    /// <summary>Computes the 512-dim speaker embedding from a mel spectrogram <c>[1, T, 100]</c> (channels-last).
    /// Returns <c>[1, 512, 1]</c> (channels-first, time-broadcastable) — the shape BigVGAN's <c>cond_layer</c>/
    /// <c>conds[i]</c> 1×1 convs expect.</summary>
    public Tensor Forward(IBackend backend, Tensor mel, int t)
    {
        if (_block0 is null) throw new InvalidOperationException("IndexTtsEcapaTdnn weights not loaded.");
        Tensor chFirst = new(new TensorShape(1, InputMels, t), DType.F32);
        backend.Transpose2D(chFirst, mel, t, InputMels);

        Tensor x0 = _block0!.Forward(backend, chFirst, t);
        chFirst.Dispose();
        Tensor x1 = _block1!.Forward(backend, x0, t);
        Tensor x2 = _block2!.Forward(backend, x1, t);
        Tensor x3 = _block3!.Forward(backend, x2, t);
        x0.Dispose();

        Tensor cat = new(new TensorShape(1, Channels[0] * 3, t), DType.F32);
        ConcatChannels(cat, [x1, x2, x3], t);
        x1.Dispose(); x2.Dispose(); x3.Dispose();

        Tensor mfaOut = _mfa!.Forward(backend, cat, t);
        cat.Dispose();

        Tensor pooled = _asp!.Forward(backend, mfaOut, t);   // [1, 2*1536, 1]
        mfaOut.Dispose();
        Tensor bnOut = _aspBn!.Forward(pooled);
        pooled.Dispose();

        Tensor emb = new(new TensorShape(1, LinNeurons, 1), DType.F32);
        Conv1x1(emb, bnOut, _fcW!, _fcB!, Channels[4] * 2, LinNeurons);
        bnOut.Dispose();
        return emb;
    }

    private static void ConcatChannels(Tensor dst, Tensor[] parts, int t)
    {
        float* dp = (float*)dst.DataPointer;
        long offsetCh = 0;
        foreach (Tensor part in parts)
        {
            int c = (int)part.Shape[1];
            float* pp = (float*)part.DataPointer;
            for (long i = 0; i < (long)c * t; i++) dp[offsetCh * t + i] = pp[i];
            offsetCh += c;
        }
    }

    private static void Conv1x1(Tensor dst, Tensor src, Tensor weight, Tensor bias, int inCh, int outCh)
    {
        float* sp = (float*)src.DataPointer, dp = (float*)dst.DataPointer, wp = (float*)weight.DataPointer, bp = (float*)bias.DataPointer;
        for (int oc = 0; oc < outCh; oc++)
        {
            float acc = bp[oc];
            float* wRow = wp + (long)oc * inCh;
            for (int ic = 0; ic < inCh; ic++) acc += wRow[ic] * sp[ic];
            dp[oc] = acc;
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _block0!.EnumerateWeights()) yield return t;
        foreach (Tensor t in _block1!.EnumerateWeights()) yield return t;
        foreach (Tensor t in _block2!.EnumerateWeights()) yield return t;
        foreach (Tensor t in _block3!.EnumerateWeights()) yield return t;
        foreach (Tensor t in _mfa!.EnumerateWeights()) yield return t;
        foreach (Tensor t in _asp!.EnumerateWeights()) yield return t;
        foreach (Tensor t in _aspBn!.EnumerateWeights()) yield return t;
        yield return _fcW!; yield return _fcB!;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    /// <summary>Conv1d(groups=1) → ReLU → BatchNorm1d, channels-first <c>[1,C,T]</c> throughout.</summary>
    private sealed class TdnnBlock
    {
        private readonly Tensor _w, _b;
        private readonly BatchNorm1d _norm;
        private readonly int _inCh, _outCh, _kernel, _dilation;

        private TdnnBlock(Tensor w, Tensor b, BatchNorm1d norm, int inCh, int outCh, int kernel, int dilation)
        { _w = w; _b = b; _norm = norm; _inCh = inCh; _outCh = outCh; _kernel = kernel; _dilation = dilation; }

        public static TdnnBlock Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int inCh, int outCh, int kernel = -1, int dilation = 1)
        {
            Tensor cw = WhisperOps.EnsureF32(w[$"{prefix}.conv.conv.weight"]);
            Tensor cb = WhisperOps.EnsureF32(w[$"{prefix}.conv.conv.bias"]);
            int k = kernel > 0 ? kernel : (int)cw.Shape[2];
            return new TdnnBlock(cw, cb, BatchNorm1d.Load(w, $"{prefix}.norm", outCh), inCh, outCh, k, dilation);
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int pad = (_kernel - 1) * _dilation / 2;
            Tensor conv = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(conv, x, _w, _b, stride: 1, padLeft: pad, padRight: pad, dilation: _dilation, groups: 1);
            backend.LeakyRelu(conv, conv, 0f);   // ReLU
            Tensor normed = _norm.Forward(conv);
            conv.Dispose();
            return normed;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            yield return _w; yield return _b;
            foreach (Tensor t in _norm.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Res2Net (scale 8): split into 8 channel groups; group 0 passes through, each later group adds the
    /// previous group's output before a <see cref="TdnnBlock"/> (k=<paramref name="kernel"/>, same channel width).</summary>
    private sealed class Res2NetBlock
    {
        private readonly TdnnBlock[] _blocks;
        private readonly int _scale, _groupCh;

        private Res2NetBlock(TdnnBlock[] blocks, int scale, int groupCh) { _blocks = blocks; _scale = scale; _groupCh = groupCh; }

        public static Res2NetBlock Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int channels, int scale, int kernel, int dilation)
        {
            int groupCh = channels / scale;
            TdnnBlock[] blocks = new TdnnBlock[scale - 1];
            for (int i = 0; i < scale - 1; i++) blocks[i] = TdnnBlock.Load(w, $"{prefix}.blocks.{i}", groupCh, groupCh, kernel, dilation);
            return new Res2NetBlock(blocks, scale, groupCh);
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int c = _groupCh;
            float* xp = (float*)x.DataPointer;
            Tensor outT = new(new TensorShape(1, c * _scale, t), DType.F32);
            float* op = (float*)outT.DataPointer;

            Tensor? prevY = null;
            for (int i = 0; i < _scale; i++)
            {
                Tensor xi = new(new TensorShape(1, c, t), DType.F32);
                float* xip = (float*)xi.DataPointer;
                long srcOff = (long)i * c * t;
                for (long e = 0; e < (long)c * t; e++) xip[e] = xp[srcOff + e];

                Tensor yi;
                if (i == 0) yi = xi;
                else
                {
                    // Res2Net (Gao et al.): group 0 passes through untouched; group 1 feeds its TDNN block
                    // directly (y_1 = K_1(x_1), no addition); only group 2 onward add the previous group's
                    // output before their block (y_i = K_i(x_i + y_{i-1})). Adding y_0 into group 1 here would
                    // corrupt every downstream SE-Res2Net stage's speaker-embedding conditioning.
                    if (i > 1 && prevY is not null)
                    {
                        float* xip2 = (float*)xi.DataPointer;
                        float* pyp = (float*)prevY.DataPointer;
                        for (long e = 0; e < (long)c * t; e++) xip2[e] += pyp[e];
                    }
                    Tensor blockOut = _blocks[i - 1].Forward(backend, xi, t);
                    xi.Dispose();
                    yi = blockOut;
                }
                float* yip = (float*)yi.DataPointer;
                long dstOff = (long)i * c * t;
                for (long e = 0; e < (long)c * t; e++) op[dstOff + e] = yip[e];

                if (i > 0) prevY?.Dispose();
                prevY = yi;
            }
            prevY?.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (TdnnBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Squeeze-excite: global-average-pool over time → conv(C→se) → ReLU → conv(se→C) → sigmoid → scale <paramref name="x"/>.</summary>
    private sealed class SeBlock
    {
        private readonly Tensor _w1, _b1, _w2, _b2;
        private readonly int _inCh, _seCh;

        private SeBlock(Tensor w1, Tensor b1, Tensor w2, Tensor b2, int inCh, int seCh) { _w1 = w1; _b1 = b1; _w2 = w2; _b2 = b2; _inCh = inCh; _seCh = seCh; }

        public static SeBlock Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int inCh, int seCh) => new(
            WhisperOps.EnsureF32(w[$"{prefix}.conv1.conv.weight"]), WhisperOps.EnsureF32(w[$"{prefix}.conv1.conv.bias"]),
            WhisperOps.EnsureF32(w[$"{prefix}.conv2.conv.weight"]), WhisperOps.EnsureF32(w[$"{prefix}.conv2.conv.bias"]),
            inCh, seCh);

        public Tensor Forward(Tensor x, int t)
        {
            float* xp = (float*)x.DataPointer;
            Span<float> mean = stackalloc float[_inCh];
            for (int c = 0; c < _inCh; c++)
            {
                float sum = 0f;
                float* row = xp + (long)c * t;
                for (int i = 0; i < t; i++) sum += row[i];
                mean[c] = sum / t;
            }

            Span<float> hidden = stackalloc float[_seCh];
            float* w1 = (float*)_w1.DataPointer; float* b1 = (float*)_b1.DataPointer;
            for (int s = 0; s < _seCh; s++)
            {
                float acc = b1[s];
                float* wRow = w1 + (long)s * _inCh;
                for (int c = 0; c < _inCh; c++) acc += wRow[c] * mean[c];
                hidden[s] = MathF.Max(0f, acc);
            }

            Span<float> gate = stackalloc float[_inCh];
            float* w2 = (float*)_w2.DataPointer; float* b2 = (float*)_b2.DataPointer;
            for (int c = 0; c < _inCh; c++)
            {
                float acc = b2[c];
                float* wRow = w2 + (long)c * _seCh;
                for (int s = 0; s < _seCh; s++) acc += wRow[s] * hidden[s];
                gate[c] = 1f / (1f + MathF.Exp(-acc));
            }

            Tensor outT = new(x.Shape, DType.F32);
            float* op = (float*)outT.DataPointer;
            for (int c = 0; c < _inCh; c++)
            {
                float g = gate[c];
                float* row = xp + (long)c * t;
                float* outRow = op + (long)c * t;
                for (int i = 0; i < t; i++) outRow[i] = row[i] * g;
            }
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights() { yield return _w1; yield return _b1; yield return _w2; yield return _b2; }
    }

    /// <summary>TDNN(1×1) → Res2Net → TDNN(1×1) → SE, residual (shortcut conv skipped: IndexTTS's three SE-Res2Net blocks are all in==out channels).</summary>
    private sealed class SeRes2NetBlock
    {
        private readonly TdnnBlock _tdnn1, _tdnn2;
        private readonly Res2NetBlock _res2net;
        private readonly SeBlock _se;

        private SeRes2NetBlock(TdnnBlock tdnn1, Res2NetBlock res2net, TdnnBlock tdnn2, SeBlock se)
        { _tdnn1 = tdnn1; _res2net = res2net; _tdnn2 = tdnn2; _se = se; }

        public static SeRes2NetBlock Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int inCh, int outCh, int kernel, int dilation) => new(
            TdnnBlock.Load(w, $"{prefix}.tdnn1", inCh, outCh, 1, 1),
            Res2NetBlock.Load(w, $"{prefix}.res2net_block", outCh, Res2NetScale, kernel, dilation),
            TdnnBlock.Load(w, $"{prefix}.tdnn2", outCh, outCh, 1, 1),
            SeBlock.Load(w, $"{prefix}.se_block", outCh, SeChannels));

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            Tensor h1 = _tdnn1.Forward(backend, x, t);
            Tensor h2 = _res2net.Forward(backend, h1, t);
            h1.Dispose();
            Tensor h3 = _tdnn2.Forward(backend, h2, t);
            h2.Dispose();
            Tensor h4 = _se.Forward(h3, t);
            h3.Dispose();

            float* hp = (float*)h4.DataPointer;
            float* xp = (float*)x.DataPointer;
            long n = h4.ElementCount;
            for (long i = 0; i < n; i++) hp[i] += xp[i];
            return h4;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _tdnn1.EnumerateWeights()) yield return t;
            foreach (Tensor t in _res2net.EnumerateWeights()) yield return t;
            foreach (Tensor t in _tdnn2.EnumerateWeights()) yield return t;
            foreach (Tensor t in _se.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Global-context attentive statistics pooling: per-channel attention-weighted mean/std over time.</summary>
    private sealed class AttentiveStatsPooling
    {
        private const float Eps = 1e-12f;
        private readonly TdnnBlock _tdnn;
        private readonly Tensor _convW, _convB;
        private readonly int _channels, _attnChannels;

        private AttentiveStatsPooling(TdnnBlock tdnn, Tensor convW, Tensor convB, int channels, int attnChannels)
        { _tdnn = tdnn; _convW = convW; _convB = convB; _channels = channels; _attnChannels = attnChannels; }

        public static AttentiveStatsPooling Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int channels, int attnChannels) => new(
            TdnnBlock.Load(w, $"{prefix}.tdnn", channels * 3, attnChannels, 1, 1),
            WhisperOps.EnsureF32(w[$"{prefix}.conv.weight"]), WhisperOps.EnsureF32(w[$"{prefix}.conv.bias"]),
            channels, attnChannels);

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int c = _channels;
            float* xp = (float*)x.DataPointer;

            Span<float> mean = c <= 2048 ? stackalloc float[c] : new float[c];
            Span<float> std = c <= 2048 ? stackalloc float[c] : new float[c];
            for (int ch = 0; ch < c; ch++)
            {
                float* row = xp + (long)ch * t;
                float sum = 0f;
                for (int i = 0; i < t; i++) sum += row[i];
                float m = sum / t;
                float sq = 0f;
                for (int i = 0; i < t; i++) { float d = row[i] - m; sq += d * d; }
                mean[ch] = m;
                std[ch] = MathF.Sqrt(MathF.Max(sq / t, Eps));
            }

            Tensor catX = new(new TensorShape(1, 3 * c, t), DType.F32);
            float* catp = (float*)catX.DataPointer;
            for (long i = 0; i < (long)c * t; i++) catp[i] = xp[i];
            for (int ch = 0; ch < c; ch++)
            {
                float* dstMean = catp + (long)(c + ch) * t;
                float* dstStd = catp + (long)(2 * c + ch) * t;
                for (int i = 0; i < t; i++) { dstMean[i] = mean[ch]; dstStd[i] = std[ch]; }
            }

            // tdnn -> tanh -> conv (attention_channels -> channels).
            Tensor tdnnOut = _tdnn.Forward(backend, catX, t);
            catX.Dispose();
            float* tp = (float*)tdnnOut.DataPointer;
            long tn = tdnnOut.ElementCount;
            for (long i = 0; i < tn; i++) tp[i] = MathF.Tanh(tp[i]);

            Tensor attnLogits = new(new TensorShape(1, c, t), DType.F32);
            float* alp = (float*)attnLogits.DataPointer;
            float* cw = (float*)_convW.DataPointer; float* cb = (float*)_convB.DataPointer;
            for (int oc = 0; oc < c; oc++)
            {
                float* wRow = cw + (long)oc * _attnChannels;
                for (int i = 0; i < t; i++)
                {
                    float acc = cb[oc];
                    for (int ac = 0; ac < _attnChannels; ac++) acc += wRow[ac] * tp[(long)ac * t + i];
                    alp[(long)oc * t + i] = acc;
                }
            }
            tdnnOut.Dispose();

            // Per-channel softmax over time, then attention-weighted mean/std.
            Tensor stats = new(new TensorShape(1, 2 * c, 1), DType.F32);
            float* statp = (float*)stats.DataPointer;
            Span<float> weights = t <= 4096 ? stackalloc float[t] : new float[t];
            for (int ch = 0; ch < c; ch++)
            {
                float* logitRow = alp + (long)ch * t;
                float maxV = float.NegativeInfinity;
                for (int i = 0; i < t; i++) if (logitRow[i] > maxV) maxV = logitRow[i];
                float sum = 0f;
                for (int i = 0; i < t; i++) { float e = MathF.Exp(logitRow[i] - maxV); weights[i] = e; sum += e; }
                float inv = 1f / sum;

                float* xRow = xp + (long)ch * t;
                float wMean = 0f;
                for (int i = 0; i < t; i++) wMean += weights[i] * inv * xRow[i];
                float wSq = 0f;
                for (int i = 0; i < t; i++) { float d = xRow[i] - wMean; wSq += weights[i] * inv * d * d; }
                statp[ch] = wMean;
                statp[c + ch] = MathF.Sqrt(MathF.Max(wSq, Eps));
            }
            attnLogits.Dispose();
            return stats;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _tdnn.EnumerateWeights()) yield return t;
            yield return _convW; yield return _convB;
        }
    }

    /// <summary>Inference-mode BatchNorm1d: <c>(x - runningMean) / sqrt(runningVar + eps) * weight + bias</c>, per channel, broadcast over time.</summary>
    private sealed class BatchNorm1d
    {
        private const float Eps = 1e-5f;
        private readonly Tensor _weight, _bias, _runningMean, _runningVar;
        private readonly int _channels;

        private BatchNorm1d(Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, int channels)
        { _weight = weight; _bias = bias; _runningMean = runningMean; _runningVar = runningVar; _channels = channels; }

        public static BatchNorm1d Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int channels) => new(
            WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]), WhisperOps.EnsureF32(w[$"{prefix}.norm.bias"]),
            WhisperOps.EnsureF32(w[$"{prefix}.norm.running_mean"]), WhisperOps.EnsureF32(w[$"{prefix}.norm.running_var"]),
            channels);

        public Tensor Forward(Tensor x)
        {
            int t = (int)(x.ElementCount / _channels);
            Tensor outT = new(x.Shape, DType.F32);
            float* xp = (float*)x.DataPointer, op = (float*)outT.DataPointer;
            float* wp = (float*)_weight.DataPointer, bp = (float*)_bias.DataPointer;
            float* mp = (float*)_runningMean.DataPointer, vp = (float*)_runningVar.DataPointer;
            for (int c = 0; c < _channels; c++)
            {
                float invStd = 1f / MathF.Sqrt(vp[c] + Eps);
                float scale = wp[c] * invStd;
                float shift = bp[c] - mp[c] * scale;
                float* row = xp + (long)c * t;
                float* outRow = op + (long)c * t;
                for (int i = 0; i < t; i++) outRow[i] = row[i] * scale + shift;
            }
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights() { yield return _weight; yield return _bias; yield return _runningMean; yield return _runningVar; }
    }
}
