using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>Full HTDemucs (Hybrid Transformer Demucs v4) assembly. Dual-branch U-Net: a time branch (1D convs over
/// the waveform) and a spectrogram branch (2D convs over the complex-as-channels STFT) encode in parallel; at the
/// bottleneck the freq dimension collapses to 1, the time branch is injected into the freq branch, both token
/// streams are up-projected to <c>bottom_channels</c>, mixed by the <see cref="DemucsCrossTransformer"/>,
/// down-projected, and decoded with U-Net skips. The spec output → complex spectrogram → inverse STFT is summed
/// with the time-branch waveform to yield 4 stereo stems <c>[1, 4, 2, L]</c>. Channels-first, batch=1.
/// Synthetic-forward / structural build — numeric parity vs the Python reference is validation-pending (the
/// freq-collapse/merge and STFT padding are the bit-exact-risky parts). See
/// <c>docs/Research/HTDEMUCS_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class HtDemucs
{
    private readonly HtDemucsConfig _cfg;
    private readonly int _depth;
    private readonly DemucsConvBlock[] _enc;     // freq encoders (2D)
    private readonly DemucsConvBlock[] _tenc;    // time encoders (1D)
    private readonly DemucsConvBlock[] _dec;     // freq decoders (2D)
    private readonly DemucsConvBlock[] _tdec;    // time decoders (1D)
    private readonly DemucsCrossTransformer _xf;
    private readonly int[] _encChannels;         // output channels per encoder layer
    private Tensor? _freqEmb;                     // [Fq0, channels] embedding table
    private Tensor? _upW, _upB, _downW, _downB;   // channel_upsampler / channel_downsampler (1×1 Conv1d)
    private Tensor? _upWt, _upBt, _downWt, _downBt;

    public HtDemucs(HtDemucsConfig cfg)
    {
        _cfg = cfg; _depth = cfg.Depth;
        _enc = new DemucsConvBlock[_depth]; _tenc = new DemucsConvBlock[_depth];
        _dec = new DemucsConvBlock[_depth]; _tdec = new DemucsConvBlock[_depth];
        _encChannels = new int[_depth];
        int specIn = cfg.SpecInChannels;             // 2C real/imag
        int timeIn = cfg.AudioChannels;
        int ch = cfg.Channels;
        for (int i = 0; i < _depth; i++)
        {
            int inSpec = i == 0 ? specIn : _encChannels[i - 1];
            int inTime = i == 0 ? timeIn : _encChannels[i - 1];
            _encChannels[i] = ch;
            _enc[i] = new DemucsConvBlock(inSpec, ch, cfg.KernelSize, cfg.Stride, is2d: true, decoder: false);
            _tenc[i] = new DemucsConvBlock(inTime, ch, cfg.KernelSize, cfg.Stride, is2d: false, decoder: false);
            ch *= cfg.Growth;
        }
        // Decoders mirror encoders in reverse: layer i restores encoder layer (depth-1-i)'s input channels.
        // The final layer (di==0) instead emits srcs·(audio channels) so every stem is produced at once.
        int srcs = cfg.NumSources;
        for (int i = 0; i < _depth; i++)
        {
            int di = _depth - 1 - i;
            int outSpec = di == 0 ? srcs * specIn : _encChannels[di - 1];
            int outTime = di == 0 ? srcs * timeIn : _encChannels[di - 1];
            bool last = di == 0;
            _dec[i] = new DemucsConvBlock(_encChannels[di], outSpec, cfg.KernelSize, cfg.Stride, is2d: true, decoder: true, last);
            _tdec[i] = new DemucsConvBlock(_encChannels[di], outTime, cfg.KernelSize, cfg.Stride, is2d: false, decoder: true, last);
        }
        _xf = new DemucsCrossTransformer(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        for (int i = 0; i < _depth; i++)
        {
            _enc[i].LoadWeights(w, $"encoder.{i}");
            _tenc[i].LoadWeights(w, $"tencoder.{i}");
            _dec[i].LoadWeights(w, $"decoder.{i}");
            _tdec[i].LoadWeights(w, $"tdecoder.{i}");
        }
        _xf.LoadWeights(w, "crosstransformer");
        if (w.TryGetValue("freq_emb.embedding.weight", out Tensor? fe)) _freqEmb = WhisperOps.EnsureF32(fe);
        // The channel (up/down)samplers are top-level modules in HTDemucs, NOT inside the crosstransformer.
        _upW = WhisperOps.EnsureF32(w["channel_upsampler.weight"]);
        _upB = Bias(w, "channel_upsampler.bias");
        _downW = WhisperOps.EnsureF32(w["channel_downsampler.weight"]);
        _downB = Bias(w, "channel_downsampler.bias");
        _upWt = WhisperOps.EnsureF32(w["channel_upsampler_t.weight"]);
        _upBt = Bias(w, "channel_upsampler_t.bias");
        _downWt = WhisperOps.EnsureF32(w["channel_downsampler_t.weight"]);
        _downBt = Bias(w, "channel_downsampler_t.bias");
    }

    /// <summary>Separates a stereo waveform into 4 stereo stems. Input waveform <c>[1, C, L]</c> (channels-first,
    /// C = <c>AudioChannels</c>); returns <c>[1, NumSources, C, L]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor wav, int length)
    {
        int channels = _cfg.AudioChannels;
        int srcs = _cfg.NumSources;

        // ── Time normalize (per the mono mix). ──
        Tensor timeIn = new(new TensorShape(1, channels, length), DType.F32);
        float* wp = (float*)wav.DataPointer; float* tp = (float*)timeIn.DataPointer;
        (float mean, float std) = MonoMeanStd(wp, channels, length);
        float invStd = 1f / (1e-5f + std);
        for (long i = 0; i < (long)channels * length; i++) tp[i] = (wp[i] - mean) * invStd;

        // ── STFT → cac → freq normalize. ──
        Tensor spec = DemucsSpec.Spec(backend, timeIn, channels, length, _cfg.NFft, _cfg.HopLength, out int freq, out int time);
        (float smean, float sstd) = TensorMeanStd(spec);
        float sInv = 1f / (1e-5f + sstd);
        float* spp = (float*)spec.DataPointer;
        for (long i = 0; i < spec.ElementCount; i++) spp[i] = (spp[i] - smean) * sInv;

        // ── Encode. Freq branch (2D, strides freq) + time branch (1D, strides time), collecting skips. ──
        Tensor[] specSkip = new Tensor[_depth];
        Tensor[] timeSkip = new Tensor[_depth];
        Tensor x = spec; int f = freq, tlen = time;
        Tensor xt = timeIn; int tt = length;
        for (int i = 0; i < _depth; i++)
        {
            Tensor nx = _enc[i].EncodeForward(backend, x, f, tlen);
            if (i != 0) x.Dispose();
            x = nx;
            f = (int)x.Shape[2];
            if (i == 0 && _freqEmb is not null) AddFreqEmb(x, f, tlen, _encChannels[0]);
            specSkip[i] = Clone(x);

            Tensor nxt = _tenc[i].EncodeForward(backend, xt, 1, tt);
            if (i != 0) xt.Dispose();
            xt = nxt;
            tt = (int)xt.Shape[2];
            timeSkip[i] = Clone(xt);
        }
        spec.Dispose(); timeIn.Dispose();

        // ── Merge: collapse the freq axis to 1 and inject the time branch. ──
        // After the encoder freq is small but not necessarily 1; average-collapse the residual freq into 1 so the
        // token layout matches the time branch, then add the (channel-aligned) time features as the inject.
        Tensor xCollapsed = CollapseFreq(x, _encChannels[_depth - 1], f, tlen); x.Dispose();
        int nSpec = tlen;                              // tokens = (1·time)
        int nTime = tt;
        AddInject(xCollapsed, xt, _encChannels[_depth - 1], nSpec, nTime);

        // ── Up-project both streams to bottom_channels, cross-transform, down-project. ──
        int bottom = _cfg.BottomChannels;
        int encCh = _encChannels[_depth - 1];
        Tensor sUp = Conv1x1(backend, xCollapsed, _upW!, _upB, encCh, bottom, nSpec); xCollapsed.Dispose();
        Tensor tUp = Conv1x1(backend, xt, _upWt!, _upBt, encCh, bottom, nTime); xt.Dispose();
        (Tensor sMix, Tensor tMix) = _xf.Forward(backend, sUp, nSpec, tUp, nTime);
        sUp.Dispose(); tUp.Dispose();
        Tensor sDown = Conv1x1(backend, sMix, _downW!, _downB, bottom, encCh, nSpec); sMix.Dispose();
        Tensor tDown = Conv1x1(backend, tMix, _downWt!, _downBt, bottom, encCh, nTime); tMix.Dispose();

        // Restore the freq axis (=1) on the spec stream → [1, encCh, 1, tlen].
        Tensor xs = ExpandFreq(sDown, encCh, tlen); sDown.Dispose();
        Tensor xtd = tDown;

        // ── Decode. Freq decoder returns features that feed the next layer; both add skips first. ──
        int curF = 1;
        for (int i = 0; i < _depth; i++)
        {
            int di = _depth - 1 - i;
            AddSkip(xs, specSkip[di]);
            int decT = (int)xs.Shape[3];
            Tensor nxs = _dec[i].DecodeForward(backend, xs, curF, decT);
            xs.Dispose(); specSkip[di].Dispose(); xs = nxs;
            curF = (int)xs.Shape[2];

            AddSkip(xtd, timeSkip[di]);
            int decTt = (int)xtd.Shape[2];
            Tensor nxtd = _tdec[i].DecodeForward(backend, xtd, 1, decTt);
            xtd.Dispose(); timeSkip[di].Dispose(); xtd = nxtd;
        }

        // ── Spec branch → complex → iSTFT (channels = 2C → C audio channels via cac); time branch denorm. ──
        // The freq decoder outputs [1, srcs*specIn, Fq, T]; reshape into [1, srcs, specIn, Fq, T] per source.
        int specIn = _cfg.SpecInChannels;
        int outFreq = (int)xs.Shape[2];
        int outTime = (int)xs.Shape[3];
        Tensor outStems = new(new TensorShape(1, srcs, channels, length), DType.F32);
        float* osp = (float*)outStems.DataPointer;
        for (int s = 0; s < srcs; s++)
        {
            Tensor cac = SliceSource(xs, s, srcs, specIn, outFreq, outTime);
            Tensor wave = DemucsSpec.InverseSpec(cac, channels, length, _cfg.NFft, _cfg.HopLength); cac.Dispose();
            Tensor twave = SliceTimeSource(xtd, s, srcs, channels, length, std, mean);
            float* wpv = (float*)wave.DataPointer; float* twv = (float*)twave.DataPointer;
            for (int c = 0; c < channels; c++)
                for (int j = 0; j < length; j++)
                    osp[((((long)s * channels) + c) * length) + j] = wpv[(long)c * length + j] + twv[(long)c * length + j];
            wave.Dispose(); twave.Dispose();
        }
        xs.Dispose(); xtd.Dispose();
        return outStems;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _depth; i++)
        {
            foreach (Tensor w in _enc[i].EnumerateWeights()) yield return w;
            foreach (Tensor w in _tenc[i].EnumerateWeights()) yield return w;
            foreach (Tensor w in _dec[i].EnumerateWeights()) yield return w;
            foreach (Tensor w in _tdec[i].EnumerateWeights()) yield return w;
        }
        foreach (Tensor w in _xf.EnumerateWeights()) yield return w;
        Tensor?[] own = [_freqEmb, _upW, _upB, _downW, _downB, _upWt, _upBt, _downWt, _downBt];
        foreach (Tensor? w in own) if (w is not null) yield return w;
    }

    private static (float Mean, float Std) MonoMeanStd(float* wp, int channels, int length)
    {
        double sum = 0;
        for (int j = 0; j < length; j++)
        {
            double m = 0;
            for (int c = 0; c < channels; c++) m += wp[(long)c * length + j];
            sum += m / channels;
        }
        double mean = sum / length;
        double var = 0;
        for (int j = 0; j < length; j++)
        {
            double m = 0;
            for (int c = 0; c < channels; c++) m += wp[(long)c * length + j];
            m /= channels;
            double d = m - mean; var += d * d;
        }
        var /= Math.Max(1, length - 1);
        return ((float)mean, (float)Math.Sqrt(var));
    }

    private static (float Mean, float Std) TensorMeanStd(Tensor t)
    {
        float* p = (float*)t.DataPointer; long n = t.ElementCount;
        double sum = 0; for (long i = 0; i < n; i++) sum += p[i];
        double mean = sum / n;
        double var = 0; for (long i = 0; i < n; i++) { double d = p[i] - mean; var += d * d; }
        var /= Math.Max(1, n - 1);
        return ((float)mean, (float)Math.Sqrt(var));
    }

    /// <summary>Adds <c>freq_emb_scale · embedding[freqIdx]</c> across channels, broadcast over time.</summary>
    private void AddFreqEmb(Tensor x, int f, int t, int channels)
    {
        float* xp = (float*)x.DataPointer; float* ep = (float*)_freqEmb!.DataPointer;
        float scale = _cfg.FreqEmbScale;
        int tableF = (int)_freqEmb.Shape[0];
        for (int c = 0; c < channels; c++)
            for (int fi = 0; fi < f; fi++)
            {
                int row = fi < tableF ? fi : tableF - 1;
                float e = scale * ep[(long)row * channels + c];
                for (int ti = 0; ti < t; ti++) xp[(((long)c * f + fi) * t) + ti] += e;
            }
    }

    /// <summary>Mean-collapses the residual freq axis into a single token row: <c>[1, C, F, T] → [1, C, T]</c>.</summary>
    private static Tensor CollapseFreq(Tensor x, int channels, int f, int t)
    {
        Tensor o = new(new TensorShape(1, channels, t), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int c = 0; c < channels; c++)
            for (int ti = 0; ti < t; ti++)
            {
                float acc = 0f;
                for (int fi = 0; fi < f; fi++) acc += xp[(((long)c * f + fi) * t) + ti];
                op[(long)c * t + ti] = acc / f;
            }
        return o;
    }

    private static Tensor ExpandFreq(Tensor x, int channels, int t)
    {
        Tensor o = new(new TensorShape(1, channels, 1, t), DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)o.DataPointer, (long)channels * t * 4, (long)channels * t * 4);
        return o;
    }

    /// <summary>Adds the time-branch features into the collapsed freq tokens (the merge inject), aligning the
    /// shorter sequence by truncation.</summary>
    private static void AddInject(Tensor spec, Tensor time, int channels, int nSpec, int nTime)
    {
        float* sp = (float*)spec.DataPointer; float* tp = (float*)time.DataPointer;
        int n = Math.Min(nSpec, nTime);
        for (int c = 0; c < channels; c++)
            for (int j = 0; j < n; j++)
                sp[(long)c * nSpec + j] += tp[(long)c * nTime + j];
    }

    private static Tensor Conv1x1(IBackend b, Tensor x, Tensor w, Tensor? bias, int inCh, int outCh, int n)
    {
        Tensor o = new(new TensorShape(1, outCh, n), DType.F32);
        b.Conv1d(o, x, w, bias, 1, 0, 0, 1, 1);
        return o;
    }

    private static void AddSkip(Tensor x, Tensor skip)
    {
        long n = Math.Min(x.ElementCount, skip.ElementCount);
        float* xp = (float*)x.DataPointer; float* sp = (float*)skip.DataPointer;
        for (long i = 0; i < n; i++) xp[i] += sp[i];
    }

    private static Tensor Clone(Tensor t)
    {
        Tensor o = new(t.Shape, DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)o.DataPointer, t.ElementCount * 4, t.ElementCount * 4);
        return o;
    }

    /// <summary>Extracts source <paramref name="s"/>'s complex-as-channels block <c>[1, specIn, Fq, T]</c> from the
    /// decoder output <c>[1, srcs·specIn, Fq, T]</c>.</summary>
    private static Tensor SliceSource(Tensor xs, int s, int srcs, int specIn, int f, int t)
    {
        Tensor o = new(new TensorShape(1, specIn, f, t), DType.F32);
        float* xp = (float*)xs.DataPointer; float* op = (float*)o.DataPointer;
        long plane = (long)f * t;
        for (int c = 0; c < specIn; c++)
        {
            int srcCh = s * specIn + c;
            Buffer.MemoryCopy(xp + srcCh * plane, op + (long)c * plane, plane * 4, plane * 4);
        }
        return o;
    }

    /// <summary>Extracts source <paramref name="s"/>'s waveform <c>[1, C, L]</c> from the time decoder output
    /// <c>[1, srcs·C, L]</c> and denormalizes (undo the input time-normalize).</summary>
    private static Tensor SliceTimeSource(Tensor xtd, int s, int srcs, int channels, int length, float std, float mean)
    {
        int outLen = (int)xtd.Shape[2];
        Tensor o = new(new TensorShape(1, channels, length), DType.F32);
        float* xp = (float*)xtd.DataPointer; float* op = (float*)o.DataPointer;
        float scale = 1e-5f + std;
        int n = Math.Min(length, outLen);
        for (int c = 0; c < channels; c++)
        {
            int srcCh = s * channels + c;
            for (int j = 0; j < n; j++) op[(long)c * length + j] = xp[(long)srcCh * outLen + j] * scale + mean;
        }
        return o;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
