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

    /// <summary>Optional per-stage activation hook for parity debugging (key = stage name). Not used in production.</summary>
    /// <remarks>Unwired on purpose: the keys emitted here match <c>tests/python-reference/demucs_reference/dump_demucs_stages.py</c>'s dump names one-for-one, so a stage-by-stage diff needs only a hook assignment.</remarks>
    public Action<string, Tensor>? DebugHook { get; set; }

    /// <summary>Separates a stereo waveform into 4 stereo stems. Input waveform <c>[1, C, L]</c> (channels-first,
    /// C = <c>AudioChannels</c>); returns <c>[1, NumSources, C, L]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor wav, int length)
    {
        int channels = _cfg.AudioChannels;
        int srcs = _cfg.NumSources;
        int encCh = _encChannels[_depth - 1];     // 384
        int bottom = _cfg.BottomChannels;          // 512

        // ── Spec from the RAW mix (cac), then normalize by the spec mean/std over all (1,2,3). ──
        Tensor spec = DemucsSpec.Spec(backend, wav, channels, length, _cfg.NFft, _cfg.HopLength, out int freq, out int time);
        DebugHook?.Invoke("spec_cac", spec);
        (float smean, float sstd) = TensorMeanStd(spec);
        float sInv = 1f / (1e-5f + sstd);
        float* spp = (float*)spec.DataPointer;
        for (long i = 0; i < spec.ElementCount; i++) spp[i] = (spp[i] - smean) * sInv;
        DebugHook?.Invoke("spec_norm", spec);
        if (DebugHook is not null) { DemucsConvBlock.ConvProbe = c => DebugHook("enc0_conv", c); DemucsConvBlock.DConvProbe = c => DebugHook("enc0_postdconv", c); }

        // ── Time branch from the RAW mix, normalized by its own mean/std over (1,2). ──
        (float tmean, float tstd) = TensorMeanStd(wav);
        float tInv = 1f / (1e-5f + tstd);
        Tensor timeIn = new(new TensorShape(1, channels, length), DType.F32);
        float* wp = (float*)wav.DataPointer; float* tp = (float*)timeIn.DataPointer;
        for (long i = 0; i < (long)channels * length; i++) tp[i] = (wp[i] - tmean) * tInv;

        // ── Encoder: parallel freq (2D) + time (1D) branches (htdemucs has no inject), collecting skips. ──
        Tensor[] specSkip = new Tensor[_depth];
        Tensor[] timeSkip = new Tensor[_depth];
        int[] timeLen = new int[_depth];
        Tensor x = spec; int f = freq, t1 = time;
        Tensor xt = timeIn; int tt = length;
        for (int i = 0; i < _depth; i++)
        {
            if (i == 1 && DebugHook is not null) { DemucsConvBlock.ConvProbe = c => DebugHook("enc1_conv", c); DemucsConvBlock.DConvProbe = c => DebugHook("enc1_postdconv", c); }
            Tensor nx = _enc[i].EncodeForward(backend, x, f, t1);
            x.Dispose(); x = nx; f = (int)x.Shape[2];
            if (i == 0) DebugHook?.Invoke("enc0_prefreq", x);
            if (i == 0 && _freqEmb is not null) AddFreqEmb(x, f, t1, _encChannels[0]);
            specSkip[i] = Clone(x);

            timeLen[i] = tt;
            Tensor nxt = _tenc[i].EncodeForward(backend, xt, 1, tt);
            xt.Dispose(); xt = nxt; tt = (int)xt.Shape[2];
            timeSkip[i] = Clone(xt);
            DebugHook?.Invoke($"enc{i}", x); DebugHook?.Invoke($"tenc{i}", xt);
        }

        // ── Channel up-project (1x1) both streams to bottom_channels, cross-transform, down-project. ──
        Tensor sUp = ChannelProj(backend, x, _upW!, _upB, encCh, bottom, f * t1, f, t1, true); x.Dispose();
        Tensor tUp = ChannelProj(backend, xt, _upWt!, _upBt, encCh, bottom, tt, 1, tt, false); xt.Dispose();
        DebugHook?.Invoke("ct_in_x", sUp); DebugHook?.Invoke("ct_in_xt", tUp);
        if (DebugHook is not null) { DemucsCrossTransformer.Probe = (k, v) => DebugHook(k, v); DemucsCrossTransformer.AttnProbe = v => DebugHook("ct_l0_attn", v); DemucsCrossTransformer.AfterAttnProbe = v => DebugHook("ct_l0_afterattn", v); DemucsCrossTransformer.PreNormOutProbe = v => DebugHook("ct_l0_out", v); }
        (Tensor sMix, Tensor tMix) = _xf.Forward(backend, sUp, bottom, f, t1, tUp, tt);
        DemucsCrossTransformer.Probe = null;
        sUp.Dispose(); tUp.Dispose();
        DebugHook?.Invoke("ct_out_x", sMix); DebugHook?.Invoke("ct_out_xt", tMix);
        Tensor xs = ChannelProj(backend, sMix, _downW!, _downB, bottom, encCh, f * t1, f, t1, true); sMix.Dispose();
        Tensor xtd = ChannelProj(backend, tMix, _downWt!, _downBt, bottom, encCh, tt, 1, tt, false); tMix.Dispose();

        // ── Decoder: freq + time branches each add the same-level encoder skip, then up-convolve. ──
        for (int i = 0; i < _depth; i++)
        {
            int di = _depth - 1 - i;
            AddSkip(xs, specSkip[di]);
            Tensor nxs = _dec[i].DecodeForward(backend, xs, f, t1, t1);
            xs.Dispose(); specSkip[di].Dispose(); xs = nxs; f = (int)xs.Shape[2];

            AddSkip(xtd, timeSkip[di]);
            Tensor nxtd = _tdec[i].DecodeForward(backend, xtd, 1, (int)xtd.Shape[2], timeLen[di]);
            xtd.Dispose(); timeSkip[di].Dispose(); xtd = nxtd;
            DebugHook?.Invoke($"dec{i}", xs); DebugHook?.Invoke($"tdec{i}", xtd);
        }

        // ── Freq branch: denorm (spec stats, ·std + mean) → per-source cac → iSTFT. Time branch: denorm (time
        // stats) → add. The cac → complex mask is the identity for cac (the decoder output IS the spectrogram). ──
        float* xsp = (float*)xs.DataPointer;
        for (long i = 0; i < xs.ElementCount; i++) xsp[i] = xsp[i] * sstd + smean;
        int specIn = _cfg.SpecInChannels;
        int outFreq = (int)xs.Shape[2];
        int outTime = (int)xs.Shape[3];
        float* xtdp = (float*)xtd.DataPointer;
        int xtdLen = (int)xtd.Shape[2];

        Tensor outStems = new(new TensorShape(1, srcs, channels, length), DType.F32);
        float* osp = (float*)outStems.DataPointer;
        for (int s = 0; s < srcs; s++)
        {
            Tensor cac = SliceSource(xs, s, srcs, specIn, outFreq, outTime);
            Tensor wave = DemucsSpec.InverseSpec(cac, channels, length, _cfg.NFft, _cfg.HopLength); cac.Dispose();
            float* wpv = (float*)wave.DataPointer;
            for (int c = 0; c < channels; c++)
            {
                int tCh = s * channels + c;
                for (int j = 0; j < length; j++)
                {
                    float tv = j < xtdLen ? xtdp[(long)tCh * xtdLen + j] * tstd + tmean : 0f;
                    osp[((((long)s * channels) + c) * length) + j] = wpv[(long)c * length + j] + tv;
                }
            }
            wave.Dispose();
        }
        xs.Dispose(); xtd.Dispose();
        return outStems;
    }

    /// <summary>1x1 channel projection (<c>channel_(up/down)sampler</c>): reshapes the (optionally 2D) feature to
    /// <c>[1, inCh, n]</c>, runs the 1x1 Conv1d, and restores the layout. Order-agnostic (1x1 is pointwise).</summary>
    private static Tensor ChannelProj(IBackend backend, Tensor x, Tensor w, Tensor? bias, int inCh, int outCh, int n, int f, int t, bool is2d)
    {
        Tensor flat = new(new TensorShape(1, inCh, n), DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)flat.DataPointer, (long)inCh * n * 4, (long)inCh * n * 4);
        Tensor o = Conv1x1(backend, flat, w, bias, inCh, outCh, n); flat.Dispose();
        if (!is2d) return o;
        Tensor o4 = new(new TensorShape(1, outCh, f, t), DType.F32);
        Buffer.MemoryCopy((void*)o.DataPointer, (void*)o4.DataPointer, (long)outCh * n * 4, (long)outCh * n * 4);
        o.Dispose();
        return o4;
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
        // demucs ScaledEmbedding.forward multiplies by its internal scale (EmbScale=10); the main forward then
        // multiplies by freq_emb_scale (0.2). Effective = 2.0 on the stored embedding weight.
        float scale = _cfg.FreqEmbScale * _cfg.EmbScale;
        int tableF = (int)_freqEmb.Shape[0];
        for (int c = 0; c < channels; c++)
            for (int fi = 0; fi < f; fi++)
            {
                int row = fi < tableF ? fi : tableF - 1;
                float e = scale * ep[(long)row * channels + c];
                for (int ti = 0; ti < t; ti++) xp[(((long)c * f + fi) * t) + ti] += e;
            }
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

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
