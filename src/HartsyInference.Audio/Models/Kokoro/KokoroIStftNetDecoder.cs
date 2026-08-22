using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kokoro;

/// <summary>Kokoro's iSTFTNet decoder — the style-conditioned HiFi-GAN-with-iSTFT head
/// that converts the predicted prosody curves into a 24 kHz waveform. This module is
/// the largest of Kokoro's submodules (~375 tensors out of 548 total) and breaks down as:
///
/// <list type="bullet">
///   <item><c>asr_res</c> — 1×1 Conv 512→64 reducing the text features to a thin residual</item>
///   <item><c>F0_conv</c>, <c>N_conv</c> — k=3 smoothing convs on the predicted curves</item>
///   <item><c>encode</c> — single AdainResBlk1d 514→1024 that fuses asr+F0+N</item>
///   <item><c>decode</c> — 4 AdainResBlk1d (last is upsampling 2×)</item>
///   <item><c>generator</c> — full HiFi-GAN-iSTFT generator with harmonic+noise source,
///         two upsampling stages (10×, 6×), MRF resblocks with AdaIN+Snake activation,
///         noise residuals, and a final k=7 Conv1D → 22-channel iSTFT pair head</item>
/// </list>
///
/// <para><b>Status:</b> full forward pass implemented — encode/decode AdaIN chain → HnNSF
/// harmonic source → forward STFT → two upsample stages with noise injection + MRF AdaIN/Snake
/// resblocks → conv_post → magnitude/phase iSTFT head. Reproduces StyleTTS2's
/// <c>kokoro/istftnet.py</c> Decoder + Generator. The harmonic source uses deterministic phase
/// accumulation (no random initial phase) and a fixed-seed Gaussian for the noise component, so
/// output is reproducible; exact-bit parity with the reference is not expected (the reference
/// randomizes both), but perceptual quality matches.</para></summary>
public sealed unsafe class KokoroIStftNetDecoder
{
    private readonly KokoroConfig _cfg;

    // ── Pre-encode convs ─────────────────────────────────────────────────
    // asr_res.0: 1×1 Conv 512 → 64
    private Tensor? _asrResW, _asrResB;
    // F0_conv / N_conv: k=3, in=1, out=1 smoothing
    private Tensor? _f0ConvW, _f0ConvB;
    private Tensor? _nConvW, _nConvB;

    // ── encode: AdainResBlk1d(514 → 1024) ────────────────────────────────
    private readonly AdaResLoader _encode;

    // ── decode: 4 AdainResBlk1d blocks ───────────────────────────────────
    private readonly AdaResLoader[] _decode = new AdaResLoader[4];

    // ── Generator ────────────────────────────────────────────────────────
    // m_source: l_linear maps 9 harmonics + bias → 1-channel sine source merge
    private Tensor? _mSourceW, _mSourceB;
    // ups.0 / ups.1: upsampling transposed convs (10×, 6×)
    private Tensor?[] _upsW = new Tensor[2];
    private Tensor?[] _upsB = new Tensor[2];
    // noise_convs: stride-aligned conv on STFT of source signal
    private Tensor?[] _noiseConvW = new Tensor[2];
    private Tensor?[] _noiseConvB = new Tensor[2];
    // noise_res, resblocks: AdaIN+Snake+Conv chains (handled via AdaSnakeResLoader)
    private readonly AdaSnakeResLoader[] _noiseRes = new AdaSnakeResLoader[2];
    private readonly AdaSnakeResLoader[] _resblocks = new AdaSnakeResLoader[6];
    // conv_post: final k=7 conv producing 22 = 2 * (n_fft/2 + 1) iSTFT bins
    private Tensor? _convPostW, _convPostB;

    // When true, the shared encode/decode feeds a StyleTTS2 HiFiGAN generator (LibriTTS `type: hifigan`) instead
    // of Kokoro's iSTFTNet generator. Default false → Kokoro path is byte-for-byte unchanged.
    private readonly bool _useHifiGan;
    private StyleTts2.StyleHifiGanGenerator? _hifiGan;

    public KokoroIStftNetDecoder(KokoroConfig cfg, bool useHifiGan = false)
    {
        _cfg = cfg;
        _useHifiGan = useHifiGan;
        if (useHifiGan) _hifiGan = new StyleTts2.StyleHifiGanGenerator(cfg.SampleRate);
        _encode = new AdaResLoader(dimIn: 514, dimOut: 1024, upsample: false, kernel: 3);
        for (int i = 0; i < 3; i++) _decode[i] = new AdaResLoader(dimIn: 1024 + 2 + 64, dimOut: 1024, upsample: false, kernel: 3);
        _decode[3] = new AdaResLoader(dimIn: 1024 + 2 + 64, dimOut: 512, upsample: true, kernel: 3);

        // resblocks: 3 per upsample stage × 2 stages = 6 total, with channel counts
        // halving at each stage. From config: kernel sizes [3, 7, 11], dilations [[1,3,5]×3].
        // The first 3 resblocks operate on 256 channels (after ups.0 reduces 512→256),
        // the next 3 on 128 channels (after ups.1 256→128).
        int[] resCh = [256, 256, 256, 128, 128, 128];
        int[] resK = [3, 7, 11, 3, 7, 11];
        for (int i = 0; i < 6; i++) _resblocks[i] = new AdaSnakeResLoader(channels: resCh[i], kernel: resK[i], dilations: [1, 3, 5]);
        _noiseRes[0] = new AdaSnakeResLoader(channels: 256, kernel: 7, dilations: [1, 3, 5]);
        _noiseRes[1] = new AdaSnakeResLoader(channels: 128, kernel: 11, dilations: [1, 3, 5]);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _asrResW = WeightNorm.Compose(w, "decoder.asr_res.0");
        _asrResB = WhisperOps.EnsureF32(w["decoder.asr_res.0.bias"]);
        _f0ConvW = WeightNorm.Compose(w, "decoder.F0_conv");
        _f0ConvB = WhisperOps.EnsureF32(w["decoder.F0_conv.bias"]);
        _nConvW = WeightNorm.Compose(w, "decoder.N_conv");
        _nConvB = WhisperOps.EnsureF32(w["decoder.N_conv.bias"]);

        _encode.LoadWeights(w, "decoder.encode");
        for (int i = 0; i < 4; i++) _decode[i].LoadWeights(w, $"decoder.decode.{i}");

        if (_useHifiGan)
        {
            // The HiFiGAN generator owns its own m_source / noise_convs / noise_res / ups / resblocks / alphas /
            // conv_post (different shapes + count than iSTFTNet), so skip the iSTFTNet generator load entirely.
            _hifiGan!.LoadWeights(w, "decoder.generator");
            return;
        }

        _mSourceW = WhisperOps.EnsureF32(w["decoder.generator.m_source.l_linear.weight"]);
        _mSourceB = WhisperOps.EnsureF32(w["decoder.generator.m_source.l_linear.bias"]);

        for (int i = 0; i < 2; i++)
        {
            _upsW[i] = WeightNorm.Compose(w, $"decoder.generator.ups.{i}");
            _upsB[i] = WhisperOps.EnsureF32(w[$"decoder.generator.ups.{i}.bias"]);
            _noiseConvW[i] = WhisperOps.EnsureF32(w[$"decoder.generator.noise_convs.{i}.weight"]);
            _noiseConvB[i] = WhisperOps.EnsureF32(w[$"decoder.generator.noise_convs.{i}.bias"]);
            _noiseRes[i].LoadWeights(w, $"decoder.generator.noise_res.{i}");
        }
        for (int i = 0; i < 6; i++) _resblocks[i].LoadWeights(w, $"decoder.generator.resblocks.{i}");

        _convPostW = WeightNorm.Compose(w, "decoder.generator.conv_post");
        _convPostB = WhisperOps.EnsureF32(w["decoder.generator.conv_post.bias"]);
    }

    /// <summary>Forward pass — runs the full decoder + generator over the predicted prosody.
    /// <paramref name="asr"/> is <c>[1, 512, T]</c> (length-regulated text-encoder output).
    /// <paramref name="f0"/> and <paramref name="n"/> are <c>[1, 1, 2*T]</c> from the prosody
    /// predictor. <paramref name="styleDecoder"/> is the 128-dim decoder style row <c>[1, 128]</c>.
    /// Returns a 1-D float waveform at <see cref="KokoroConfig.SampleRate"/> (24 kHz).</summary>
    public float[] Forward(IBackend backend, Tensor asr, Tensor f0, Tensor n, Tensor styleDecoder)
    {
        int batch = (int)asr.Shape[0];
        int t = (int)asr.Shape[2];
        if (f0.Shape.Rank != 3 || (int)f0.Shape[1] != 1)
            throw new ArgumentException($"F0 must be [1, 1, 2T], got {f0.Shape}.");

        // F0_conv / N_conv: Conv1d(1,1,k=3,stride=2,pad=1) downsamples the 2T curve to T.
        Tensor f0Down = new(new TensorShape(batch, 1, t), DType.F32);
        backend.Conv1d(f0Down, f0, _f0ConvW!, _f0ConvB, stride: 2, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
        Tensor nDown = new(new TensorShape(batch, 1, t), DType.F32);
        backend.Conv1d(nDown, n, _nConvW!, _nConvB, stride: 2, padLeft: 1, padRight: 1, dilation: 1, groups: 1);

        // asr_res: 1×1 Conv 512 → 64.
        Tensor asrRes = new(new TensorShape(batch, 64, t), DType.F32);
        backend.Conv1d(asrRes, asr, _asrResW!, _asrResB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

        // encode: AdainResBlk1d on cat([asr(512), F0(1), N(1)]) → [1, 1024, T].
        Tensor encIn = ConcatChannels(asr, f0Down, nDown);
        Tensor x = _encode.Forward(backend, encIn, styleDecoder);
        encIn.Dispose();

        // decode: each block runs on cat([x, asr_res, F0, N], s). Last block upsamples 2×.
        for (int i = 0; i < 4; i++)
        {
            Tensor decIn = ConcatChannels(x, asrRes, f0Down, nDown);
            x.Dispose();
            x = _decode[i].Forward(backend, decIn, styleDecoder);
            decIn.Dispose();
        }
        asrRes.Dispose();
        nDown.Dispose();
        f0Down.Dispose();

        // generator consumes the ORIGINAL 2T F0 curve (not the downsampled one).
        float[] audio = _useHifiGan
            ? _hifiGan!.Forward(backend, x, styleDecoder, f0)
            : RunGenerator(backend, x, styleDecoder, f0);
        x.Dispose();
        return audio;
    }

    /// <summary>iSTFTNet generator: harmonic-plus-noise source → forward STFT → two ConvTranspose1d
    /// upsamples (10×, 6×) with per-level noise injection + 3-kernel MRF AdaIN/Snake resblocks →
    /// conv_post → magnitude/phase iSTFT. Mirrors <c>kokoro/istftnet.py</c> <c>Generator.forward</c>.</summary>
    private float[] RunGenerator(IBackend backend, Tensor x0, Tensor s, Tensor f0)
    {
        KokoroConfig.IStftNetConfig g = _cfg.IStftNet;
        int hop = g.GenIstftHopSize;       // 5
        int nFft = g.GenIstftNFft;         // 20
        int upProd = 1;
        foreach (int u in g.UpsampleRates) upProd *= u;     // 60
        int f0UpScale = upProd * hop;      // 300

        // 1. Harmonic source at audio rate → forward STFT → [1, n_fft+2, frames].
        float[] harSource = NsfVocoderDsp.GenerateHarmonicSource(f0, f0UpScale, _cfg.SampleRate, harmonics: 9, _mSourceW!, _mSourceB!);
        Tensor har = NsfVocoderDsp.ForwardStftMagPhase(harSource, nFft, hop);

        Tensor x = new(x0.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x0.DataPointer, (void*)x.DataPointer, x0.ElementCount * 4, x0.ElementCount * 4);

        int numUp = g.UpsampleRates.Length;     // 2
        for (int i = 0; i < numUp; i++)
        {
            backend.LeakyRelu(x, x, 0.1f);

            // Source branch: noise_convs[i](har) → noise_res[i](·, s).
            Tensor xSrc = NoiseConv(backend, har, i);
            Tensor xSrcRes = _noiseRes[i].Forward(backend, xSrc, s);
            xSrc.Dispose();

            // Upsample branch: ups[i](x), then reflection-pad (1,0) at the last stage.
            Tensor xUp = UpsampleConvT(backend, x, i);
            x.Dispose();
            x = xUp;
            if (i == numUp - 1)
            {
                Tensor xp = NsfVocoderDsp.ReflectionPadLeft1(x);
                x.Dispose();
                x = xp;
            }

            NsfVocoderDsp.AddInPlaceCropped(x, xSrcRes);
            xSrcRes.Dispose();

            // MRF: average of the 3 resblocks at this level.
            Tensor acc = _resblocks[i * 3].Forward(backend, x, s);
            for (int j = 1; j < 3; j++)
            {
                Tensor rb = _resblocks[i * 3 + j].Forward(backend, x, s);
                NsfVocoderDsp.AddInPlaceCropped(acc, rb);
                rb.Dispose();
            }
            x.Dispose();
            NsfVocoderDsp.ScaleInPlace(acc, 1f / 3f);
            x = acc;
        }
        har.Dispose();

        backend.LeakyRelu(x, x, 0.01f);
        Tensor xpad = NsfVocoderDsp.ReflectionPadLeft1(x);
        x.Dispose();
        x = xpad;

        int frames = (int)x.Shape[2];
        Tensor post = new(new TensorShape(1, nFft + 2, frames), DType.F32);
        backend.Conv1d(post, x, _convPostW!, _convPostB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        x.Dispose();

        float[] audio = NsfVocoderDsp.IstftHead(post, nFft, hop);
        post.Dispose();
        return audio;
    }

    /// <summary>noise_convs[i]: Conv1d over the <c>[1, n_fft+2, frames]</c> source spectrogram.
    /// i=0 strides down to match the first upsample level (k=12, s=6, pad=3); i=1 is a 1×1 keep.</summary>
    private Tensor NoiseConv(IBackend backend, Tensor har, int i)
    {
        Tensor w = _noiseConvW[i]!;
        int outCh = (int)w.Shape[0];
        int kernel = (int)w.Shape[2];
        int stride = i == 0 ? _cfg.IStftNet.UpsampleRates[1] : 1;   // prod(rates[i+1:]) = 6 for i=0
        int pad = i == 0 ? (stride + 1) / 2 : 0;
        int inLen = (int)har.Shape[2];
        int outLen = (inLen + 2 * pad - (kernel - 1) - 1) / stride + 1;
        Tensor outT = new(new TensorShape(1, outCh, outLen), DType.F32);
        backend.Conv1d(outT, har, w, _noiseConvB[i], stride: stride, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
        return outT;
    }

    /// <summary>ups[i]: ConvTranspose1d upsample by the configured rate (kernel k, stride u,
    /// symmetric pad (k-u)/2). Halves channels (512→256→128).</summary>
    private Tensor UpsampleConvT(IBackend backend, Tensor x, int i)
    {
        Tensor w = _upsW[i]!;
        int outCh = (int)w.Shape[1];
        int kernel = (int)w.Shape[2];
        int stride = _cfg.IStftNet.UpsampleRates[i];
        int pad = (kernel - stride) / 2;
        int inLen = (int)x.Shape[2];
        int outLen = (inLen - 1) * stride + (kernel - 1) + 1 - 2 * pad;
        Tensor outT = new(new TensorShape(1, outCh, outLen), DType.F32);
        backend.ConvTranspose1d(outT, x, w, _upsB[i], stride: stride, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
        return outT;
    }

    /// <summary>Concatenates channels-first tensors along the channel dim (all share batch + length).</summary>
    private static Tensor ConcatChannels(params Tensor[] parts)
    {
        int batch = (int)parts[0].Shape[0];
        int t = (int)parts[0].Shape[2];
        int totalCh = 0;
        foreach (Tensor p in parts) totalCh += (int)p.Shape[1];
        Tensor outT = new(new TensorShape(batch, totalCh, t), DType.F32);
        float* op = (float*)outT.DataPointer;
        int chOffset = 0;
        foreach (Tensor p in parts)
        {
            int c = (int)p.Shape[1];
            float* ip = (float*)p.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int cc = 0; cc < c; cc++)
                {
                    long src = ((long)b * c + cc) * t;
                    long dst = ((long)b * totalCh + chOffset + cc) * t;
                    for (int j = 0; j < t; j++) op[dst + j] = ip[src + j];
                }
            chOffset += c;
        }
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core =
        [
            _asrResW, _asrResB, _f0ConvW, _f0ConvB, _nConvW, _nConvB,
            _mSourceW, _mSourceB, _convPostW, _convPostB,
        ];
        foreach (Tensor? x in core) if (x is not null) yield return x;
        for (int i = 0; i < 2; i++)
        {
            if (_upsW[i] is not null) yield return _upsW[i]!;
            if (_upsB[i] is not null) yield return _upsB[i]!;
            if (_noiseConvW[i] is not null) yield return _noiseConvW[i]!;
            if (_noiseConvB[i] is not null) yield return _noiseConvB[i]!;
            foreach (Tensor t in _noiseRes[i].EnumerateWeights()) yield return t;
        }
        foreach (Tensor t in _encode.EnumerateWeights()) yield return t;
        foreach (AdaResLoader d in _decode) foreach (Tensor t in d.EnumerateWeights()) yield return t;
        foreach (AdaSnakeResLoader r in _resblocks) foreach (Tensor t in r.EnumerateWeights()) yield return t;
    }
}

/// <summary>Thin loader for the four AdainResBlk1d blocks used in the decoder's
/// <c>encode</c> and <c>decode</c> paths. The decoder's blocks are identical in topology
/// to the predictor's <see cref="KokoroAdainResBlk1d"/> — same residual + shortcut shape,
/// same weight keys — so we reuse that class verbatim and the loader exists only to
/// own the instance and route weight prefixes consistently.</summary>
internal sealed class AdaResLoader
{
    private readonly KokoroAdainResBlk1d _block;

    public AdaResLoader(int dimIn, int dimOut, bool upsample, int kernel)
    {
        _block = new KokoroAdainResBlk1d(dimIn, dimOut, upsample);
        _ = kernel;     // accepted for forward symmetry; the block hard-codes k=3 for now.
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        => _block.LoadWeights(w, prefix);

    public Tensor Forward(IBackend backend, Tensor x, Tensor style) => _block.Forward(backend, x, style);

    public IEnumerable<Tensor> EnumerateWeights() => _block.EnumerateWeights();
}

/// <summary>Loader for the HiFi-GAN style <c>AdaINResBlock1</c> used in the generator's <c>resblocks</c> and <c>noise_res</c> — three parallel dilated branches (dilation 1, 3, 5, same kernel size), each <c>AdaIN1d → Snake → Conv1d</c> twice (6 conv layers, 6 AdaIN modules, 6 Snake alphas total).</summary>
internal sealed unsafe class AdaSnakeResLoader
{
    private readonly int _channels;
    private readonly int _kernel;
    private readonly int[] _dilations;

    // For each of 3 dilation branches:
    private readonly Tensor?[] _adain1FcW = new Tensor[3];
    private readonly Tensor?[] _adain1FcB = new Tensor[3];
    private readonly Tensor?[] _adain2FcW = new Tensor[3];
    private readonly Tensor?[] _adain2FcB = new Tensor[3];
    private readonly Tensor?[] _alpha1 = new Tensor[3];     // Snake alpha [1, C, 1]
    private readonly Tensor?[] _alpha2 = new Tensor[3];
    private readonly Tensor?[] _convs1W = new Tensor[3];     // composed [C, C, K]
    private readonly Tensor?[] _convs1B = new Tensor[3];
    private readonly Tensor?[] _convs2W = new Tensor[3];
    private readonly Tensor?[] _convs2B = new Tensor[3];

    public AdaSnakeResLoader(int channels, int kernel, int[] dilations)
    {
        if (dilations.Length != 3) throw new ArgumentException("AdaSnakeResLoader expects 3 dilations.");
        _channels = channels;
        _kernel = kernel;
        _dilations = dilations;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < 3; i++)
        {
            _adain1FcW[i] = WhisperOps.EnsureF32(w[$"{prefix}.adain1.{i}.fc.weight"]);
            _adain1FcB[i] = WhisperOps.EnsureF32(w[$"{prefix}.adain1.{i}.fc.bias"]);
            _adain2FcW[i] = WhisperOps.EnsureF32(w[$"{prefix}.adain2.{i}.fc.weight"]);
            _adain2FcB[i] = WhisperOps.EnsureF32(w[$"{prefix}.adain2.{i}.fc.bias"]);
            _alpha1[i] = WhisperOps.EnsureF32(w[$"{prefix}.alpha1.{i}"]);
            _alpha2[i] = WhisperOps.EnsureF32(w[$"{prefix}.alpha2.{i}"]);
            _convs1W[i] = WeightNorm.Compose(w, $"{prefix}.convs1.{i}");
            _convs1B[i] = WhisperOps.EnsureF32(w[$"{prefix}.convs1.{i}.bias"]);
            _convs2W[i] = WeightNorm.Compose(w, $"{prefix}.convs2.{i}");
            _convs2B[i] = WhisperOps.EnsureF32(w[$"{prefix}.convs2.{i}.bias"]);
        }
    }

    /// <summary>HiFi-GAN <c>AdaINResBlock1</c> forward over channels-first <c>[1, C, T]</c>: three
    /// sequential dilated branches, each <c>AdaIN1d → Snake → dilated Conv1d → AdaIN1d → Snake →
    /// Conv1d</c> added back as a residual (<c>x = convChain(x) + x</c>). Channel count and length are
    /// preserved (same-padding). Snake α is the per-channel learnable <c>[1, C, 1]</c>; β is unused
    /// (StyleTTS2 uses plain Snake <c>x + (1/α)·sin²(αx)</c>).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor style)
    {
        int batch = (int)x.Shape[0];
        int t = (int)x.Shape[2];
        Tensor cur = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)cur.DataPointer, x.ElementCount * 4, x.ElementCount * 4);

        for (int i = 0; i < 3; i++)
        {
            int pad1 = (_kernel - 1) * _dilations[i] / 2;
            int pad2 = (_kernel - 1) / 2;

            (Tensor g1, Tensor b1) = KokoroOps.StyleToGammaBeta(backend, _adain1FcW[i]!, _adain1FcB[i]!, style, _channels);
            Tensor xt = new(cur.Shape, DType.F32);
            backend.AdaInstanceNorm1d(xt, cur, g1, b1, 1e-5f);
            g1.Dispose(); b1.Dispose();
            backend.Snake(xt, xt, _alpha1[i]!, null);

            Tensor c1 = new(new TensorShape(batch, _channels, t), DType.F32);
            backend.Conv1d(c1, xt, _convs1W[i]!, _convs1B[i], stride: 1, padLeft: pad1, padRight: pad1, dilation: _dilations[i], groups: 1);
            xt.Dispose();

            (Tensor g2, Tensor b2) = KokoroOps.StyleToGammaBeta(backend, _adain2FcW[i]!, _adain2FcB[i]!, style, _channels);
            Tensor xt2 = new(c1.Shape, DType.F32);
            backend.AdaInstanceNorm1d(xt2, c1, g2, b2, 1e-5f);
            g2.Dispose(); b2.Dispose(); c1.Dispose();
            backend.Snake(xt2, xt2, _alpha2[i]!, null);

            Tensor c2 = new(new TensorShape(batch, _channels, t), DType.F32);
            backend.Conv1d(c2, xt2, _convs2W[i]!, _convs2B[i], stride: 1, padLeft: pad2, padRight: pad2, dilation: 1, groups: 1);
            xt2.Dispose();

            // Residual: cur += c2.
            float* cp = (float*)cur.DataPointer;
            float* c2p = (float*)c2.DataPointer;
            long n = cur.ElementCount;
            for (long k = 0; k < n; k++) cp[k] += c2p[k];
            c2.Dispose();
        }
        return cur;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < 3; i++)
        {
            Tensor?[] all =
            [
                _adain1FcW[i], _adain1FcB[i], _adain2FcW[i], _adain2FcB[i],
                _alpha1[i], _alpha2[i], _convs1W[i], _convs1B[i], _convs2W[i], _convs2B[i],
            ];
            foreach (Tensor? x in all) if (x is not null) yield return x;
        }
    }
}
