using SharpInference.Audio.Layers;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Kokoro;

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
/// <para><b>Status:</b> this class loads <i>all</i> decoder weights — every key under
/// <c>decoder.*</c> is materialized via <see cref="LoadWeights"/>. The weight-loading
/// side is therefore <i>complete</i> and the safetensors mapping is verifiable. The
/// forward pass however is a documented <b>placeholder</b> that synthesizes a simple
/// F0-modulated sine-wave directly from the predicted F0 curve. This proves the
/// prosody → audio path runs end-to-end (durations and pitch take effect) so the user
/// can audibly QA the predictor, while the high-fidelity HiFi-GAN+iSTFT forward is
/// staged as a separate follow-up. Each TODO marker below documents the missing piece.</para></summary>
internal sealed unsafe class KokoroIStftNetDecoder
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

    public KokoroIStftNetDecoder(KokoroConfig cfg)
    {
        _cfg = cfg;
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

    /// <summary>Forward pass — runs the full decoder over the predicted prosody features.
    /// <paramref name="asr"/> is <c>[1, 512, T_total]</c> (the text-encoder output after
    /// length regulation). <paramref name="f0"/> and <paramref name="n"/> are
    /// <c>[1, 1, 2*T_total]</c> from the prosody predictor. <paramref name="styleDecoder"/>
    /// is the 128-dim decoder half of the voice-pack style row <c>[1, 128]</c>.
    /// Returns a 1-D float waveform at <see cref="KokoroConfig.SampleRate"/>.
    ///
    /// <para><b>Placeholder implementation</b>: this method does <i>not</i> currently run
    /// the full HiFi-GAN+iSTFT generator. Instead it directly synthesizes a sine wave at
    /// the predicted F0 frequency, modulated by the predicted energy curve. This proves
    /// the prosody → audio path is wired correctly and produces something audible from
    /// the predicted prosody, so the predictor can be audibly QA'd before the full
    /// generator forward is dialed in. See class XML for the staged TODOs.</para></summary>
    public float[] Forward(IBackend backend, Tensor asr, Tensor f0, Tensor n, Tensor styleDecoder)
    {
        // The full generator forward will live here. For now, synthesize F0 directly.
        // The F0 from the predictor is at 2 * T_total rate. The model intends 24000 /
        // (2 * 5 * upsample_rates_product) = 24000 / (2 * 5 * 60) = 40 Hz frame rate.
        // We treat the F0 curve as Hz values at that rate, then upsample to audio rate
        // and integrate to get phase.
        int sr = _cfg.SampleRate;
        if (f0.Shape.Rank != 3 || (int)f0.Shape[1] != 1)
            throw new ArgumentException($"F0 must be [1, 1, T], got {f0.Shape}.");
        int tF0 = (int)f0.Shape[2];
        int upsamplePerFrame = sr / 80;     // ~300 samples per F0 frame at 24 kHz / 80 Hz mel rate × 2× F0 upsample
        if (upsamplePerFrame < 1) upsamplePerFrame = 1;
        int audioLen = tF0 * upsamplePerFrame;

        float[] audio = new float[audioLen];
        float* fp = (float*)f0.DataPointer;
        float* np = (float*)n.DataPointer;
        double phase = 0d;
        double twoPiOverSr = 2.0 * Math.PI / sr;
        for (int i = 0; i < tF0; i++)
        {
            // F0 is in log-Hz at training; raw values are roughly in range [-5, 5].
            // For audible output we clamp to a reasonable Hz range — this is a deliberate
            // placeholder until the generator is in.
            float rawF0 = fp[i];
            float hz = MathF.Min(MathF.Max(MathF.Abs(rawF0) * 200f, 50f), 500f);
            float energy = MathF.Tanh(np[i]);     // squash to [-1, 1] for a smooth volume curve
            for (int j = 0; j < upsamplePerFrame; j++)
            {
                phase += twoPiOverSr * hz;
                audio[i * upsamplePerFrame + j] = (float)(0.2 * energy * Math.Sin(phase));
            }
        }
        return audio;
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

    public IEnumerable<Tensor> EnumerateWeights() => _block.EnumerateWeights();
}

/// <summary>Loader for the HiFi-GAN style <c>AdaINResBlock1</c> used in the generator's
/// <c>resblocks</c> and <c>noise_res</c>. Each block has three parallel dilated branches
/// (dilation 1, 3, 5 with the same kernel size). Each branch has AdaIN1d before its conv,
/// a Snake activation with learnable per-channel alpha, then the same pattern again
/// (so 6 conv layers per block, 6 AdaIN modules, 6 Snake alphas total).
///
/// <para>This class only loads the weights — the forward pass is not yet wired into
/// <see cref="KokoroIStftNetDecoder.Forward"/> (placeholder; see class XML).</para></summary>
internal sealed class AdaSnakeResLoader
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
