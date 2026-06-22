using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>Continuous-latent wrapper over the Mimi codec for Pocket-TTS. Pocket-TTS is <b>not</b> a discrete
/// RVQ codec-LM — its codec is a modified Mimi VAE whose quantizer is bypassed (a <c>DummyQuantizer</c>),
/// producing a continuous Gaussian latent at 12.5 Hz. This wrapper composes the same SEANet encoder/decoder +
/// transformer-of-codecs that <see cref="Mimi"/> uses (identical weight keys, so the same <c>mimi.*</c>
/// checkpoint subtree loads), but stops <b>before</b> the RVQ on the encode side and skips it on the decode side.
///
/// <para>We compose rather than edit <see cref="Mimi"/>: the SEANet/transformer modules are assembly-internal,
/// so re-instantiating them here with the same config + prefixes reuses every loaded tensor. The contextual
/// features fed to (and read from) the quantizer in <see cref="Mimi"/> <i>are</i> the continuous latent.</para>
///
/// <para><b>Config-gated:</b> <c>LatentDim</c> and the transformer dims come from the real <c>mimi.*</c> config
/// at load — this class never bakes them in.</para></summary>
public sealed unsafe class MimiContinuousLatent
{
    private readonly MimiConfig _cfg;
    private readonly SeaNetEncoder _encoder;
    private readonly MimiTransformer _encoderTransformer;
    private readonly MimiTransformer _decoderTransformer;
    private readonly SeaNetDecoder _decoder;

    public MimiConfig Config => _cfg;
    public int SampleRate => _cfg.SampleRate;
    public int FrameRate => _cfg.FrameRate;

    /// <summary>Continuous latent channel count (Mimi's pre-quantizer feature dim).</summary>
    public int LatentDim => _cfg.LatentDim;

    public MimiContinuousLatent(MimiConfig cfg)
    {
        _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
        EnCodecConfig enc = LiftToEnCodec(cfg);
        _encoder = new SeaNetEncoder(enc, "encoder");
        _encoderTransformer = new MimiTransformer(cfg, "encoder_transformer");
        _decoderTransformer = new MimiTransformer(cfg, "decoder_transformer");
        _decoder = new SeaNetDecoder(enc, "decoder");
    }

    /// <summary>Loads the codec weights from a Mimi-style key dict (same keys as <see cref="Mimi.LoadWeights"/>,
    /// minus the quantizer subtree which this continuous path never touches).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        if (w is null) throw new ArgumentNullException(nameof(w));
        _encoder.LoadWeights(w);
        _encoderTransformer.LoadWeights(w);
        _decoderTransformer.LoadWeights(w);
        _decoder.LoadWeights(w);
    }

    /// <summary>Encodes PCM <c>[batch, 1, T_pcm]</c> (24 kHz mono) to the continuous Gaussian latent
    /// <c>[batch, LatentDim, T_frames]</c> at 12.5 Hz, bypassing the RVQ. Mirrors <see cref="Mimi.Encode"/> up to
    /// (but not including) the quantizer — the contextual transformer output is the latent.</summary>
    public Tensor EncodeToLatent(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), batch, "batch must be > 0");

        Tensor latentCf = _encoder.Forward(backend, pcm, batch, tPcm);
        int tFrames = (int)latentCf.Shape[2];

        Tensor latentCl = new(new TensorShape(batch, tFrames, _cfg.LatentDim), DType.F32);
        backend.Transpose2D(latentCl, latentCf, _cfg.LatentDim, tFrames);
        latentCf.Dispose();

        Tensor contextualCl = _encoderTransformer.Forward(backend, latentCl, batch, tFrames);
        latentCl.Dispose();

        Tensor contextualCf = new(new TensorShape(batch, _cfg.LatentDim, tFrames), DType.F32);
        backend.Transpose2D(contextualCf, contextualCl, tFrames, _cfg.LatentDim);
        contextualCl.Dispose();
        return contextualCf;
    }

    /// <summary>Decodes a continuous latent <c>[batch, LatentDim, T_frames]</c> back to PCM
    /// <c>[batch, 1, T_frames * hop]</c> at 24 kHz. Mirrors <see cref="Mimi.Decode"/> from the post-quantizer
    /// point onward (decoder-side transformer-of-codecs then SEANet decoder), with no dequantization.</summary>
    public Tensor DecodeFromLatent(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), batch, "batch must be > 0");
        if (tFrames <= 0) throw new ArgumentOutOfRangeException(nameof(tFrames), tFrames, "tFrames must be > 0");

        Tensor latentCl = new(new TensorShape(batch, tFrames, _cfg.LatentDim), DType.F32);
        backend.Transpose2D(latentCl, latent, _cfg.LatentDim, tFrames);

        Tensor contextualCl = _decoderTransformer.Forward(backend, latentCl, batch, tFrames);
        latentCl.Dispose();

        Tensor contextualCf = new(new TensorShape(batch, _cfg.LatentDim, tFrames), DType.F32);
        backend.Transpose2D(contextualCf, contextualCl, tFrames, _cfg.LatentDim);
        contextualCl.Dispose();

        Tensor pcm = _decoder.Forward(backend, contextualCf, batch, tFrames);
        contextualCf.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encoderTransformer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoderTransformer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }

    /// <summary>Lifts a <see cref="MimiConfig"/> into the <see cref="EnCodecConfig"/> shape the SEANet
    /// encoder/decoder consume — same fusion rules as <see cref="Mimi"/>'s private lift, kept identical so the
    /// weight keys line up. <c>LstmLayers = 0</c> because Mimi replaces the LSTM bottleneck with the
    /// transformer-of-codecs.</summary>
    private static EnCodecConfig LiftToEnCodec(MimiConfig cfg)
    {
        return new EnCodecConfig
        {
            SampleRate = cfg.SampleRate,
            Channels = cfg.Channels,
            LatentDim = cfg.LatentDim,
            NFilters = cfg.EncoderDim,
            Ratios = cfg.EncoderRates,
            NResidualLayers = cfg.ResidualDilations.Count,
            KernelSize = 7,
            LastKernelSize = 7,
            ResidualKernelSize = 3,
            DilationBase = 2,
            Compress = 2,
            LstmLayers = 0,
            Causal = true,
            PadMode = "constant",
            Norm = "weight_norm",
            EluAlpha = 1.0f,
            VqCodebookSize = cfg.CodebookSize,
            VqMaxCodebooks = cfg.TotalCodebooks,
            ActiveCodebooks = cfg.TotalCodebooks,
        };
    }
}
