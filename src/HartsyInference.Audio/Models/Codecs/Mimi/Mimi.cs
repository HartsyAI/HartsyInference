using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Mimi neural audio codec (Kyutai). Three-stage pipeline:
/// <list type="number">
///   <item><see cref="EnCodec.SeaNetEncoder"/>-equivalent encoder → 12.5 Hz latent</item>
///   <item><see cref="MimiTransformer"/> — small causal transformer for context-aware quantization</item>
///   <item>Hybrid quantizer: 1 semantic VQ (codebook 0) + 7 acoustic RVQ codebooks (1..7)</item>
/// </list>
/// Decode reverses: codes → quantizer → transformer (decoder side) → decoder → PCM.
///
/// <para>For our port we reuse the existing EnCodec SeaNet encoder/decoder by lifting the
/// Mimi config into an <see cref="EnCodecConfig"/>. The transformer-of-codecs and the
/// hybrid quantizer are Mimi-specific.</para></summary>
public sealed class Mimi
{
    public MimiConfig Config { get; }
    public int SampleRate => Config.SampleRate;
    public int FrameRate => Config.FrameRate;
    public int NCodebooks => Config.TotalCodebooks;

    private readonly SeaNetEncoder _encoder;
    private readonly MimiTransformer _encoderTransformer;
    private readonly MimiTransformer _decoderTransformer;
    private readonly DacResidualVectorQuantizer _quantizer;
    private readonly SeaNetDecoder _decoder;

    public Mimi(MimiConfig cfg)
    {
        Config = cfg;
        EnCodecConfig enc = LiftToEnCodec(cfg);
        _encoder = new SeaNetEncoder(enc, "encoder");
        _encoderTransformer = new MimiTransformer(cfg, "encoder_transformer");
        _decoderTransformer = new MimiTransformer(cfg, "decoder_transformer");
        _quantizer = new DacResidualVectorQuantizer(
            nCodebooks: cfg.TotalCodebooks,
            codebookSize: cfg.CodebookSize,
            codebookDim: cfg.CodebookDim,
            latentDim: cfg.LatentDim);
        _decoder = new SeaNetDecoder(enc, "decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _encoder.LoadWeights(w);
        _encoderTransformer.LoadWeights(w);
        _decoderTransformer.LoadWeights(w);
        _quantizer.LoadWeights(w, "quantizer");
        _decoder.LoadWeights(w);
    }

    /// <summary>Encodes PCM to <c>[nQ, B, T_frames]</c> Int32 codes.</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        // 1) SEANet encoder.
        Tensor latentCf = _encoder.Forward(backend, pcm, batch, tPcm);
        int tFrames = (int)latentCf.Shape[2];

        // 2) Transformer-of-codecs (channels-last input).
        Tensor latentCl = new(new TensorShape(batch, tFrames, Config.LatentDim), DType.F32);
        backend.Transpose2D(latentCl, latentCf, Config.LatentDim, tFrames);
        latentCf.Dispose();
        Tensor contextualCl = _encoderTransformer.Forward(backend, latentCl, batch, tFrames);
        latentCl.Dispose();

        // 3) Back to channels-first for the quantizer.
        Tensor contextualCf = new(new TensorShape(batch, Config.LatentDim, tFrames), DType.F32);
        backend.Transpose2D(contextualCf, contextualCl, tFrames, Config.LatentDim);
        contextualCl.Dispose();

        // 4) Quantize.
        Tensor codes = _quantizer.Encode(backend, contextualCf, batch, tFrames);
        contextualCf.Dispose();
        return codes;
    }

    /// <summary>Decodes <c>[nQ, B, T_frames]</c> codes back to PCM.</summary>
    public Tensor Decode(IBackend backend, Tensor codes, int batch, int tFrames)
    {
        Tensor latentCf = _quantizer.Decode(backend, codes, batch, tFrames);

        // Transformer (decoder side) — channels-last.
        Tensor latentCl = new(new TensorShape(batch, tFrames, Config.LatentDim), DType.F32);
        backend.Transpose2D(latentCl, latentCf, Config.LatentDim, tFrames);
        latentCf.Dispose();
        Tensor contextualCl = _decoderTransformer.Forward(backend, latentCl, batch, tFrames);
        latentCl.Dispose();
        Tensor contextualCf = new(new TensorShape(batch, Config.LatentDim, tFrames), DType.F32);
        backend.Transpose2D(contextualCf, contextualCl, tFrames, Config.LatentDim);
        contextualCl.Dispose();

        // SEANet decoder.
        Tensor pcm = _decoder.Forward(backend, contextualCf, batch, tFrames);
        contextualCf.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encoderTransformer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoderTransformer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _quantizer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }

    /// <summary>Lifts a <see cref="MimiConfig"/> into an <see cref="EnCodecConfig"/>
    /// shape so we can reuse the existing SeaNet encoder/decoder. Mimi's SEANet stack
    /// matches EnCodec's exactly — same Block1D structure, same weight-norm fusion
    /// rules, just different per-stage stride product.</summary>
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
            LstmLayers = 0,     // Mimi uses the transformer-of-codecs instead of LSTM bottleneck
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
