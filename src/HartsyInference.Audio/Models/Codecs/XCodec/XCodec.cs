using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.XCodec;

/// <summary>XCodec — YuE's neural audio codec. Architecture is a SEANet encoder + RVQ
/// (8 codebooks × 1024 entries) + SEANet decoder at 50 Hz / 16 kHz mono. The "X" stands
/// for "cross-modal": separate semantic and acoustic branches are trained jointly.
/// At inference YuE consumes only the acoustic codes, so we focus on the acoustic
/// path here — the semantic branch is a HuBERT extractor used at training time only.
///
/// <para>Architectural notes vs DAC:</para>
/// <list type="bullet">
///   <item>Operates at 16 kHz mono; <c>encoder_rates = [2, 4, 5, 8]</c> → 320× downsample → 50 Hz</item>
///   <item>Same DAC-style residual unit pattern (snake + 2-conv)</item>
///   <item>RVQ has 8 codebooks of 1024 entries each, cosine codebooks (same as DAC)</item>
///   <item>Default <c>codebook_dim = 8</c>, latent_dim = 1024</item>
/// </list>
///
/// <para>State-dict roots match the YuE x-codec-mini-infer checkpoint:</para>
/// <list type="bullet">
///   <item><c>encoder.block.*</c></item>
///   <item><c>quantizer.quantizers.*</c></item>
///   <item><c>decoder.model.*</c></item>
/// </list></summary>
public sealed class XCodec
{
    public XCodecConfig Config { get; }
    public int LatentDim => Config.LatentDim;
    public int SampleRate => Config.SampleRate;
    public int FrameRate => Config.FrameRate;
    public int NCodebooks => Config.NCodebooks;

    private readonly DacEncoder _encoder;       // XCodec encoder is structurally identical to DAC's
    private readonly DacResidualVectorQuantizer _quantizer;
    private readonly DacDecoder _decoder;

    public XCodec(XCodecConfig cfg)
    {
        Config = cfg;
        DacConfig dacView = cfg.ToDacConfig();
        _encoder = new DacEncoder(dacView, "encoder");
        _quantizer = new DacResidualVectorQuantizer(
            nCodebooks: cfg.NCodebooks,
            codebookSize: cfg.CodebookSize,
            codebookDim: cfg.CodebookDim,
            latentDim: cfg.LatentDim);
        _decoder = new DacDecoder(dacView, "decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _encoder.LoadWeights(w);
        _quantizer.LoadWeights(w, "quantizer");
        _decoder.LoadWeights(w);
    }

    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm, int? nQ = null)
    {
        Tensor latent = _encoder.Forward(backend, pcm, batch, tPcm);
        int tFrames = (int)latent.Shape[2];
        Tensor codes = _quantizer.Encode(backend, latent, batch, tFrames, nQ);
        latent.Dispose();
        return codes;
    }

    public Tensor Decode(IBackend backend, Tensor codes, int batch, int tFrames)
    {
        Tensor latent = _quantizer.Decode(backend, codes, batch, tFrames);
        Tensor pcm = _decoder.Forward(backend, latent, batch, tFrames);
        latent.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _quantizer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }
}

public sealed record XCodecConfig
{
    public int SampleRate { get; init; } = 16_000;
    public int Channels { get; init; } = 1;
    public int EncoderDim { get; init; } = 64;
    public int DecoderDim { get; init; } = 1_536;
    public IReadOnlyList<int> EncoderRates { get; init; } = [2, 4, 5, 8];
    public IReadOnlyList<int> DecoderRates { get; init; } = [8, 5, 4, 2];
    public int NCodebooks { get; init; } = 8;
    public int CodebookSize { get; init; } = 1_024;
    public int CodebookDim { get; init; } = 8;
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 3, 9];

    public int LatentDim => EncoderDim * (1 << EncoderRates.Count);

    public int FrameRate
    {
        get
        {
            int p = 1;
            for (int i = 0; i < EncoderRates.Count; i++) p *= EncoderRates[i];
            return SampleRate / p;
        }
    }

    public static XCodecConfig XCodec16kHz => new();

    /// <summary>Lifts this XCodec config into a <see cref="DacConfig"/> shape so the
    /// existing <see cref="DacEncoder"/> / <see cref="DacDecoder"/> can be reused
    /// verbatim — XCodec is essentially DAC with a different sample rate + codebook
    /// count, so we avoid duplicating the encoder/decoder code.</summary>
    public DacConfig ToDacConfig() => new()
    {
        SampleRate = SampleRate,
        Channels = Channels,
        EncoderDim = EncoderDim,
        DecoderDim = DecoderDim,
        EncoderRates = EncoderRates,
        DecoderRates = DecoderRates,
        NCodebooks = NCodebooks,
        CodebookSize = CodebookSize,
        CodebookDim = CodebookDim,
        ResidualDilations = ResidualDilations,
    };
}
