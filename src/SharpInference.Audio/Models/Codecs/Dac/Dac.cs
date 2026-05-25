using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Codecs.Dac;

/// <summary>Top-level Descript Audio Codec (DAC). Ties together <see cref="DacEncoder"/>,
/// <see cref="DacResidualVectorQuantizer"/>, and <see cref="DacDecoder"/> into the
/// standard <c>Encode</c> / <c>Decode</c> surface. State-dict layout matches
/// <c>descript/dac_*khz</c>:</summary>
/// <list type="bullet">
///   <item><c>encoder.block.*</c> — DAC encoder</item>
///   <item><c>quantizer.quantizers.{i}.{in_proj|out_proj|codebook}.*</c> — RVQ</item>
///   <item><c>decoder.model.*</c> — DAC decoder</item>
/// </list>
///
/// <para>Used by IndexTTS 1.5/2 (44.1 kHz), Higgs Audio v2 acoustic branch (44.1 kHz),
/// and Spark-TTS HiFi-GAN wave generator (16 kHz). Three sample-rate variants share
/// the same architecture; pick the matching preset via
/// <see cref="DacConfig.Dac44kHz"/> / <see cref="DacConfig.Dac24kHz"/> / <see cref="DacConfig.Dac16kHz"/>.</para>
public sealed class Dac
{
    public DacConfig Config { get; }
    public int LatentDim => Config.LatentDim;
    public int SampleRate => Config.SampleRate;
    public int FrameRate => Config.FrameRate;
    public int MaxCodebooks => Config.NCodebooks;

    private readonly DacEncoder _encoder;
    private readonly DacResidualVectorQuantizer _quantizer;
    private readonly DacDecoder _decoder;

    public Dac(DacConfig config)
    {
        Config = config;
        _encoder = new DacEncoder(config, "encoder");
        _quantizer = new DacResidualVectorQuantizer(
            nCodebooks: config.NCodebooks,
            codebookSize: config.CodebookSize,
            codebookDim: config.CodebookDim,
            latentDim: config.LatentDim);
        _decoder = new DacDecoder(config, "decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _encoder.LoadWeights(w);
        _quantizer.LoadWeights(w, "quantizer");
        _decoder.LoadWeights(w);
    }

    /// <summary>Encodes PCM to RVQ codes. <paramref name="pcm"/> is
    /// <c>[B, 1, T_pcm]</c>; output is <c>[nQ, B, T_frames]</c> Int32 where
    /// <c>T_frames = T_pcm / hop</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm, int? nQ = null)
    {
        Tensor latent = _encoder.Forward(backend, pcm, batch, tPcm);
        int tFrames = (int)latent.Shape[2];
        Tensor codes = _quantizer.Encode(backend, latent, batch, tFrames, nQ);
        latent.Dispose();
        return codes;
    }

    /// <summary>Decodes RVQ codes back to PCM. <paramref name="codes"/> is
    /// <c>[nQ, B, T_frames]</c>; output is <c>[B, 1, T_frames * hop]</c>.</summary>
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
