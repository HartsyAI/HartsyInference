using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>Top-level SNAC codec. Unlike EnCodec / DAC the codes are a variable-length
/// list rather than a rectangular tensor — each codebook emits codes at its own
/// temporal rate per <see cref="SnacConfig.VqStrides"/>.
///
/// <para>State-dict roots match <c>hubertsiuzdak/snac_24khz</c>:</para>
/// <list type="bullet">
///   <item><c>encoder.block.*</c></item>
///   <item><c>quantizer.quantizers.*</c></item>
///   <item><c>decoder.model.*</c></item>
/// </list>
///
/// <para>Used by Orpheus TTS (the SNAC 24 kHz variant) and forks.</para></summary>
public sealed class Snac
{
    public SnacConfig Config { get; }
    public int LatentDim => Config.LatentDim;
    public int SampleRate => Config.SampleRate;
    public int FrameRate => Config.FrameRate;
    public int NCodebooks => Config.NCodebooks;
    public IReadOnlyList<int> VqStrides => Config.VqStrides;

    private readonly SnacEncoder _encoder;
    private readonly SnacResidualVectorQuantizer _quantizer;
    private readonly SnacDecoder _decoder;

    public Snac(SnacConfig config)
    {
        Config = config;
        _encoder = new SnacEncoder(config, "encoder");
        _quantizer = new SnacResidualVectorQuantizer(
            nCodebooks: config.NCodebooks,
            codebookSize: config.CodebookSize,
            codebookDim: config.CodebookDim,
            latentDim: config.LatentDim,
            vqStrides: config.VqStrides);
        _decoder = new SnacDecoder(config, "decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _encoder.LoadWeights(w);
        _quantizer.LoadWeights(w, "quantizer");
        _decoder.LoadWeights(w);
    }

    /// <summary>Encodes PCM to hierarchical codes. <paramref name="pcm"/> is
    /// <c>[B, 1, T_pcm]</c>; output is an array of length <see cref="NCodebooks"/> where
    /// entry <c>i</c> has shape <c>[B, T_frames / VqStrides[i]]</c> Int32. Caller owns
    /// disposal of every returned tensor.</summary>
    public Tensor[] Encode(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        Tensor latent = _encoder.Forward(backend, pcm, batch, tPcm);
        int tFrames = (int)latent.Shape[2];
        Tensor[] codes = _quantizer.Encode(backend, latent, batch, tFrames);
        latent.Dispose();
        return codes;
    }

    /// <summary>Decodes hierarchical codes to PCM. <paramref name="codes"/> must have
    /// length <see cref="NCodebooks"/>; entry <c>i</c> shape <c>[B, T_frames / VqStrides[i]]</c>.</summary>
    public Tensor Decode(IBackend backend, IReadOnlyList<Tensor> codes, int batch)
    {
        Tensor latent = _quantizer.Decode(backend, codes, batch);
        int tFrames = (int)latent.Shape[2];
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
