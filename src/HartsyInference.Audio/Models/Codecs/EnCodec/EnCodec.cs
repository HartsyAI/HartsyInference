using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.EnCodec;

/// <summary>Top-level EnCodec neural audio codec. Wraps the
/// <see cref="SeaNetEncoder"/>, <see cref="ResidualVectorQuantizer"/>, and
/// <see cref="SeaNetDecoder"/> into the standard <c>Encode</c> / <c>Decode</c>
/// surface.
///
/// <para>State-dict layout matches Meta's <c>facebook/encodec_24khz</c>:</para>
/// <list type="bullet">
///   <item><c>encoder.model.*</c> — SEANet encoder</item>
///   <item><c>quantizer.vq.layers.{q}._codebook.embed</c> — RVQ codebooks</item>
///   <item><c>decoder.model.*</c> — SEANet decoder</item>
/// </list>
///
/// <para>Used by Bark (8 codebooks, 6 kbps), MusicGen (4 codebooks mono, 8 codebooks
/// stereo), and AudioGen (4 codebooks). The active codebook count is set via the
/// <paramref name="nQ"/> parameter on <see cref="Encode"/> / <see cref="Decode"/>;
/// fewer codebooks = lower bandwidth + lower reconstruction quality.</para>
///
/// <para>Inference-only: no training path, no straight-through estimator, no
/// commitment loss. The encoder and decoder run independently — typical pipelines
/// encode reference audio once and decode generated tokens many times.</para></summary>
public sealed class EnCodec
{
    public EnCodecConfig Config { get; }
    public int LatentDim => Config.LatentDim;
    public int SampleRate => Config.SampleRate;
    public int FrameRate => Config.FrameRate;
    public int MaxCodebooks => Config.VqMaxCodebooks;

    private readonly SeaNetEncoder _encoder;
    private readonly ResidualVectorQuantizer _quantizer;
    private readonly SeaNetDecoder _decoder;

    public EnCodec(EnCodecConfig config)
    {
        Config = config;
        _encoder = new SeaNetEncoder(config, "encoder");
        _quantizer = new ResidualVectorQuantizer(config.VqMaxCodebooks, config.VqCodebookSize, config.LatentDim);
        _decoder = new SeaNetDecoder(config, "decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _encoder.LoadWeights(w);
        _quantizer.LoadWeights(w, "quantizer");
        _decoder.LoadWeights(w);
    }

    /// <summary>Encodes raw PCM to integer RVQ codes. <paramref name="pcm"/> is
    /// channels-first <c>[batch, channels=1, T_pcm]</c>; output is <c>[nQ, batch, T_frames]</c>
    /// where <c>T_frames = T_pcm / hop</c> (320 for the 24 kHz model).
    ///
    /// <para><paramref name="nQ"/> defaults to <see cref="EnCodecConfig.ActiveCodebooks"/>.
    /// Pass a smaller value for a lower-bandwidth encoding; larger values up to
    /// <see cref="EnCodecConfig.VqMaxCodebooks"/> yield higher quality.</para></summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm, int? nQ = null)
    {
        int activeCodebooks = nQ ?? Config.ActiveCodebooks;
        Tensor latentCf = _encoder.Forward(backend, pcm, batch, tPcm);
        int tFrames = (int)latentCf.Shape[2];
        Tensor codes = _quantizer.Encode(latentCf, batch, tFrames, activeCodebooks);
        latentCf.Dispose();
        return codes;
    }

    /// <summary>Decodes RVQ codes back to PCM. <paramref name="codes"/> is
    /// <c>[nQ, batch, T_frames]</c> Int32. Returns channels-first
    /// <c>[batch, channels=1, T_frames * 320]</c>.</summary>
    public unsafe Tensor Decode(IBackend backend, Tensor codes, int batch, int tFrames)
    {
        Tensor latentCf = _quantizer.Decode(codes, batch, tFrames);
        string? dl = Environment.GetEnvironmentVariable("MUSICGEN_DUMP_LATENT");   // debug: quantizer-vs-decoder isolation
        if (!string.IsNullOrEmpty(dl))
        {
            long nEl = latentCf.Shape.ElementCount;
            byte[] ob = new ReadOnlySpan<byte>((byte*)latentCf.DataPointer, (int)(nEl * 4)).ToArray();
            System.IO.File.WriteAllBytes(dl, ob);
        }
        Tensor pcm = _decoder.Forward(backend, latentCf, batch, tFrames);
        latentCf.Dispose();
        return pcm;
    }

    /// <summary>Parity-debug only: installs a per-decoder-stage activation hook (HF Sequential index → tensor).</summary>
    public void SetDecoderDebugHook(Action<int, Tensor>? hook) => _decoder.DebugStageHook = hook;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _quantizer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }
}
