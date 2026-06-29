using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.XCodec;

/// <summary>XCodec — YuE's neural audio codec (the upstream <c>SoundStream</c> in
/// <c>soundstream_hubert_new.py</c>). The real decode path YuE Stage-1 needs is:
/// <code>
///   code indices [n_q, B, T]
///     -> quantizer.decode  (EMA-VQ ResidualVectorQuantization: F.embedding lookup + sum)  -> [B, 1024, T]
///     -> fc_post2          (Linear 1024 -> 256, applied over channels = 1x1 conv)          -> [B, 256, T]
///     -> decoder_2         (descript dac2.Decoder: input=256, channels=1024, rates=[8,5,4,2]) -> [B, 1, S]
/// </code>
/// The latent dimension is <c>D + 768 = 1024</c> (acoustic D=256 concatenated with the 768-d semantic
/// branch); <c>fc_post2</c> projects that fused latent back down to the acoustic D=256 before the conv
/// decoder. The 320× hop (8·5·4·2) gives 50 Hz frames / 16 kHz audio.
///
/// <para><b>Architecture vs DAC.</b> This is NOT the factorized cosine-codebook DAC RVQ. The quantizer is an
/// EMA <c>EuclideanCodebook</c> ResidualVQ — each of the 12 codebooks is a single <c>[1024, 1024]</c> table
/// (<c>quantizer.vq.layers.{i}._codebook.embed</c>); decode is a pure table gather + sum with no
/// in/out projection. YuE Stage-1 emits only vocal codebook-0, so decode is invoked with <c>n_q = 1</c>.
/// The waveform decoder <c>decoder_2</c> IS structurally a descript dac2.Decoder, so it is reused verbatim
/// via <see cref="DacDecoder"/> (its final <c>tanh</c> is disabled — the upstream <c>nn.Tanh()</c> is
/// commented out).</para>
///
/// <para><b>State-dict roots</b> (after the converter strips <c>codec_model.</c> and renames
/// <c>decoder_2.*</c> -> <c>decoder.*</c>): <c>quantizer.vq.layers.{i}._codebook.embed</c>,
/// <c>fc_post2.{weight,bias}</c>, <c>decoder.model.*</c>. The training-only semantic branch
/// (HuBERT + encoder/decoder_semantic + fc_prior/fc_post1) is dropped at conversion.</para></summary>
public sealed unsafe class XCodec
{
    public XCodecConfig Config { get; }
    public int LatentDim => Config.LatentDim;
    public int SampleRate => Config.SampleRate;
    public int FrameRate => Config.FrameRate;
    public int NCodebooks => Config.NCodebooks;

    private readonly XCodecEmaResidualVectorQuantizer _quantizer;
    private readonly DacDecoder _decoder;

    private Tensor? _fcPost2W;   // [D_out, latent, 1] — fc_post2 reshaped as a 1x1 conv weight
    private Tensor? _fcPost2B;   // [D_out]

    public XCodec(XCodecConfig cfg)
    {
        Config = cfg;
        _quantizer = new XCodecEmaResidualVectorQuantizer(cfg.NCodebooks, cfg.CodebookSize, cfg.LatentDim);
        _decoder = new DacDecoder(cfg.ToDacConfig(), "decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _quantizer.LoadWeights(w, "quantizer");

        // fc_post2: Linear(latent_dim -> acoustic_dim). Applied over channels of a [B, latent, T] tensor, i.e.
        // identical to a 1x1 Conv1d with weight [D_out, latent, 1]. Reshape the [D_out, latent] matrix in place.
        Tensor fc = WhisperOps.EnsureF32(w["fc_post2.weight"]);   // [D_out, latent]
        if (fc.Shape[0] != Config.AcousticDim || fc.Shape[1] != Config.LatentDim)
            throw new InvalidOperationException(
                $"XCodec fc_post2.weight shape {fc.Shape} != [{Config.AcousticDim}, {Config.LatentDim}].");
        _fcPost2W = fc.Reshape(new TensorShape(Config.AcousticDim, Config.LatentDim, 1));
        _fcPost2B = WhisperOps.EnsureF32(w["fc_post2.bias"]);

        _decoder.LoadWeights(w);
    }

    /// <summary>Decodes code indices to a 16 kHz waveform. <paramref name="codes"/> may be channels-first
    /// integer indices either as <c>[B, n_q, T]</c> (pipeline-style, F32 or I32) — see
    /// <see cref="DecodeFromCodes"/>. This overload takes the EMA codes laid out <c>[n_q, B, T]</c> I32.</summary>
    public Tensor Decode(IBackend backend, Tensor codes, int batch, int tFrames)
    {
        if (_fcPost2W is null) throw new InvalidOperationException("XCodec weights not loaded.");

        // 1) EMA-VQ ResidualVQ decode: lookup + sum -> [B, latent, T].
        Tensor latent = _quantizer.Decode(backend, codes, batch, tFrames);

        // 2) fc_post2 as a 1x1 conv: [B, latent, T] -> [B, acoustic, T].
        Tensor acoustic = new(new TensorShape(batch, Config.AcousticDim, tFrames), DType.F32);
        backend.Conv1d(acoustic, latent, _fcPost2W!, _fcPost2B,
            stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        latent.Dispose();

        // 3) decoder_2 (dac2.Decoder) -> [B, 1, S].
        Tensor pcm = _decoder.Forward(backend, acoustic, batch, tFrames);
        acoustic.Dispose();
        return pcm;
    }

    /// <summary>Decodes from a single-codebook (cb0) index list — the YuE Stage-1 vocal stream. Builds the
    /// <c>[n_q=1, B=1, T]</c> I32 grid the EMA decode expects and runs the full codec.</summary>
    public Tensor DecodeCb0(IBackend backend, ReadOnlySpan<int> cb0Indices)
    {
        int t = cb0Indices.Length;
        Tensor codes = new(new TensorShape(1, 1, t), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int i = 0; i < t; i++)
        {
            int idx = cb0Indices[i];
            cp[i] = (uint)idx < (uint)Config.CodebookSize ? idx : 0;
        }
        Tensor pcm = Decode(backend, codes, batch: 1, tFrames: t);
        codes.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _quantizer.EnumerateWeights()) yield return t;
        if (_fcPost2W is not null) yield return _fcPost2W;
        if (_fcPost2B is not null) yield return _fcPost2B;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }
}

public sealed record XCodecConfig
{
    public int SampleRate { get; init; } = 16_000;
    public int Channels { get; init; } = 1;

    /// <summary>Acoustic latent dim D (the DAC encoder bottleneck, = fc_post2 output / decoder_2 input).</summary>
    public int AcousticDim { get; init; } = 256;

    /// <summary>Semantic branch dim (HuBERT/RepCodec), concatenated with the acoustic branch.</summary>
    public int SemanticDim { get; init; } = 768;

    /// <summary>decoder_2 initial channel count (descript "channels" arg).</summary>
    public int DecoderDim { get; init; } = 1_024;

    /// <summary>Decoder upsample factors. 8·5·4·2 = 320× -> 50 Hz / 16 kHz.</summary>
    public IReadOnlyList<int> DecoderRates { get; init; } = [8, 5, 4, 2];

    /// <summary>EMA-VQ codebook count (full RVQ is 12; YuE Stage-1 decode uses only cb0).</summary>
    public int NCodebooks { get; init; } = 12;
    public int CodebookSize { get; init; } = 1_024;
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 3, 9];

    /// <summary>The fused RVQ latent dimension (acoustic + semantic = 1024).</summary>
    public int LatentDim => AcousticDim + SemanticDim;

    public int FrameRate
    {
        get
        {
            int p = 1;
            for (int i = 0; i < DecoderRates.Count; i++) p *= DecoderRates[i];
            return SampleRate / p;
        }
    }

    public static XCodecConfig XCodec16kHz => new();

    /// <summary>Lifts this XCodec config into a <see cref="DacConfig"/> so the existing
    /// <see cref="DacDecoder"/> drives the <c>decoder_2</c> (dac2.Decoder) verbatim. The decoder input
    /// channels = <see cref="AcousticDim"/>; initial channels = <see cref="DecoderDim"/>; the transposed
    /// convs use descript's dim-0 weight-norm and the final tanh is disabled (upstream commented it out).</summary>
    public DacConfig ToDacConfig() => new()
    {
        SampleRate = SampleRate,
        Channels = Channels,
        DecoderDim = DecoderDim,
        DecoderRates = DecoderRates,
        ResidualDilations = ResidualDilations,
        StemKernelSize = 7,
        DecoderFinalKernelSize = 7,
        TransposeWeightNormDim0 = true,
        DecoderFinalTanh = false,
    };
}
