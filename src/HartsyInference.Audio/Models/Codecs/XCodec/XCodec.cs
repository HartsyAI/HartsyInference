using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HubertModel = HartsyInference.Audio.Models.Hubert.Hubert;

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
/// <c>fc_post2.{weight,bias}</c>, <c>decoder.model.*</c>. The semantic-reconstruction head
/// (<c>decoder_semantic</c> + <c>fc_post1</c>) is dropped at conversion.</para>
///
/// <para><b>Encode</b> (YuE's ICL audio prompt) is built only when the converter kept the encode roots
/// (<c>YueCheckpointConverter.LoadXCodec(..., forEncode: true)</c>) — see <see cref="CanEncode"/>:
/// <code>
///   wav [1, 1, S]  16 kHz mono, raw float, NO normalization
///     ├─ semantic:  zero-pad ±160 -> HuBERT-base, MEAN of all 13 hidden states -> encoder_semantic  [1, 768, T]
///     ├─ acoustic:  dac2.Encoder(64, [8,5,4,2], 256)                                                [1, 256, T]
///     ├─ cat([acoustic, semantic]) -> fc_prior (1024 -> 1024 over channels)                         [1, 1024, T]
///     └─ quantizer.encode                                                                           [n_q, 1, T]
/// </code></para></summary>
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

    // Encode side — built only when the converted weights carry it (the generation-only path skips the
    // ~95M-param HuBERT branch entirely).
    private HubertModel? _semanticModel;
    private XCodecSemanticEncoder? _semanticEncoder;
    private DacEncoder? _acousticEncoder;
    private Tensor? _fcPriorW;   // [latent, latent, 1] — fc_prior reshaped as a 1x1 conv weight
    private Tensor? _fcPriorB;

    /// <summary>Whether the encode path (HuBERT + <c>encoder_semantic</c> + acoustic <c>encoder</c> + <c>fc_prior</c>)
    /// was present in the loaded weights.</summary>
    public bool CanEncode => _fcPriorW is not null;

    public XCodec(XCodecConfig cfg)
    {
        Config = cfg;
        _quantizer = new XCodecEmaResidualVectorQuantizer(cfg.NCodebooks, cfg.CodebookSize, cfg.LatentDim);
        _decoder = new DacDecoder(cfg.ToDacConfig(), "decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _quantizer.LoadWeights(w, "quantizer");
        LoadEncodeWeights(w);

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
    /// <summary>Decodes codes to the raw 1024-dim quantizer latent <c>[B, 1024, T]</c> (EMA-VQ lookup + sum) — the
    /// fused (256 acoustic + 768 semantic) representation the YuE per-stem Vocos vocoders consume in place of the
    /// SeaNet/DAC decode. Caller owns the returned tensor.</summary>
    public Tensor DecodeToLatent(IBackend backend, Tensor codes, int batch, int tFrames)
        => _quantizer.Decode(backend, codes, batch, tFrames);

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

    /// <summary>Builds the encode branch when the weights carry it. Absent <c>fc_prior.weight</c> the codec stays
    /// decode-only and none of the ~95M-param semantic branch is constructed.</summary>
    private void LoadEncodeWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        if (!w.ContainsKey("fc_prior.weight")) return;

        Tensor fc = WhisperOps.EnsureF32(w["fc_prior.weight"]);
        if (fc.Shape[0] != Config.LatentDim || fc.Shape[1] != Config.LatentDim)
            throw new InvalidOperationException(
                $"XCodec fc_prior.weight shape {fc.Shape} != [{Config.LatentDim}, {Config.LatentDim}].");

        _semanticModel = new HubertModel(Config.ToHubertConfig());
        _semanticModel.LoadWeights(w, "semantic_model.");
        _semanticEncoder = new XCodecSemanticEncoder("encoder_semantic", Config.SemanticDim);
        _semanticEncoder.LoadWeights(w);
        _acousticEncoder = new DacEncoder(Config.ToDacConfig(), "encoder");
        _acousticEncoder.LoadWeights(w);

        _fcPriorW = fc.Reshape(new TensorShape(Config.LatentDim, Config.LatentDim, 1));
        _fcPriorB = WhisperOps.EnsureF32(w["fc_prior.bias"]);
    }

    /// <summary>Every encode stage boundary. All tensors are owned by the caller; <see cref="Dispose"/> them via
    /// <see cref="XCodecEncodeStages.Dispose"/>.</summary>
    internal readonly record struct XCodecEncodeStages(Tensor Acoustic, Tensor Semantic, Tensor Prior, int Frames, bool UsedPadFallback)
    {
        public void Dispose() { Acoustic.Dispose(); Semantic.Dispose(); Prior.Dispose(); }
    }

    /// <summary>Runs the encode path up to (and including) <c>fc_prior</c>, exposing each stage boundary.
    /// <paramref name="pcm"/> is 16 kHz mono channels-first <c>[1, 1, S]</c>, raw float — the HuBERT branch takes
    /// the waveform UNNORMALIZED (the checkpoint's <c>preprocessor_config.json</c> claims <c>do_normalize</c>, but
    /// <c>get_regress_target</c> never builds a feature extractor).</summary>
    internal XCodecEncodeStages EncodeStages(IBackend backend, Tensor pcm, int tPcm)
    {
        if (_fcPriorW is null) throw new InvalidOperationException("XCodec encode weights not loaded.");
        if (pcm.Shape[0] != 1) throw new ArgumentException("XCodec encode is single-clip (batch 1).", nameof(pcm));

        int pad = Config.SemanticPad;
        Tensor padded = ZeroPad(pcm, tPcm, pad);

        Tensor feats = _semanticModel!.ForwardMeanHiddenStates(backend, padded, tPcm + 2 * pad);
        int t = (int)feats.Shape[2];
        Tensor semantic = _semanticEncoder!.Forward(backend, feats, 1, t);
        feats.Dispose();

        // The HuBERT stack (valid convs on the 160-padded waveform) and the DAC stack (symmetric padding on the
        // raw waveform) disagree on the frame count for most lengths; upstream re-runs the acoustic encoder on
        // the SAME padded waveform. This is the common case, not an edge case.
        Tensor acoustic = _acousticEncoder!.Forward(backend, pcm, 1, tPcm);
        bool fallback = (int)acoustic.Shape[2] != t;
        if (fallback)
        {
            acoustic.Dispose();
            acoustic = _acousticEncoder.Forward(backend, padded, 1, tPcm + 2 * pad);
        }
        padded.Dispose();
        if ((int)acoustic.Shape[2] != t)
        {
            int ta = (int)acoustic.Shape[2];
            acoustic.Dispose(); semantic.Dispose();
            throw new InvalidOperationException(
                $"XCodec encode branches still disagree after the pad-fallback: acoustic {ta} vs semantic {t} frames.");
        }

        // cat([acoustic, semantic], dim=1) — channels 0..255 acoustic, 256..1023 semantic. Reversing this is
        // silent (the shapes still match) and destroys the codes.
        Tensor fused = new(new TensorShape(1, Config.LatentDim, t), DType.F32);
        long acousticBytes = acoustic.ElementCount * sizeof(float);
        long semanticBytes = semantic.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)acoustic.DataPointer, (void*)fused.DataPointer, acousticBytes, acousticBytes);
        Buffer.MemoryCopy((void*)semantic.DataPointer, (void*)((float*)fused.DataPointer + acoustic.ElementCount),
            semanticBytes, semanticBytes);

        Tensor prior = new(new TensorShape(1, Config.LatentDim, t), DType.F32);
        backend.Conv1d(prior, fused, _fcPriorW!, _fcPriorB,
            stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        fused.Dispose();
        return new XCodecEncodeStages(acoustic, semantic, prior, t, fallback);
    }

    /// <summary>Encodes 16 kHz mono PCM <c>[1, 1, S]</c> to RVQ code indices <c>[nQ, 1, T]</c> I32.
    /// YuE's ICL path uses <c>target_bw = 0.5</c> ⇒ <c>nQ = 1</c> (codebook 0 only).</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int tPcm, int nQ = 1)
    {
        XCodecEncodeStages stages = EncodeStages(backend, pcm, tPcm);
        try
        {
            return _quantizer.Encode(backend, stages.Prior, batch: 1, t: stages.Frames, nQOverride: nQ);
        }
        finally
        {
            stages.Dispose();
        }
    }

    /// <summary>Zero-pads the sample axis of a channels-first <c>[1, 1, S]</c> waveform by <paramref name="pad"/>
    /// on each side (<c>F.pad(x[:,0,:], (160,160))</c>).</summary>
    private static Tensor ZeroPad(Tensor pcm, int tPcm, int pad)
    {
        Tensor padded = new(new TensorShape(1, 1, tPcm + 2 * pad), DType.F32);
        float* dst = (float*)padded.DataPointer;
        for (int i = 0; i < pad; i++) { dst[i] = 0f; dst[tPcm + pad + i] = 0f; }
        long bytes = (long)tPcm * sizeof(float);
        Buffer.MemoryCopy((void*)pcm.DataPointer, (void*)(dst + pad), bytes, bytes);
        return padded;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _quantizer.EnumerateWeights()) yield return t;
        if (_fcPost2W is not null) yield return _fcPost2W;
        if (_fcPost2B is not null) yield return _fcPost2B;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
        if (_semanticModel is not null) foreach (Tensor t in _semanticModel.EnumerateWeights()) yield return t;
        if (_semanticEncoder is not null) foreach (Tensor t in _semanticEncoder.EnumerateWeights()) yield return t;
        if (_acousticEncoder is not null) foreach (Tensor t in _acousticEncoder.EnumerateWeights()) yield return t;
        if (_fcPriorW is not null) yield return _fcPriorW;
        if (_fcPriorB is not null) yield return _fcPriorB;
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

    /// <summary>Acoustic encoder stem filter count (<c>dac2.Encoder</c> <c>d_model</c>).</summary>
    public int EncoderDim { get; init; } = 64;

    /// <summary>Acoustic encoder downsample factors — the SAME encoder-order list as <see cref="DecoderRates"/>,
    /// not its reverse (confirmed by <c>encoder.block.1.block.4.weight_v [128,64,16]</c> ⇒ k = 2·8).</summary>
    public IReadOnlyList<int> EncoderRates { get; init; } = [8, 5, 4, 2];

    /// <summary>HuBERT semantic-branch waveform pad (samples per side) applied before <c>get_regress_target</c>.</summary>
    public int SemanticPad { get; init; } = 160;

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
    /// <summary>The <c>semantic_model.*</c> HuBERT-base config (<c>semantic_ckpts/hf_1_325000/config.json</c>):
    /// identical to the engine default — 768/12/12/3072, bias-free group-normed conv extractor, post-LN encoder,
    /// pos-conv k128 g16.</summary>
    public HubertConfig ToHubertConfig() => new() { SampleRate = SampleRate };

    public DacConfig ToDacConfig() => new()
    {
        SampleRate = SampleRate,
        Channels = Channels,
        EncoderDim = EncoderDim,
        EncoderRates = EncoderRates,
        EncoderLatentDim = AcousticDim,
        DecoderDim = DecoderDim,
        DecoderRates = DecoderRates,
        ResidualDilations = ResidualDilations,
        StemKernelSize = 7,
        DecoderFinalKernelSize = 7,
        TransposeWeightNormDim0 = true,
        DecoderFinalTanh = false,
    };
}
