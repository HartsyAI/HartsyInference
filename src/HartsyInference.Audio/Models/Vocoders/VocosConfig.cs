namespace HartsyInference.Audio.Models.Vocoders;

/// <summary>Configuration for the Vocos vocoder (Siuzdak 2023). Vocos converts an
/// 80-or-100-bin log-mel spectrogram into a waveform via a ConvNeXt-V2-style backbone
/// followed by an iSTFT head — no transposed convolutions, no learned upsampling. It is
/// ~13× faster than HiFi-GAN at comparable quality and is the vocoder of choice for
/// F5-TTS, ChatTTS, and several modern TTS models.
///
/// <para>Values default to the standard <c>charactr/vocos-mel-24khz</c> release. The
/// preset matches the upstream <c>config.yaml</c> exactly.</para></summary>
public sealed record VocosConfig
{
    /// <summary>Number of mel-bin input channels.</summary>
    public int InputChannels { get; init; } = 100;

    /// <summary>Backbone hidden dimension.</summary>
    public int HiddenDim { get; init; } = 512;

    /// <summary>ConvNeXt MLP intermediate dim (typically 3× HiddenDim).</summary>
    public int IntermediateDim { get; init; } = 1_536;

    /// <summary>Number of stacked ConvNeXt blocks.</summary>
    public int NumLayers { get; init; } = 8;

    /// <summary>Depthwise conv kernel size in each ConvNeXt block.</summary>
    public int DwConvKernel { get; init; } = 7;

    /// <summary>LayerNorm epsilon (Vocos uses 1e-6, NOT the standard 1e-5).</summary>
    public float LayerNormEps { get; init; } = 1e-6f;

    /// <summary>Initial value for the per-block LayerScale (gamma). Null means the upstream
    /// default of 1/NumLayers.</summary>
    public float? LayerScaleInitValue { get; init; } = null;

    /// <summary>Number of embeddings for the AdaLayerNorm conditioning. Null for the mel backbone
    /// (plain LayerNorm), 4 for the encodec backbone (conditioned on bandwidth id).</summary>
    public int? AdaNormNumEmbeddings { get; init; } = null;

    /// <summary>STFT/iSTFT padding mode: "center" for the mel release, "same" for the encodec release.</summary>
    public string Padding { get; init; } = "center";

    // ── iSTFT head ─────────────────────────────────────────────────────────────

    /// <summary>FFT size used by the synthesis iSTFT.</summary>
    public int NFft { get; init; } = 1_024;

    /// <summary>Hop length used by the synthesis iSTFT. Audio sample rate / hop = mel frame rate.
    /// For the standard release this gives 24000 / 256 = ~93.75 Hz mel frame rate.</summary>
    public int HopLength { get; init; } = 256;

    /// <summary>Output audio sample rate.</summary>
    public int SampleRate { get; init; } = 24_000;

    /// <summary>Linear gain applied to the iSTFT output; 1.0 = disabled (the correct value).
    /// A previous 44.53 "calibration" was compensating for a broken reference resampler that
    /// leaked ~12% high-freq imaging into the input mel; with the resampler fixed, feeding the
    /// reference F5 mel through this vocoder reproduces the upstream output RMS exactly
    /// (0.1527 vs 0.1524). Override per-checkpoint only if a future fork is genuinely mis-scaled.</summary>
    public float OutputGain { get; init; } = 1.0f;

    /// <summary>Standard preset: <c>charactr/vocos-mel-24khz</c>. 24 kHz mono, 100 mel input,
    /// 512 hidden, 8 ConvNeXt blocks, n_fft=1024, hop=256.</summary>
    public static VocosConfig Mel24k => new();
}
