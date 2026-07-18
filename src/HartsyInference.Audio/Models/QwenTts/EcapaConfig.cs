namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Configuration for the ECAPA-TDNN speaker encoder used by Qwen3-TTS x-vector voice cloning
/// (<c>speaker_encoder.*</c> in the Base checkpoint). Matches the reference <c>Qwen3TTSSpeakerEncoder</c>:
/// a TDNN stem + three SE-Res2Net blocks (dilations 2/3/4) → multi-layer aggregation (concat of the three
/// SE-Res2 outputs) → attentive statistics pooling → a final 1×1 conv to the embedding. <b>No BatchNorm</b>
/// (TDNN blocks are Conv1d+ReLU only) and <b>no post-fc projection</b> — the fc output IS the x-vector.</summary>
public sealed record EcapaConfig
{
    /// <summary>Mel-bin count of the input feature sequence.</summary>
    public int InputChannels { get; init; } = 128;

    public int StemChannels { get; init; } = 512;

    /// <summary>Kernel size of the TDNN stem (Conv1d) layer.</summary>
    public int StemKernelSize { get; init; } = 5;

    /// <summary>Per-SE-Res2Block output channel counts.</summary>
    public IReadOnlyList<int> Channels { get; init; } = [512, 512, 512];
    public IReadOnlyList<int> KernelSizes { get; init; } = [3, 3, 3];
    public IReadOnlyList<int> Dilations { get; init; } = [2, 3, 4];

    /// <summary>Multi-layer-aggregation channels (concat of the 3 SE-Res2 outputs = 3×512 = 1536).</summary>
    public int MfaChannels { get; init; } = 1_536;

    /// <summary>Attentive-statistics-pooling attention bottleneck channels.</summary>
    public int AttentionChannels { get; init; } = 128;

    /// <summary>Res2Net scale (number of within-channel chunks).</summary>
    public int Res2Scale { get; init; } = 8;
    public int SeBottleneck { get; init; } = 128;

    /// <summary>Final speaker-embedding dimension (fc output = enc_dim). This IS the injected x-vector width and
    /// must equal the talker hidden size (2048 for 1.7B).</summary>
    public int EmbeddingDim { get; init; } = 2_048;

    /// <summary>Input waveform sample rate (Hz) for mel-feature extraction.</summary>
    public int SampleRate { get; init; } = 24_000;

    public static EcapaConfig Default => new();
}
