namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Configuration for the ECAPA-TDNN speaker encoder used by Qwen3-TTS x-vector voice cloning. Mirrors
/// the SpeechBrain ECAPA-TDNN topology (3 SE-Res2Blocks with dilations 2/3/4, channel 512, Res2 scale 8,
/// SE bottleneck 128, attentive statistics pooling → 192-d embedding).</summary>
public sealed record EcapaConfig
{
    /// <summary>Mel-bin count of the input feature sequence.</summary>
    public int InputChannels { get; init; } = 80;

    public int StemChannels { get; init; } = 512;

    /// <summary>Per-SE-Res2Block output channel counts.</summary>
    public IReadOnlyList<int> Channels { get; init; } = [512, 512, 512];
    public IReadOnlyList<int> KernelSizes { get; init; } = [3, 3, 3];
    public IReadOnlyList<int> Dilations { get; init; } = [2, 3, 4];

    /// <summary>Res2Net scale (number of within-channel groups).</summary>
    public int Res2Scale { get; init; } = 8;
    public int SeBottleneck { get; init; } = 128;

    /// <summary>Final speaker-embedding dimension.</summary>
    public int EmbeddingDim { get; init; } = 192;

    /// <summary>Talker speaker-conditioning width the embedding is projected to.</summary>
    public int ConditioningDim { get; init; } = 2_048;

    public static EcapaConfig Default => new();
}
