using HartsyInference.Audio.Models.Vits;

namespace HartsyInference.Audio.Models.MeloTts;

/// <summary>Configuration for MeloTTS (MyShell). A VITS model whose text encoder additionally consumes tone
/// + language ids and two BERT feature streams, with multispeaker conditioning. The flow / duration / HiFi-GAN
/// stages are the shared <see cref="VitsConfig">VITS</see> core. See <c>docs/Research/VITS_ARCHITECTURE.md</c>
/// (MeloTTS deltas).</summary>
public sealed record MeloTtsConfig
{
    /// <summary>The shared VITS core config (hidden 192, gin 256 for multispeaker).</summary>
    public VitsConfig Core { get; init; } = new() { GinChannels = 256, NumSpeakers = 1, SampleRate = 44_100, NumVocab = 256 };

    public int NumTones { get; init; } = 16;
    public int NumLanguages { get; init; } = 10;
    public int BertDim { get; init; } = 1_024;       // Chinese-RoBERTa / XLM-R
    public int JaBertDim { get; init; } = 768;       // Japanese BERT
    public int NumSpeakers { get; init; } = 1;

    /// <summary>Blend ratio between the stochastic and deterministic duration predictors.</summary>
    public float SdpRatio { get; init; } = 0.2f;

    public float NoiseScale { get; init; } = 0.6f;
    public float LengthScale { get; init; } = 1.0f;
    public float NoiseScaleW { get; init; } = 0.8f;

    public static MeloTtsConfig EnglishV3 => new();
}
