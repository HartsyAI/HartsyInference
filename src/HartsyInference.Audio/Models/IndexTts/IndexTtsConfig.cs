using HartsyInference.Audio.Models.LanguageModels.Gpt;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>IndexTTS-1.5 hyperparameters, hand-ported from the real <c>config.yaml</c> (no YAML dependency in this
/// solution — every other model with an upstream <c>config.yaml</c> hand-ports its values into a static preset the
/// same way, e.g. <c>F5TtsConfig.V1Base</c>).</summary>
public sealed record IndexTtsConfig
{
    public required GptConfig Gpt { get; init; }
    public required IndexTtsConformerConfig ConditioningEncoder { get; init; }
    public required IndexTtsBigVganConfig BigVgan { get; init; }
    public required int MaxTextTokens { get; init; }
    public required int MaxMelTokens { get; init; }
    public required int SampleRate { get; init; }

    public static IndexTtsConfig V1_5 => new()
    {
        Gpt = GptConfig.IndexTts15,
        ConditioningEncoder = IndexTtsConformerConfig.V1_5,
        BigVgan = IndexTtsBigVganConfig.V1_5,
        MaxTextTokens = 600,
        MaxMelTokens = 800,
        SampleRate = 24_000,
    };
}
