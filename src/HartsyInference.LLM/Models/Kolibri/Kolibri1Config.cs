using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Models.Kolibri;

/// <summary>Checkpoint contract for Aleph Alpha Kolibri-1.</summary>
public sealed record Kolibri1Config
{
    public int HiddenSize { get; init; } = 2_560;
    public int NumLayers { get; init; } = 50;
    public int NumHeads { get; init; } = 48;
    public int NumKeyValueHeads { get; init; } = 4;
    public int HeadDim { get; init; } = 128;
    public int VocabSize { get; init; } = 128_000;
    public int ExpertCount { get; init; } = 384;
    public int RoutedExpertsPerToken { get; init; } = 6;
    public int SharedExpertCount { get; init; } = 1;
    public int SlidingWindow { get; init; } = 512;
    public int FullAttentionEvery { get; init; } = 5;
    public MoeConfig Moe { get; init; } = new()
    {
        NumExperts = 384,
        NumExpertsPerTok = 6,
        MoeIntermediateSize = 512,
        SharedExpertIntermediateSize = 512,
        NormTopKProb = false,
        Scoring = MoeScoring.Sigmoid,
    };
    public int NativeContextLength { get; init; } = 262_144;
    public bool UsesFp8BlockWeights { get; init; } = true;
    public static Kolibri1Config Default => new();
}
