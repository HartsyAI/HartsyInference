namespace HartsyInference.Engine.Decision.Clef;

/// <summary>Checkpoint contract for Cloudflare Clef typed-decision inference.</summary>
public sealed record ClefConfig
{
    public int BackboneHiddenSize { get; init; } = 5_120;
    public int BackboneLayers { get; init; } = 64;
    public int BackboneVocabularySize { get; init; } = 248_320;
    public int MaxQuestions { get; init; } = 256;
    public bool SupportsImageInput { get; init; } = true;
    public bool SupportsVideoInput { get; init; } = true;
    public static ClefConfig Default => new();
}
