namespace HartsyInference.Vision.Detection;

/// <summary>Result of tokenizing an open-vocabulary caption: the flat token id sequence fed to the text encoder,
/// plus the per-phrase token spans used to map per-token detection scores back to phrases.</summary>
public sealed record GroundingCaptionTokens
{
    /// <summary>Flat token ids (including any special / separator tokens) for the text encoder.</summary>
    public required int[] TokenIds { get; init; }

    /// <summary>Phrase spans into <see cref="TokenIds"/>, in caption order.</summary>
    public required IReadOnlyList<GroundingPhraseSpan> Phrases { get; init; }
}
