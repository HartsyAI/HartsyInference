namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>The on-disk JSON side of a profile. The embeddings themselves live in the companion binary named by <see cref="EmbeddingFile"/>: 192 floats per utterance is not something a human edits, and keeping them out of the JSON keeps the sidecar readable and hand-fixable. <see cref="UtteranceCount"/> and <see cref="Dimension"/> are duplicated here purely so a truncated or mismatched binary is caught at load instead of producing a wrong centroid.</summary>
internal sealed record SpeakerProfileDocument
{
    public string Name { get; init; } = string.Empty;

    public string? Phrase { get; init; }

    public DateTimeOffset EnrolledUtc { get; init; }

    public int UtteranceCount { get; init; }

    public int Dimension { get; init; }

    public string EmbeddingFile { get; init; } = string.Empty;
}
