namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>One enrolled household speaker: the raw enrollment embeddings kept for re-derivation, and the centroid
/// scored against at recognition time. Arrays are shared, not copied — treat them as immutable.</summary>
public sealed record SpeakerProfile
{
    /// <summary>Display name, matched case-insensitively; this is the identity <c>WakeWordConfig.RequiredSpeaker</c> names.</summary>
    public required string Name { get; init; }

    /// <summary>Unit-length speaker model, the L2-normalized mean of <see cref="EnrollmentEmbeddings"/>.</summary>
    public required float[] Centroid { get; init; }

    /// <summary>The individual enrollment embeddings, each L2-normalized. Persisted so the centroid can be re-derived
    /// (or adapted) later without asking the household to record again.</summary>
    public required float[][] EnrollmentEmbeddings { get; init; }

    /// <summary>The phrase repeated during enrollment when it was text-dependent, else null. Recorded because a
    /// text-dependent profile is only trustworthy when scored on that same phrase.</summary>
    public string? Phrase { get; init; }

    /// <summary>When the profile was last (re-)enrolled.</summary>
    public DateTimeOffset EnrolledUtc { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Embedding width — 192 for CAM++; a profile of a different width cannot be scored against it.</summary>
    public int Dimension => Centroid.Length;

    /// <summary>How many utterances went into the centroid; below <see cref="SpeakerProfileStore.RecommendedEnrollmentUtterances"/>
    /// the centroid is dominated by the acoustics of a single recording.</summary>
    public int UtteranceCount => EnrollmentEmbeddings.Length;

    /// <summary>Whether enrollment was text-dependent, which is what makes ~1 s verification usable.</summary>
    public bool IsTextDependent => !string.IsNullOrWhiteSpace(Phrase);
}
