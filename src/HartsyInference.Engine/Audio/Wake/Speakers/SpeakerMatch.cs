namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>The result of scoring one utterance against the enrolled household.
///
/// <para><see cref="Name"/> is the nearest profile whether or not it was accepted, and <see cref="Score"/> is its raw
/// cosine similarity — both are populated on a rejection too, because the pairs of (name, score) a deployment rejects
/// are exactly the data needed to calibrate the threshold. Only <see cref="Outcome"/> says whether the identity may
/// be acted on.</para></summary>
public readonly record struct SpeakerMatch(string? Name, float Score, SpeakerMatchOutcome Outcome, float Threshold)
{
    /// <summary>True only when the nearest centroid cleared the threshold; false for guests and unscorable audio.</summary>
    public bool IsIdentified => Outcome == SpeakerMatchOutcome.Identified;

    /// <summary>The accepted identity, or null for a guest or an unscorable clip.</summary>
    public string? IdentifiedName => IsIdentified ? Name : null;

    /// <summary>Whether <paramref name="required"/> is satisfied — a null or empty requirement accepts anyone,
    /// including a guest, which is what an unrestricted wake word wants.</summary>
    public bool Satisfies(string? required) =>
        string.IsNullOrWhiteSpace(required)
        || (IsIdentified && string.Equals(Name, required, StringComparison.OrdinalIgnoreCase));

    /// <summary>Log-shaped one-liner, e.g. <c>kaleb (0.62 >= 0.35)</c> or <c>guest, nearest kaleb 0.19 &lt; 0.35</c>.</summary>
    public override string ToString() => Outcome switch
    {
        SpeakerMatchOutcome.Identified => $"{Name} ({Score:0.00} >= {Threshold:0.00})",
        SpeakerMatchOutcome.Unknown => $"guest, nearest {Name} {Score:0.00} < {Threshold:0.00}",
        SpeakerMatchOutcome.NoProfiles => "guest (nobody enrolled)",
        _ => "guest (clip too short to score)",
    };
}
