namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>Why a speaker lookup ended the way it did. A caller gating on a required speaker must distinguish
/// "this is somebody else" from "there was nothing to decide with" — the first is a rejection, the second is a
/// configuration or capture problem that should be logged rather than silently treated as an intruder.</summary>
public enum SpeakerMatchOutcome
{
    /// <summary>The nearest centroid cleared the threshold.</summary>
    Identified,

    /// <summary>Somebody was heard, but no enrolled centroid came close enough — a guest.</summary>
    Unknown,

    /// <summary>Nothing is enrolled yet, so every speaker is a guest by construction.</summary>
    NoProfiles,

    /// <summary>The clip was too short for a stable embedding and was not scored at all.</summary>
    AudioTooShort,
}
