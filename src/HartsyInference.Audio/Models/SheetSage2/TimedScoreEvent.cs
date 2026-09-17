namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>A decoded event placed on the clip's timeline, which is what the ABC serializer reads.
///
/// <para>The decoder writes positions in subbeats relative to its own window (<see cref="ScoreEvent"/>); the
/// stitching pass resolves those to seconds from the start of the clip and gives every note an end time. This is
/// that resolved form — the raw tokens are gone, because nothing downstream of stitching replays them.</para>
///
/// <para>A field that the event does not carry is null, and a null field means "unchanged" rather than "none":
/// key, chord and structure each run until the next event states a new one.</para></summary>
public sealed class TimedScoreEvent
{
    /// <summary>Seconds into the clip.</summary>
    public required double Time { get; init; }

    /// <summary>Meter and position in the bar, when the event carries them.</summary>
    public ScoreRhythm? Rhythm { get; init; }

    /// <summary>Section label, when the event names one.</summary>
    public string? Structure { get; init; }

    /// <summary>Key as <c>root:major|minor</c>, when the event names one.</summary>
    public string? Key { get; init; }

    /// <summary>Chord label, when the event names one.</summary>
    public string? Chord { get; init; }

    /// <summary>Notes starting at this event's time, when it carries any.</summary>
    public IReadOnlyList<TimedScoreNote>? Melody { get; init; }
}
