namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>One note in a melody field.</summary>
/// <param name="Pitch">MIDI pitch, 0-127.</param>
/// <param name="Track">Which of the two melody lines it belongs to — 0 is the vocal, 1 the instrumental.</param>
/// <param name="DurationBin">Index into <see cref="ScoreTokenizer.DurationTemplates"/>.</param>
/// <param name="DurationSteps">That template's length, in subbeats.</param>
public readonly record struct ScoreNote(int Pitch, int Track, int DurationBin, int DurationSteps);

/// <summary>Where in the bar an event sits, and under what time signature.</summary>
/// <param name="Meter">Time signature, when the event states one.</param>
/// <param name="EighthPosition">Position within the bar, when the event states one.</param>
public readonly record struct ScoreRhythm((int Numerator, int Denominator)? Meter, int? EighthPosition);

/// <summary>One decoded event: a position in the piece plus whichever of the six fields it carries.
///
/// <para>Both the raw tokens and their read values are kept. The values are what a serializer wants; the tokens
/// are what a later window has to replay verbatim as context, and re-encoding from values would not reproduce
/// them.</para></summary>
public sealed class ScoreEvent
{
    /// <summary>Position in the piece, counted in subbeats from the start of this window.</summary>
    public required int Subbeat { get; set; }

    /// <summary>Raw tokens, grouped by the field they belong to.</summary>
    public required Dictionary<string, List<int>> TokensByField { get; init; }

    /// <summary>Seconds into the window, when the event carries a timestamp.</summary>
    public double? Timestamp { get; init; }

    /// <summary>Meter and position in the bar, when the event carries them.</summary>
    public ScoreRhythm? Rhythm { get; init; }

    /// <summary>Section label, when the event names one.</summary>
    public string? Structure { get; init; }

    /// <summary>Key as <c>root:major|minor</c>, when the event names one.</summary>
    public string? Key { get; init; }

    /// <summary>Chord label, when the event names one.</summary>
    public string? Chord { get; init; }

    /// <summary>Notes starting at this position, when the event carries any.</summary>
    public IReadOnlyList<ScoreNote>? Melody { get; init; }

    /// <summary>Position in the whole piece, once windows have been stitched together.</summary>
    public int GlobalSubbeat { get; set; }

    /// <summary>Seconds into the whole piece, once windows have been stitched together.</summary>
    public double Time { get; set; }
}
