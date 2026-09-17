namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>A decoded note placed on the clip's own timeline.</summary>
/// <param name="Note">The note as the window decoded it.</param>
/// <param name="EndSeconds">Where it stops, in seconds from the start of the clip.</param>
public readonly record struct StitchedNote(ScoreNote Note, double EndSeconds);

/// <summary>One decoded event placed on the clip's timeline, after the window that read it was trusted for it.
///
/// <para>A window counts subbeats and seconds from its own start, so the same passage read by two windows comes
/// back with two different positions. Stitching adds the two coordinates that are the whole piece's: seconds
/// from the start of the clip, and a subbeat count continuous across window boundaries. The decoded event is
/// kept as it was read — borrowed, never rewritten — because the next window has to replay its raw tokens as
/// context and a re-encoding would not reproduce them.</para></summary>
public sealed class StitchedScoreEvent
{
    /// <summary>The event as its window decoded it: positions relative to that window, with its raw tokens.</summary>
    public required ScoreEvent Source { get; init; }

    /// <summary>Where the event falls in the whole clip, in seconds.</summary>
    public required double Seconds { get; init; }

    /// <summary>Where the event falls in the whole piece, in subbeats, continuous across window boundaries.</summary>
    public required int GlobalSubbeat { get; init; }

    /// <summary>Notes starting here with their end times on the clip's timeline; null when the event states none.</summary>
    public IReadOnlyList<StitchedNote>? Melody { get; init; }

    /// <summary>Position in subbeats within the window that read this event.</summary>
    public int Subbeat => Source.Subbeat;

    /// <summary>The event's timestamp on the clip's timeline, when it states one. The released implementation
    /// overwrites the window-relative value in place; here <see cref="Source"/> keeps it.</summary>
    public double? Timestamp => Source.Timestamp is null ? null : Seconds;
}
