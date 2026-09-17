namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>One melody note placed on the clip's timeline.</summary>
/// <param name="Pitch">MIDI pitch, 0-127.</param>
/// <param name="Track">Which of the two melody lines it belongs to — 0 is the vocal, 1 the instrumental.</param>
/// <param name="EndTime">Seconds into the clip where the note stops; it starts at its event's own time.</param>
public readonly record struct TimedScoreNote(int Pitch, int Track, double EndTime);
