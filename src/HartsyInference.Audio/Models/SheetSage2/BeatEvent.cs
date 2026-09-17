namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>One beat of the decoded grid.</summary>
/// <param name="Time">Seconds into the clip.</param>
/// <param name="BeatId">Position in the bar, counted from 1.</param>
/// <param name="DeclaredNumerator">Beats per bar the event claimed, which need not match the bar's real span.</param>
/// <param name="Denominator">Which note value gets the beat.</param>
public readonly record struct BeatEvent(double Time, int BeatId, int DeclaredNumerator, int Denominator);
