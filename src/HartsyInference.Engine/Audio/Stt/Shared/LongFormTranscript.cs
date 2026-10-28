namespace HartsyInference.Engine.Audio;

/// <summary>The joined result of a long clip: text, timed segments when the model gave them, and how the clip was split.
/// <paramref name="EmptyWindowCount"/> counts non-silent windows that decoded to no text; their audio is missing from <paramref name="Text"/>.</summary>
internal sealed record LongFormTranscript(string Text, IReadOnlyList<SttSegment>? Segments, int WindowCount, int SilentWindowCount, int HardCutCount, int EmptyWindowCount);
