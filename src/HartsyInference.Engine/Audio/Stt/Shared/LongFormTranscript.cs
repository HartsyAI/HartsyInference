namespace HartsyInference.Engine.Audio;

/// <summary>The joined result of a long clip: text, timed segments when the model gave them, and how the clip was split.</summary>
internal sealed record LongFormTranscript(string Text, IReadOnlyList<SttSegment>? Segments, int WindowCount, int SilentWindowCount, int HardCutCount);
