namespace HartsyInference.Audio.Models.Whisper;

/// <summary>A word as it appears in a transcript and how many consecutive BPE tokens spell it.</summary>
internal readonly record struct WhisperWordTokens(string Text, int TokenCount);
