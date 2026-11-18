namespace HartsyInference.Audio.Models.Whisper;

/// <summary>A word as it appears in a transcript and how many consecutive BPE tokens spell it, including the
/// <paramref name="LeadTokens"/> of attached opening punctuation and the <paramref name="TrailTokens"/> of closing
/// punctuation, which do not take part in the word's timing.</summary>
internal readonly record struct WhisperWordTokens(string Text, int TokenCount, int LeadTokens = 0, int TrailTokens = 0);
