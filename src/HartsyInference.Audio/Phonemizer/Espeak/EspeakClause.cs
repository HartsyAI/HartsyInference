namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>One clause of text and the punctuation espeak-ng ends it with (<c>'\0'</c> when it ends with none).</summary>
internal readonly record struct EspeakClause(string Text, char Mark);
