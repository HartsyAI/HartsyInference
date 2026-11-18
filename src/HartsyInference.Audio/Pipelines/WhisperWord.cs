namespace HartsyInference.Audio.Pipelines;

/// <summary>One word of a Whisper transcript with the span of audio it was spoken over, aligned from the decoder's
/// cross-attention. <paramref name="Text"/> keeps the leading space Whisper's tokenizer gives a word that follows
/// another, so concatenating the words rebuilds the transcript for any script, and attached punctuation, which the
/// span leaves out. <paramref name="Start"/> and <paramref name="End"/> are seconds from the start of the audio on the
/// encoder's 0.02 s grid; a pause is attributed to the word before it, so ends are looser than starts.</summary>
public readonly record struct WhisperWord(string Text, double Start, double End);
