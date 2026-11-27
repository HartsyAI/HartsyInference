namespace HartsyInference.Audio.Models.LongCat;

/// <summary>One speaker's 16 kHz mono audio: what the speech encoder hears (the separated vocals, or the recording itself) and the recording the
/// soundtrack is mixed from.</summary>
public readonly record struct LongCatSpeakerSpeech(float[] Speech, float[] Recording);
