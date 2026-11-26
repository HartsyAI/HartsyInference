using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>A separated driving track in the two forms the video pre-step hands on.</summary>
/// <param name="Stem">The 44.1 kHz stereo vocals as a 16-bit WAV clip.</param>
/// <param name="Speech16k">The same vocals as 16 kHz mono, taken from the float stem and not cropped to a clip length.</param>
internal readonly record struct SeparatedVocals(AudioClip Stem, float[] Speech16k)
{
    /// <summary>Rate of <see cref="Speech16k"/>, the one the audio-conditioned recipes encode.</summary>
    internal const int SpeechSampleRate = 16_000;
}
