using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Rejects a voice reference too long for CosyVoice 2's LM prompt and warns about a long one.</summary>
public static class CosyVoiceReferenceGuard
{
    public const double WarnSeconds = 15.0;
    public const double MaxSeconds = 20.0;

    /// <summary>Throws above <see cref="MaxSeconds"/>, warns above <see cref="WarnSeconds"/>; a missing reference passes.</summary>
    public static void Check(int sampleCount, int sampleRate)
    {
        if (sampleCount <= 0 || sampleRate <= 0)
        {
            return;
        }
        CheckSeconds(sampleCount / (double)sampleRate);
    }

    /// <summary>The same check on a reference length already known in seconds, for callers that read it from a header.</summary>
    public static void CheckSeconds(double seconds)
    {
        if (seconds > MaxSeconds)
        {
            throw new InvalidOperationException(
                $"CosyVoice 2 can't clone from a {seconds:0.#} s reference (the limit is {MaxSeconds:0} s). "
                + "The model is prompted with the whole clip, and past about that length it drops words from the end "
                + "of the text or turns the speech into garble; the official implementation does the same. "
                + "Use a clean 5 to 15 s clip of one speaker and give its exact transcript.");
        }
        if (seconds > WarnSeconds)
        {
            Logs.Warning($"CosyVoice: a {seconds:0.#} s reference is long; the model gets less reliable above about "
                + $"{WarnSeconds:0} s. A clean 5 to 15 s clip with its exact transcript clones more accurately.");
        }
    }
}
