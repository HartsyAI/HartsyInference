using HartsyInference.Audio.Dsp.Loudness;
using HartsyInference.Audio.Models.Separation;

namespace HartsyInference.Audio.Models.LongCat;

/// <summary>The driving speech as the reference's demo hands it to the audio encoder (<c>run_demo_avatar_single_audio_to_video.py</c> and
/// <c>get_audio_embedding_whisper</c> in LongCat-Video @ 58eb336): cut to the clip plus one second, padded with silence up to that length, and,
/// on the 1.5 build, normalised to −23 LUFS. The encoder attends over the whole window, so audio past the cut would change every frame.</summary>
public static partial class LongCatDrivingAudio
{
    /// <summary>The rate of the speech in and out.</summary>
    public const int SampleRate = 16_000;

    /// <summary>The loudness the 1.5 build's speech is moved to.</summary>
    public const double TargetLufs = -23.0;

    private const double MarginSeconds = 1.0;

    // The reference leaves audio whose loudness is past this magnitude alone (silence measures as negative infinity).
    private const double LoudnessLimit = 100.0;

    /// <summary>A new array holding the encoder's input for a clip of <paramref name="clipFrames"/> frames at <paramref name="fps"/>.
    /// <paramref name="normalizeLoudness"/> is on for the 1.5 build and off for 1.6.</summary>
    public static float[] Prepare(ReadOnlySpan<float> speech16k, int clipFrames, int fps, bool normalizeLoudness)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(clipFrames);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fps);
        double seconds = clipFrames / (double)fps + MarginSeconds;
        int kept = Math.Min(CutSamples(seconds), speech16k.Length);
        int padding = Math.Max(0, (int)Math.Ceiling((seconds - kept / (double)SampleRate) * SampleRate));
        float[] prepared = new float[kept + padding];
        speech16k[..kept].CopyTo(prepared);
        if (normalizeLoudness)
            NormalizeLoudness(prepared);
        return prepared;
    }

    /// <summary>The samples at 16 kHz that <paramref name="seconds"/> of speech are cut to. The cut is made as the separator makes it, on its
    /// 44.1 kHz input, then expressed at 16 kHz.</summary>
    private static int CutSamples(double seconds)
    {
        int separatorRate = MdxSettings.KimVocal2.SampleRate;
        double sourceSamples = Math.Ceiling(seconds * separatorRate);
        return (int)Math.Ceiling(sourceSamples * SampleRate / separatorRate);
    }

    private static void NormalizeLoudness(float[] speech)
    {
        double loudness = Bs1770Loudness.Integrated(speech, SampleRate);
        if (Math.Abs(loudness) <= LoudnessLimit)
            Bs1770Loudness.ApplyGain(speech, loudness, TargetLufs);
    }
}
