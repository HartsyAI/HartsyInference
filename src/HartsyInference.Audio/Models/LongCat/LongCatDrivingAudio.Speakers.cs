using HartsyInference.Audio.Dsp.Loudness;

namespace HartsyInference.Audio.Models.LongCat;

public static partial class LongCatDrivingAudio
{
    /// <summary>The two speakers' audio as the reference's multi-speaker demo hands it to the encoder
    /// (<c>run_demo_avatar_multi_audio_to_video.py</c> in LongCat-Video @ 58eb336, <c>audio_prepare_multi</c> as upstream pull request 162 has it):
    /// each speaker's speech is cut to the clip plus one second, the two are laid out in time, padded with silence up to that length and, on the
    /// 1.5 build, normalised to −23 LUFS each. A sequential pair is not cut again after joining.</summary>
    /// <param name="first">The first speaker; null for one who does not speak.</param>
    /// <param name="second">The second speaker; null for one who does not speak. At least one speaker is given.</param>
    /// <param name="timing">Whether the speakers talk at once or one after the other.</param>
    /// <param name="clipFrames">The clip's frames.</param>
    /// <param name="fps">The frame rate.</param>
    /// <param name="normalizeLoudness">On for the 1.5 build and off for 1.6.</param>
    /// <param name="silentTrack">Adds a third, silent track: the reference does when other people are marked in the picture.</param>
    public static LongCatSpeakerAudio PrepareSpeakers(LongCatSpeakerSpeech? first, LongCatSpeakerSpeech? second, LongCatSpeakerTiming timing,
        int clipFrames, int fps, bool normalizeLoudness, bool silentTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(clipFrames);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fps);
        double seconds = clipFrames / (double)fps + MarginSeconds;
        LongCatSpeakerAudio audio = Combine(Cut(first, seconds), Cut(second, seconds), timing, seconds);
        if (normalizeLoudness)
            foreach (float[] track in audio.Tracks)
                NormalizeLoudness(track);
        return silentTrack
            ? audio with { Tracks = [.. audio.Tracks, new float[audio.Tracks[0].Length]] }
            : audio;
    }

    /// <summary>The reference's <c>audio_prepare_multi</c> over speech already cut to the clip: the two tracks and the soundtrack laid out by
    /// <paramref name="timing"/> and padded with silence to <paramref name="seconds"/>. A missing speaker is silence of the other one's
    /// lengths.</summary>
    internal static LongCatSpeakerAudio Combine(LongCatSpeakerSpeech? first, LongCatSpeakerSpeech? second, LongCatSpeakerTiming timing,
        double seconds)
    {
        if (first is null && second is null)
            throw new ArgumentException("At least one speaker needs speech.", nameof(first));
        LongCatSpeakerSpeech one = first ?? Silent(second!.Value);
        LongCatSpeakerSpeech two = second ?? Silent(one);
        float[] firstTrack, secondTrack, soundtrack;
        switch (timing)
        {
            case LongCatSpeakerTiming.Sequential:
                {
                    // One boundary for the conditioning and for what is heard.
                    int oneLength = Math.Min(one.Speech.Length, one.Recording.Length);
                    int twoLength = Math.Min(two.Speech.Length, two.Recording.Length);
                    firstTrack = new float[oneLength + twoLength];
                    secondTrack = new float[oneLength + twoLength];
                    soundtrack = new float[oneLength + twoLength];
                    one.Speech.AsSpan(0, oneLength).CopyTo(firstTrack);
                    two.Speech.AsSpan(0, twoLength).CopyTo(secondTrack.AsSpan(oneLength));
                    one.Recording.AsSpan(0, oneLength).CopyTo(soundtrack);
                    two.Recording.AsSpan(0, twoLength).CopyTo(soundtrack.AsSpan(oneLength));
                    break;
                }
            case LongCatSpeakerTiming.Parallel:
                {
                    int length = Math.Min(Math.Min(one.Speech.Length, one.Recording.Length), Math.Min(two.Speech.Length, two.Recording.Length));
                    if (length == 0)
                        throw new ArgumentException("Parallel speakers need speech from both; one clip is empty.", nameof(first));
                    firstTrack = one.Speech[..length];
                    secondTrack = two.Speech[..length];
                    soundtrack = new float[length];
                    for (int i = 0; i < length; i++)
                        soundtrack[i] = one.Recording[i] + two.Recording[i];
                    break;
                }
            default:
                throw new ArgumentOutOfRangeException(nameof(timing), timing, "Unknown speaker timing.");
        }
        int total = (int)Math.Ceiling(seconds * SampleRate);
        if (total > firstTrack.Length)
        {
            Array.Resize(ref firstTrack, total);
            Array.Resize(ref secondTrack, total);
            Array.Resize(ref soundtrack, total);
        }
        return new LongCatSpeakerAudio([firstTrack, secondTrack], soundtrack);
    }

    private static LongCatSpeakerSpeech? Cut(LongCatSpeakerSpeech? speaker, double seconds)
    {
        if (speaker is not { } given)
            return null;
        ArgumentNullException.ThrowIfNull(given.Speech);
        ArgumentNullException.ThrowIfNull(given.Recording);
        int cut = CutSamples(seconds);
        return given.Speech.Length > cut ? given with { Speech = given.Speech[..cut] } : given;
    }

    private static LongCatSpeakerSpeech Silent(LongCatSpeakerSpeech other) =>
        new(new float[other.Speech.Length], new float[other.Recording.Length]);
}
