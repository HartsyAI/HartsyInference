using HartsyInference.Audio.Dsp.Loudness;
using HartsyInference.Audio.Models.LongCat;
using Xunit;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>The two speakers' audio layout, and the cut, loudness and silent track around it. The comparison with the reference is
/// <c>LongCatSpeakerAudioParityTests</c>.</summary>
[Trait("Category", "Unit")]
public sealed class LongCatSpeakerAudioTests
{
    private const int Rate = LongCatDrivingAudio.SampleRate;

    private static float[] Tone(int samples, double amplitude, double hertz)
    {
        float[] signal = new float[samples];
        for (int n = 0; n < samples; n++)
            signal[n] = (float)(amplitude * Math.Sin(2 * Math.PI * hertz * n / Rate));
        return signal;
    }

    private static LongCatSpeakerSpeech Speech(float[] signal) => new(signal, signal);

    [Fact]
    public void ParallelSpeechEndsWithTheShorterSpeakerAndPadsToTheClipPlusASecond()
    {
        float[] longer = Tone(3 * Rate, 0.2, 220), shorter = Tone(Rate / 2, 0.2, 330);

        LongCatSpeakerAudio audio = LongCatDrivingAudio.PrepareSpeakers(Speech(longer), Speech(shorter), LongCatSpeakerTiming.Parallel, 25, 25,
            normalizeLoudness: false, silentTrack: false);

        Assert.Equal(2, audio.Tracks.Length);
        Assert.All(audio.Tracks, track => Assert.Equal(2 * Rate, track.Length));
        Assert.Equal(longer.Take(Rate / 2), audio.Tracks[0].Take(Rate / 2));
        Assert.All(audio.Tracks[0].Skip(Rate / 2), sample => Assert.Equal(0f, sample));
        Assert.Equal(longer[100] + shorter[100], audio.Soundtrack[100]);
        Assert.Equal(2 * Rate, audio.Soundtrack.Length);
    }

    [Fact]
    public void SequentialSpeechCutsEachSpeakerToTheClipAndIsNotCutAfterJoining()
    {
        // A one-second clip: each speaker is cut to two seconds, so the pair runs to four, past the clip plus one second.
        float[] first = Tone(3 * Rate, 0.2, 220), second = Tone(3 * Rate, 0.2, 330);

        LongCatSpeakerAudio audio = LongCatDrivingAudio.PrepareSpeakers(Speech(first), Speech(second), LongCatSpeakerTiming.Sequential, 25, 25,
            normalizeLoudness: false, silentTrack: false);

        int cut = audio.Tracks[0].Length / 2;
        Assert.InRange(cut, 2 * Rate, 2 * Rate + 2);
        Assert.Equal(first.Take(cut), audio.Tracks[0].Take(cut));
        Assert.All(audio.Tracks[0].Skip(cut), sample => Assert.Equal(0f, sample));
        Assert.All(audio.Tracks[1].Take(cut), sample => Assert.Equal(0f, sample));
        Assert.Equal(second.Take(cut), audio.Tracks[1].Skip(cut));
        // What is heard switches speakers where the conditioning does.
        Assert.Equal(first.Take(cut), audio.Soundtrack.Take(cut));
        Assert.Equal(second.Take(cut), audio.Soundtrack.Skip(cut));
    }

    [Fact]
    public void TheSoundtrackComesFromTheRecordingsAndTheTracksFromTheSpeech()
    {
        float[] vocals = Tone(Rate, 0.1, 220), recording = Tone(Rate, 0.4, 440), other = Tone(Rate, 0.1, 330);

        LongCatSpeakerAudio audio = LongCatDrivingAudio.PrepareSpeakers(new LongCatSpeakerSpeech(vocals, recording), Speech(other),
            LongCatSpeakerTiming.Parallel, 25, 25, normalizeLoudness: false, silentTrack: false);

        Assert.Equal(vocals.Take(Rate), audio.Tracks[0].Take(Rate));
        Assert.Equal(recording[200] + other[200], audio.Soundtrack[200]);
    }

    [Fact]
    public void LoudnessIsSetPerSpeakerAndTheSilentTrackStaysSilent()
    {
        LongCatSpeakerAudio audio = LongCatDrivingAudio.PrepareSpeakers(Speech(Tone(2 * Rate, 0.02, 220)), Speech(Tone(2 * Rate, 0.5, 330)),
            LongCatSpeakerTiming.Parallel, 25, 25, normalizeLoudness: true, silentTrack: true);

        Assert.Equal(3, audio.Tracks.Length);
        Assert.InRange(Bs1770Loudness.Integrated(audio.Tracks[0], Rate), -23.001, -22.999);
        Assert.InRange(Bs1770Loudness.Integrated(audio.Tracks[1], Rate), -23.001, -22.999);
        Assert.All(audio.Tracks[2], sample => Assert.Equal(0f, sample));
        Assert.Equal(audio.Tracks[0].Length, audio.Tracks[2].Length);
    }

    [Fact]
    public void AnAbsentSpeakerIsSilentAndTwoAbsentSpeakersAreRefused()
    {
        float[] only = Tone(Rate, 0.2, 220);

        LongCatSpeakerAudio audio = LongCatDrivingAudio.PrepareSpeakers(null, Speech(only), LongCatSpeakerTiming.Parallel, 25, 25,
            normalizeLoudness: false, silentTrack: false);

        Assert.All(audio.Tracks[0], sample => Assert.Equal(0f, sample));
        Assert.Equal(only.Take(Rate), audio.Tracks[1].Take(Rate));
        Assert.Throws<ArgumentException>(() => LongCatDrivingAudio.PrepareSpeakers(null, null, LongCatSpeakerTiming.Parallel, 25, 25,
            normalizeLoudness: false, silentTrack: false));
    }

    [Fact]
    public void ParallelSpeakersWithAnEmptyClipAreRefused() =>
        Assert.Throws<ArgumentException>(() => LongCatDrivingAudio.PrepareSpeakers(Speech([]), Speech(Tone(Rate, 0.2, 220)),
            LongCatSpeakerTiming.Parallel, 25, 25, normalizeLoudness: false, silentTrack: false));
}
