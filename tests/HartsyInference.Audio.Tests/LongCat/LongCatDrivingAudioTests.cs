using HartsyInference.Audio.Dsp.Loudness;
using HartsyInference.Audio.Models.LongCat;
using Xunit;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>The loudness meter against pyloudnorm 0.1.1's values on synthetic signals, and the cut, padding and normalisation of the driving
/// speech. The known values are printed by tests/python-reference/longcat_audio/dump_driving_audio.py.</summary>
[Trait("Category", "Unit")]
public sealed class LongCatDrivingAudioTests
{
    private const int Rate = LongCatDrivingAudio.SampleRate;

    private static float[] Tone(double seconds, double amplitude, double hertz)
    {
        float[] samples = new float[(int)Math.Round(seconds * Rate)];
        for (int n = 0; n < samples.Length; n++)
            samples[n] = (float)(amplitude * Math.Sin(2 * Math.PI * hertz * n / Rate));
        return samples;
    }

    [Theory]
    [InlineData("tone", -15.779499216792901)]
    [InlineData("gated", -10.464508718956594)]
    [InlineData("one block", -23.076910951560688)]
    [InlineData("uneven", -14.466379507743339)]
    public void IntegratedLoudnessMatchesPyloudnorm(string name, double expected)
    {
        float[] signal = name switch
        {
            "tone" => Tone(2.0, 0.25, 440),
            // A loud second, a silent one and a quiet one: both gates drop blocks.
            "gated" => [.. Tone(1.0, 0.5, 440), .. new float[Rate], .. Tone(1.0, 0.01, 1000)],
            "one block" => Tone(0.4, 0.1, 1000),
            _ => Tone(1.23, 0.3, 200),
        };

        Assert.Equal(expected, Bs1770Loudness.Integrated(signal, Rate), 1e-6);
    }

    [Fact]
    public void SilenceHasNoLoudnessAndIsLeftAlone()
    {
        Assert.Equal(double.NegativeInfinity, Bs1770Loudness.Integrated(new float[Rate], Rate));

        float[] prepared = LongCatDrivingAudio.Prepare(new float[Rate], 93, 25, normalizeLoudness: true);

        Assert.All(prepared, sample => Assert.Equal(0f, sample));
    }

    [Fact]
    public void AudioShorterThanABlockHasNoMeasurement()
    {
        Assert.Throws<ArgumentException>(() => Bs1770Loudness.Integrated(new float[(int)(0.4 * Rate) - 1], Rate));
    }

    // The reference's lengths: its separator cuts the input at the clip plus one second, and its demo pads a shorter result up to there.
    [Theory]
    [InlineData(93, 25, 200_000, 75_521)]
    [InlineData(253, 25, 200_000, 177_920)]
    [InlineData(93, 25, 1_000, 75_521)]
    [InlineData(17, 25, 19_200, 26_881)]
    public void SpeechIsCutOrPaddedToTheClipPlusOneSecond(int clipFrames, int fps, int samples, int expected)
    {
        float[] speech = Tone(samples / (double)Rate, 0.2, 300);

        float[] prepared = LongCatDrivingAudio.Prepare(speech, clipFrames, fps, normalizeLoudness: false);

        Assert.Equal(expected, prepared.Length);
        int kept = Math.Min(samples, expected);
        Assert.Equal(speech.AsSpan(0, kept).ToArray(), prepared.AsSpan(0, kept).ToArray());
        Assert.All(prepared.Skip(kept), sample => Assert.Equal(0f, sample));
    }

    [Fact]
    public void NormalisationMovesThePreparedSpeechToMinus23LufsAndLeavesTheInput()
    {
        float[] speech = Tone(2.0, 0.5, 440);
        float[] original = [.. speech];

        float[] prepared = LongCatDrivingAudio.Prepare(speech, 93, 25, normalizeLoudness: true);

        Assert.Equal(LongCatDrivingAudio.TargetLufs, Bs1770Loudness.Integrated(prepared, Rate), 1e-5);
        Assert.Equal(original, speech);
    }

    [Fact]
    public void LoudnessIsMeasuredOnWhatIsKept()
    {
        // Quiet inside the cut, loud after it: only the quiet part sets the gain.
        float[] speech = [.. Tone(4.8, 0.01, 440), .. Tone(2.0, 0.9, 440)];

        float[] prepared = LongCatDrivingAudio.Prepare(speech, 93, 25, normalizeLoudness: true);

        Assert.Equal(LongCatDrivingAudio.TargetLufs, Bs1770Loudness.Integrated(prepared, Rate), 1e-5);
    }
}
