using HartsyInference.Audio.Dsp.Telephony;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.CallAudio;

public sealed class DtmfDetectorTests(ITestOutputHelper log)
{
    private const string AllKeys = "123A456B789C*0#D";

    private static List<DtmfEvent> Run(float[] audio, int chunk = 320, DtmfDetector? detector = null)
    {
        detector ??= new DtmfDetector();
        List<DtmfEvent> events = [];
        for (int i = 0; i < audio.Length; i += chunk)
        {
            detector.Process(audio.AsSpan(i, Math.Min(chunk, audio.Length - i)));
            while (detector.TryDequeue(out DtmfEvent e))
            {
                events.Add(e);
            }
        }
        return events;
    }

    private static string Digits(List<DtmfEvent> events) => string.Concat(events.Select(e => e.Digit));

    private static float[] AllKeysSignal(int toneMs = 100, int pauseMs = 100, float amplitude = 0.25f, double twistDb = 0)
    {
        CallAudioSynth s = new();
        s.Silence(200);
        foreach (char key in AllKeys)
        {
            s.Key(key, toneMs, amplitude, twistDb).Silence(pauseMs);
        }
        return s.Silence(200).ToArray();
    }

    [Fact]
    public void AllSixteenKeysAreReportedOnceEachInOrderWithTheirOffsets()
    {
        List<DtmfEvent> events = Run(AllKeysSignal());
        Assert.Equal(AllKeys, Digits(events));
        for (int i = 0; i < events.Count; i++)
        {
            long trueStart = CallAudioSynth.Ms(200 + i * 200);
            Assert.InRange(events[i].StartSample, trueStart - CallAudioSynth.Ms(20), trueStart + CallAudioSynth.Ms(10));
            Assert.InRange(events[i].ConfirmedSample - trueStart, CallAudioSynth.Ms(40), CallAudioSynth.Ms(75));
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(97)]
    [InlineData(320)]
    [InlineData(1000)]
    public void TheChunkSizeDoesNotChangeTheEvents(int chunk)
    {
        float[] audio = AllKeysSignal();
        Assert.Equal(Run(audio), Run(audio, chunk));
    }

    [Theory]
    [InlineData(20.0)]
    [InlineData(10.0)]
    [InlineData(3.0)]
    public void AllKeysSurviveWhiteNoise(double snrDb)
    {
        float[] audio = CallAudioSynth.AddNoise(AllKeysSignal(), snrDb, seed: 7);
        Assert.Equal(AllKeys, Digits(Run(audio)));
    }

    [Fact]
    public void TheQuietestKeysAreStillHeard()
    {
        // -45 dBFS per component, with a line noise floor 12 dB under it.
        float[] audio = CallAudioSynth.AddNoise(AllKeysSignal(amplitude: 0.0056f), 12, seed: 3);
        Assert.Equal(AllKeys, Digits(Run(audio)));
    }

    [Theory]
    [InlineData(3.0, true)]
    [InlineData(-7.0, true)]
    [InlineData(-3.0, true)]
    [InlineData(-12.0, false)]
    [InlineData(8.0, false)]
    public void TwistWithinTheLimitsIsAcceptedAndBeyondIsRejected(double highRelativeDb, bool detected)
    {
        // Negative = the high tone is weaker (forward twist, up to 8 dB); positive = stronger (reverse, up to 4 dB).
        float[] audio = new CallAudioSynth().Silence(100).Key('5', 150, 0.25f, highRelativeDb).Silence(100).ToArray();
        Assert.Equal(detected ? "5" : "", Digits(Run(audio)));
    }

    [Fact]
    public void ATonePairFromOneGroupOrASingleToneIsNotAKey()
    {
        CallAudioSynth s = new();
        s.Silence(100).Tones(200, (697, 0.25f), (770, 0.25f)).Silence(100);
        s.Tones(200, (1209, 0.25f), (1336, 0.25f)).Silence(100);
        s.Tones(200, (1000, 0.4f)).Silence(100);
        s.Tones(200, (697, 0.4f)).Silence(100);
        s.Tones(200, (440, 0.2f), (480, 0.2f)).Silence(100);
        s.Tones(200, (350, 0.2f), (440, 0.2f)).Silence(100);
        Assert.Empty(Run(s.ToArray()));
    }

    [Fact]
    public void AToneOffTheNominalFrequenciesByTwoPercentIsStillAKeyAndByFivePercentIsNot()
    {
        float[] near = new CallAudioSynth().Silence(100).Tones(150, (697 * 1.015, 0.25f), (1209 * 0.985, 0.25f)).Silence(100).ToArray();
        float[] far = new CallAudioSynth().Silence(100).Tones(150, (697 * 1.06, 0.25f), (1209 * 0.94, 0.25f)).Silence(100).ToArray();
        Assert.Equal("1", Digits(Run(near)));
        Assert.Empty(Run(far));
    }

    [Fact]
    public void ATruncatedToneIsIgnoredAndAShortOneIsKept()
    {
        Assert.Empty(Run(new CallAudioSynth().Silence(100).Key('7', 25).Silence(100).ToArray()));
        Assert.Equal("7", Digits(Run(new CallAudioSynth().Silence(100).Key('7', 60).Silence(100).ToArray())));
        // Cut by the end of the stream mid-tone: still reported once it has lasted long enough.
        Assert.Equal("7", Digits(Run(new CallAudioSynth().Silence(100).Key('7', 120).ToArray())));
    }

    [Fact]
    public void ADropoutShorterThanThePauseKeepsOneKeyAndALongerOneMakesTwo()
    {
        float[] shortGap = new CallAudioSynth().Silence(100).Key('4', 100).Silence(20).Key('4', 100).Silence(100).ToArray();
        float[] longGap = new CallAudioSynth().Silence(100).Key('4', 100).Silence(80).Key('4', 100).Silence(100).ToArray();
        Assert.Equal("4", Digits(Run(shortGap)));
        Assert.Equal("44", Digits(Run(longGap)));
    }

    [Fact]
    public void KeysWithNoPauseBetweenThemAreEachReported()
    {
        float[] audio = new CallAudioSynth().Silence(100).Key('1', 100).Key('2', 100).Key('3', 100).Silence(100).ToArray();
        Assert.Equal("123", Digits(Run(audio)));
    }

    [Fact]
    public void KeysOverSpeechAreStillHeard()
    {
        float[] speech = CallAudioSynth.Normalize(CallAudioSynth.Jfk(), 0.05f);
        float[] keys = AllKeysSignal(toneMs: 120, pauseMs: 100, amplitude: 0.15f);
        float[] mixed = new float[Math.Min(keys.Length, speech.Length)];
        for (int i = 0; i < mixed.Length; i++)
        {
            mixed[i] = keys[i] + speech[i];
        }
        List<DtmfEvent> events = Run(mixed);
        log.WriteLine($"keys over speech: {Digits(events)}");
        string heard = Digits(events);
        Assert.True(heard.Length >= AllKeys.Length - 2, $"heard {heard}");
        Assert.True(IsSubsequence(heard, AllKeys), $"heard {heard}, which is not the keys that were pressed in order");
    }

    private static bool IsSubsequence(string part, string whole)
    {
        int at = 0;
        foreach (char c in part)
        {
            at = whole.IndexOf(c, at) + 1;
            if (at == 0)
            {
                return false;
            }
        }
        return true;
    }

    [Fact]
    public void SpeechAndNoiseProduceNoKeys()
    {
        // Real speech is 14.6 s: a smoke-level talk-off check, not a Bellcore run. The synthetic babble adds minutes of
        // voiced audio whose partials sweep across every DTMF frequency.
        Dictionary<string, int> falseKeys = [];
        double seconds = 0;
        void Count(string source, float[] x)
        {
            List<DtmfEvent> events = Run(x);
            falseKeys[source] = falseKeys.GetValueOrDefault(source) + events.Count;
            seconds += x.Length / 16000.0;
            if (events.Count > 0)
            {
                log.WriteLine($"  {source}: {Digits(events)} at {string.Join(",", events.Select(e => e.StartSample / 16000.0))}");
            }
        }
        foreach (float rms in new[] { 0.02f, 0.1f, 0.3f })
        {
            Count($"jfk@{rms}", CallAudioSynth.Normalize(CallAudioSynth.Jfk(), rms));
            Count($"alexa@{rms}", CallAudioSynth.Normalize(CallAudioSynth.Alexa(), rms));
        }
        for (int seed = 1; seed <= 24; seed++)
        {
            Count($"babble{seed}", CallAudioSynth.Babble(16000 * 10, seed, 0.1f));
        }
        foreach (float rms in new[] { 0.001f, 0.01f, 0.2f })
        {
            Count($"noise@{rms}", CallAudioSynth.Noise(16000 * 30, rms, seed: 5));
        }
        int total = falseKeys.Values.Sum();
        log.WriteLine($"talk-off: {total} false keys in {seconds:F0} s of speech, babble and noise");
        Assert.Equal(0, total);
    }
}
