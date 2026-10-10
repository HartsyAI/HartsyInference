using HartsyInference.Audio.Dsp.Telephony;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.CallAudio;

public sealed class CallProgressClassifierTests(ITestOutputHelper log)
{
    private const float Amp = 0.1f;

    /// <summary>Audio plus what the host would tell the classifier about it.</summary>
    private sealed class Script
    {
        public readonly List<float> Audio = [];
        public readonly List<float> Probability = [];
        public readonly List<bool> Local = [];

        public Script Add(float[] audio, float probability = 0f, bool local = false)
        {
            foreach (float s in audio)
            {
                Audio.Add(s);
                Probability.Add(probability);
                Local.Add(local);
            }
            return this;
        }

        public Script Silence(int ms) => Add(new float[CallAudioSynth.Ms(ms)]);

        public Script Speech(float[] speech) => Add(speech, 0.95f);

        public Script Tone(int ms, params (double, float)[] parts) => Add(new CallAudioSynth().Tones(ms, parts).ToArray());

        public Script Cadence(int onMs, int offMs, int cycles, params (double, float)[] parts)
        {
            for (int i = 0; i < cycles; i++)
            {
                Tone(onMs, parts);
                Silence(offMs);
            }
            return this;
        }

        public Script Mix(float[] noise)
        {
            for (int i = 0; i < Audio.Count; i++)
            {
                Audio[i] += noise[i % noise.Length];
            }
            return this;
        }
    }

    private static (double, float)[] Pair(double a, double b) => [(a, Amp), (b, Amp)];

    private static List<CallProgressEvent> Run(Script script, int chunk = 320, CallProgressClassifier? classifier = null)
    {
        classifier ??= new CallProgressClassifier();
        List<CallProgressEvent> events = [];
        float[] audio = [.. script.Audio];
        for (int i = 0; i < audio.Length; i += chunk)
        {
            int n = Math.Min(chunk, audio.Length - i);
            classifier.Process(audio.AsSpan(i, n), script.Probability[i], script.Local[i]);
            while (classifier.TryDequeue(out CallProgressEvent e))
            {
                events.Add(e);
            }
        }
        return events;
    }

    private static CallProgressKind[] Kinds(List<CallProgressEvent> events) => [.. events.Select(e => e.Kind).Distinct()];

    private static float Best(List<CallProgressEvent> events, CallProgressKind kind) =>
        events.Where(e => e.Kind == kind).Select(e => e.Confidence).DefaultIfEmpty(0f).Max();

    [Fact]
    public void NorthAmericanRingbackIsHeardDuringTheFirstRingAndConfirmedByTheSecond()
    {
        Script s = new Script().Silence(200).Cadence(2000, 4000, 2, Pair(440, 480));
        List<CallProgressEvent> events = Run(s);
        Assert.Equal([CallProgressKind.RingbackTone], Kinds(events));
        CallProgressEvent first = events[0];
        Assert.InRange(first.ConfirmedSample, CallAudioSynth.Ms(1000), CallAudioSynth.Ms(2500));
        Assert.True(Best(events, CallProgressKind.RingbackTone) >= 0.85f);
    }

    [Theory]
    [InlineData(480, 620, 500, CallProgressKind.BusyTone)]
    [InlineData(425, 0, 200, CallProgressKind.FastBusyTone)]
    public void BusyAndCongestionAreToldApartByCadence(double a, double b, int ms, CallProgressKind expected)
    {
        (double, float)[] parts = b == 0 ? [(a, 0.15f)] : Pair(a, b);
        Script s = new Script().Silence(200).Cadence(ms, ms, 5, parts);
        List<CallProgressEvent> events = Run(s);
        Assert.Equal([expected], Kinds(events));
        Assert.True(Best(events, expected) >= 0.9f, $"confidence {Best(events, expected)}");
        // Reliable once the second cycle has been heard.
        Assert.InRange(events[0].ConfirmedSample, 0, CallAudioSynth.Ms(200 + 3 * 2 * ms + 200));
    }

    [Fact]
    public void ALoneBeepIsABeepWithItsFrequencyAndNotAMachineGreeting()
    {
        List<CallProgressEvent> events = Run(new Script().Silence(500).Tone(400, (1000, 0.2f)).Silence(500));
        Assert.Equal([CallProgressKind.Beep], Kinds(events));
        CallProgressEvent beep = events[0];
        Assert.InRange(beep.FrequencyHz, 975f, 1025f);
        Assert.InRange(beep.DurationMs, 320, 520);
        Assert.True(beep.Confidence >= 0.8f);
    }

    [Fact]
    public void ABeepAfterALongGreetingIsStrongEvidenceOfAMachine()
    {
        Script s = new Script().Silence(300).Speech(CallAudioSynth.Jfk()).Silence(600).Tone(400, (1000, 0.2f)).Silence(400);
        List<CallProgressEvent> events = Run(s);
        CallProgressEvent machine = events.Last(e => e.Kind == CallProgressKind.MachineGreeting);
        Assert.Equal(CallProgressReason.BeepAfterGreeting, machine.Reason);
        Assert.True(machine.Confidence >= 0.9f);
    }

    [Fact]
    public void LongUninterruptedSpeechIsAWeakMachineSignalAndNeverCertain()
    {
        List<CallProgressEvent> events = Run(new Script().Silence(500).Speech(CallAudioSynth.Jfk()).Silence(2000));
        CallProgressEvent machine = Assert.Single(events, e => e.Kind == CallProgressKind.MachineGreeting);
        Assert.Equal(CallProgressReason.LongUninterruptedSpeech, machine.Reason);
        Assert.InRange(machine.Confidence, 0.5f, 0.7f);
        Assert.DoesNotContain(CallProgressKind.HumanSpeech, Kinds(events));
    }

    [Fact]
    public void SustainedNonSpeechSoundIsHoldMusicAtLowConfidenceAndSilenceIsPromptSilence()
    {
        float[] music = new CallAudioSynth().Tones(8000, (261.6, 0.05f), (329.6, 0.05f), (392.0, 0.05f), (523.2, 0.03f)).ToArray();
        music = CallAudioSynth.AddNoise(music, 15, seed: 2);
        List<CallProgressEvent> hold = Run(new Script().Add(music, 0.1f));
        CallProgressEvent music1 = Assert.Single(hold, e => e.Kind == CallProgressKind.HoldMusic);
        Assert.True(music1.Confidence <= 0.6f);

        List<CallProgressEvent> silent = Run(new Script().Add(music, 0.1f).Silence(4000));
        Assert.Contains(CallProgressKind.PromptSilence, Kinds(silent));
        Assert.Equal([CallProgressKind.PromptSilence], Kinds(Run(new Script().Silence(4000))));
    }

    [Fact]
    public void SpeechAloneRaisesNoToneEvents()
    {
        ToneKindsAbsent(new Script().Speech(CallAudioSynth.Normalize(CallAudioSynth.Jfk(), 0.1f)));
        ToneKindsAbsent(new Script().Speech(CallAudioSynth.Normalize(CallAudioSynth.Jfk(), 0.3f)).Speech(CallAudioSynth.Normalize(CallAudioSynth.Alexa(), 0.3f)));
        for (int seed = 1; seed <= 3; seed++)
        {
            ToneKindsAbsent(new Script().Speech(CallAudioSynth.Babble(16000 * 10, seed, 0.1f)));
        }
        ToneKindsAbsent(new Script().Add(CallAudioSynth.Noise(16000 * 20, 0.05f, 4), 0.1f));
    }

    private void ToneKindsAbsent(Script script)
    {
        CallProgressKind[] tones =
        [
            CallProgressKind.RingbackTone, CallProgressKind.BusyTone, CallProgressKind.FastBusyTone, CallProgressKind.SitTone,
            CallProgressKind.DialTone, CallProgressKind.Beep,
        ];
        List<CallProgressEvent> events = Run(script);
        foreach (CallProgressEvent e in events)
        {
            log.WriteLine($"  {e.Kind} {e.Confidence} {e.Reason} at {e.StartSample / 16000.0}s");
        }
        Assert.DoesNotContain(events, e => tones.Contains(e.Kind));
    }

    [Fact]
    public void TonesSurviveLineNoise()
    {
        float[] noise = CallAudioSynth.Noise(16000 * 30, 0.004f, 11);
        Script busy = new Script().Silence(200).Cadence(500, 500, 6, Pair(480, 620)).Mix(noise);
        Assert.Contains(CallProgressKind.BusyTone, Kinds(Run(busy)));
        Script ring = new Script().Silence(200).Cadence(2000, 4000, 2, Pair(440, 480)).Mix(noise);
        Assert.Contains(CallProgressKind.RingbackTone, Kinds(Run(ring)));
    }

}
