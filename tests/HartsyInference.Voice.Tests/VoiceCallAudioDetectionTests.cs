using HartsyInference.Audio.Dsp.Telephony;
using HartsyInference.Cpu;
using HartsyInference.Engine.Requests;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Tests.Fakes;
using HartsyInference.Voice.Turns;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The optional call-audio detectors in the session: off by default and then not run at all, on they raise
/// <see cref="VoiceAgentEventKind.InbandDtmfDetected"/> and <see cref="VoiceAgentEventKind.CallProgressDetected"/> from the
/// raw inbound audio, only <see cref="VoiceAgentOptions.ForwardInbandDtmfToModel"/> reaches the model, and neither changes
/// what the front end decides about speech.</summary>
public sealed class VoiceCallAudioDetectionTests
{
    private const int Rate = VoiceAudioFrontend.SampleRate;
    private const int Frame = VoiceAudioFrontend.FrameSamples;

    private static readonly double[] Low = [697, 770, 852, 941];
    private static readonly double[] High = [1209, 1336, 1477, 1633];
    private static readonly string Keys = "123A456B789C*0#D";

    private static float[] Tone(int ms, params (double Hz, float Amp)[] parts)
    {
        float[] x = new float[ms * Rate / 1000];
        for (int i = 0; i < x.Length; i++)
        {
            double sum = 0;
            foreach ((double hz, float amp) in parts)
            {
                sum += amp * Math.Sin(2 * Math.PI * hz * i / Rate);
            }
            x[i] = (float)sum;
        }
        return x;
    }

    private static float[] Key(char key, int ms)
    {
        int i = Keys.IndexOf(key);
        return Tone(ms, (Low[i / 4], 0.25f), (High[i % 4], 0.25f));
    }

    /// <summary>Silence, keys, a busy signal, then speech and silence: what a call into a phone menu sounds like.</summary>
    private static float[] CallAudio(string keys, bool speech = true)
    {
        List<float> audio = [];
        audio.AddRange(new float[Rate / 4]);
        foreach (char key in keys)
        {
            audio.AddRange(Key(key, 100));
            audio.AddRange(new float[Rate / 10]);
        }
        for (int i = 0; i < 4; i++)
        {
            audio.AddRange(Tone(500, (480, 0.1f), (620, 0.1f)));
            audio.AddRange(new float[Rate / 2]);
        }
        if (speech)
        {
            audio.AddRange(Tone(1000, (300, VoiceHarness.SpeechLevel)));
        }
        audio.AddRange(new float[Rate * 2]);
        return [.. audio];
    }

    private static VoiceAgentOptions On(bool forward = false) =>
        VoiceHarness.DefaultOptions() with { DetectInbandDtmf = true, DetectCallProgress = true, ForwardInbandDtmfToModel = forward };

    [Fact]
    public void TurningTheDetectorsOnLeavesEveryOtherFrontEndDecisionUntouched()
    {
        VoiceTurnSignals signals = new();
        using CpuBackend cpu = new();
        using VoiceAudioFrontend off = new(cpu, new LevelVadModel(), null, signals, new VoiceAgentOptions());
        using VoiceAudioFrontend on = new(cpu, new LevelVadModel(), null, signals, new VoiceAgentOptions { DetectInbandDtmf = true, DetectCallProgress = true });
        float[] audio = CallAudio("123#");
        const VoiceFrameEvents Detector = VoiceFrameEvents.InbandDtmf | VoiceFrameEvents.CallProgress;
        int speechDecisions = 0;
        int detections = 0;
        for (int i = 0; i + Frame <= audio.Length; i += Frame)
        {
            VoiceFrameEvents expected = off.ProcessFrame(audio.AsSpan(i, Frame));
            VoiceFrameEvents actual = on.ProcessFrame(audio.AsSpan(i, Frame));
            Assert.Equal(expected, actual & ~Detector);
            Assert.Equal(off.ClockSamples, on.ClockSamples);
            Assert.Equal(off.InSpeech, on.InSpeech);
            speechDecisions += expected != VoiceFrameEvents.None ? 1 : 0;
            detections += (actual & Detector) != 0 ? 1 : 0;
        }
        Assert.True(speechDecisions > 0, "the scenario must contain a decision to compare");
        Assert.True(detections > 0);
    }

    [Fact]
    public void TheFrontEndHearsKeysAndABusySignalInTheRawAudioWithOffsetsOnTheInputClock()
    {
        VoiceTurnSignals signals = new();
        using CpuBackend cpu = new();
        using VoiceAudioFrontend frontend = new(cpu, new LevelVadModel(), null, signals, new VoiceAgentOptions { DetectInbandDtmf = true, DetectCallProgress = true });
        float[] audio = CallAudio("42#");
        List<DtmfEvent> keys = [];
        List<CallProgressEvent> findings = [];
        for (int i = 0; i + Frame <= audio.Length; i += Frame)
        {
            frontend.ProcessFrame(audio.AsSpan(i, Frame));
            while (frontend.TryTakeDtmf(out DtmfEvent key))
            {
                keys.Add(key);
            }
            while (frontend.TryTakeCallProgress(out CallProgressEvent finding))
            {
                findings.Add(finding);
            }
        }
        Assert.Equal("42#", string.Concat(keys.Select(k => k.Digit)));
        for (int i = 0; i < keys.Count; i++)
        {
            long trueStart = Rate / 4 + i * (Rate / 5);
            Assert.InRange(keys[i].StartSample, trueStart - Rate / 50, trueStart + Rate / 100);
        }
        Assert.Contains(findings, f => f.Kind == CallProgressKind.BusyTone && f.Confidence >= 0.9f);
        Assert.DoesNotContain(findings, f => f.Kind is CallProgressKind.RingbackTone or CallProgressKind.SitTone);
    }

    [Fact]
    public void TheDetectorsAllocateNothingOnTheFrontEndFramePathAndDrainWithoutAllocating()
    {
        VoiceTurnSignals signals = new();
        using CpuBackend cpu = new();
        using VoiceAudioFrontend frontend = new(cpu, new LevelVadModel(), null, signals, new VoiceAgentOptions { DetectInbandDtmf = true, DetectCallProgress = true });
        float[] audio = CallAudio("0123456789*#ABCD");
        int frames = audio.Length / Frame;
        int taken = 0;
        void Pass(int first, int count)
        {
            for (int f = first; f < first + count; f++)
            {
                frontend.ProcessFrame(audio.AsSpan(f % frames * Frame, Frame));
                while (frontend.TryTakeDtmf(out _))
                {
                    taken++;
                }
                while (frontend.TryTakeCallProgress(out _))
                {
                    taken++;
                }
            }
        }
        Pass(0, frames);
        long before = GC.GetAllocatedBytesForCurrentThread();
        Pass(frames, frames);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(taken > 16);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void ForwardingWithoutDetectionIsRejected()
    {
        Assert.Throws<ArgumentException>(new VoiceAgentOptions { ForwardInbandDtmfToModel = true }.Validate);
        new VoiceAgentOptions { DetectInbandDtmf = true, ForwardInbandDtmfToModel = true }.Validate();
        new VoiceAgentOptions { DetectCallProgress = true }.Validate();
    }

    [Fact]
    public async Task ForwardedKeysReachTheModelAsInbandDtmfAndNotAsAKeypadPress()
    {
        // No speech in this call: an utterance that starts while the replies to the keys are playing is discarded as
        // overlap, which would make the turn count depend on playback timing.
        ScriptedTextService text = new ScriptedTextService().Reply("One.").Reply("Two.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(On(forward: true), text: text);
        harness.Push(CallAudio("42", speech: false));

        await harness.TurnCompletedAsync(2);
        string[] users = [.. text.Requests.SelectMany(r => r.Messages).Where(m => m.Role == TextRole.User).Select(m => m.Content).Distinct()];
        Assert.Contains("[INBAND DTMF 4]", users);
        Assert.Contains("[INBAND DTMF 2]", users);
        Assert.DoesNotContain("[DTMF 4]", users);
    }

}
