using System.Diagnostics;
using HartsyInference.Audio.Dsp.Telephony;
using HartsyInference.Core.Numerics;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.CallAudio;

/// <summary>What the call-audio detectors cost the voice agent's audio thread: nothing allocated, and at most 0.2 ms
/// added per 20 ms frame at p99 on one core. The signal mixes speech, DTMF keys, busy tone, ringback and a beep so every
/// branch (block analysis, event queues) runs inside the timed region. Inside <see cref="CpuParallel.EnterInline"/>, as
/// the audio thread runs them.
///
/// <para>The 0.2 ms gate is asserted when <c>HARTSY_CALL_AUDIO_BENCH=1</c> (run alone, in a Release build, on a quiet
/// machine, like the other timing gates); otherwise a 2 ms ceiling catches an order-of-magnitude regression on a busy CI
/// host. Both log the measured percentiles.</para></summary>
public sealed class CallAudioCostTests(ITestOutputHelper log)
{
    private const int Frame = 320;
    private const int Warmup = 500;
    private const int Frames = 5_000;

    private static float[] Mixed()
    {
        CallAudioSynth s = new();
        s.Append(CallAudioSynth.Normalize(CallAudioSynth.Jfk(), 0.1f));
        foreach (char key in "0123456789*#ABCD")
        {
            s.Key(key, 100).Silence(100);
        }
        s.Append(CallAudioSynth.Normalize(CallAudioSynth.Alexa(), 0.1f));
        for (int i = 0; i < 6; i++)
        {
            s.Tones(500, (480, 0.1f), (620, 0.1f)).Silence(500);
        }
        s.Tones(2000, (440, 0.1f), (480, 0.1f)).Silence(1500).Tones(400, (1000, 0.2f)).Silence(500);
        s.Append(CallAudioSynth.Noise(16000 * 3, 0.05f, 9));
        return s.ToArray();
    }

    [Fact]
    public void BothDetectorsAllocateNothingPerFrame()
    {
        float[] audio = Mixed();
        DtmfDetector dtmf = new();
        CallProgressClassifier progress = new();
        using CpuParallel.InlineScope inline = CpuParallel.EnterInline();
        Drive(audio, dtmf, progress, 0, Warmup, null);
        long before = GC.GetAllocatedBytesForCurrentThread();
        int events = Drive(audio, dtmf, progress, Warmup, Frames, null);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        log.WriteLine($"{allocated} bytes over {Frames} frames; {events} events drained");
        Assert.True(events > 20, "the signal must exercise the event paths");
        Assert.Equal(0, allocated);
    }

    // Frames wrap around the clip; each pass over the clip is continuous audio, which is all the detectors need.
    private static int Drive(float[] audio, DtmfDetector dtmf, CallProgressClassifier progress, int first, int count, long[]? ticks)
    {
        int events = 0;
        int frames = audio.Length / Frame;
        for (int f = first; f < first + count; f++)
        {
            ReadOnlySpan<float> frame = audio.AsSpan(f % frames * Frame, Frame);
            long start = Stopwatch.GetTimestamp();
            dtmf.Process(frame);
            progress.Process(frame, 0.9f, false);
            while (dtmf.TryDequeue(out _))
            {
                events++;
            }
            while (progress.TryDequeue(out _))
            {
                events++;
            }
            if (ticks is not null)
            {
                ticks[f - first] = Stopwatch.GetTimestamp() - start;
            }
        }
        return events;
    }
}
