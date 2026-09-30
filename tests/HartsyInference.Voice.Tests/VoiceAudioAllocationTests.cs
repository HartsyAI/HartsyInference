using HartsyInference.Cpu;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Tests.Fakes;
using HartsyInference.Voice.Turns;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>The audio thread's zero-allocation contract, measured with <see cref="GC.GetAllocatedBytesForCurrentThread"/>
/// on the thread that does the work: the front-end over 1000 frames that open, close, cut and barge in, and the real
/// audio thread (ring drain, doorbell wait) over 1000 frames. The VAD is scripted, so this covers the session's own
/// per-frame code; the real Silero is measured in the Integration lane, where the CPU kernels it calls are counted too.</summary>
public sealed class VoiceAudioAllocationTests
{
    private const int Frames = 1_000;

    private readonly ITestOutputHelper _output;

    public VoiceAudioAllocationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void TheFrontEndAllocatesNothingOver1000Frames()
    {
        VoiceTurnSignals signals = new();
        using CpuBackend cpu = new();
        using VoiceAudioFrontend frontend = new(cpu, new LevelVadModel(), null, signals, new VoiceAgentOptions { MaxUtteranceMs = 3_000 });
        float[] speech = new float[VoiceAudioFrontend.FrameSamples];
        float[] silence = new float[VoiceAudioFrontend.FrameSamples];
        Array.Fill(speech, VoiceHarness.SpeechLevel);

        Run(frontend, signals, speech, silence, firstTurn: 1);
        frontend.Reset();
        long before = GC.GetAllocatedBytesForCurrentThread();
        VoiceFrameEvents seen = Run(frontend, signals, speech, silence, firstTurn: 3);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        _output.WriteLine($"front-end: {allocated} bytes over {Frames} frames; events seen: {seen}");
        Assert.Equal(VoiceFrameEvents.SpeechStarted | VoiceFrameEvents.Endpoint | VoiceFrameEvents.BargeIn, seen);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public async Task TheAudioThreadAllocatesNothingOver1000Frames()
    {
        VoiceTurnSignals signals = new();
        using CpuBackend cpu = new();
        VoiceAudioFrontend frontend = new(cpu, new LevelVadModel(), null, signals, new VoiceAgentOptions { MaxUtteranceMs = 60_000 });
        VoiceEndpointingTests.RecordingSink sink = new();
        using VoiceAudioWorker worker = new(frontend, signals, sink, inboundCapacity: 1 << 17, backlogLimitSamples: 1 << 17);
        worker.Start();
        await worker.Ready.WaitAsync(TimeSpan.FromSeconds(10));

        await PushAsync(worker, 0, 200);
        long before = worker.AllocatedBytes;
        await PushAsync(worker, 200, Frames);
        long allocated = worker.AllocatedBytes - before;

        _output.WriteLine($"audio thread: {allocated} bytes over {Frames} frames");
        Assert.Empty(sink.Utterances);
        Assert.Equal(0, allocated);
    }

    /// <summary>1000 frames: an utterance that ends, one cut at the 3 s maximum, then two replies the caller talks over.</summary>
    private static VoiceFrameEvents Run(VoiceAudioFrontend frontend, VoiceTurnSignals signals, float[] speech, float[] silence, int firstTurn)
    {
        VoiceFrameEvents seen = VoiceFrameEvents.None;
        for (int f = 0; f < Frames; f++)
        {
            if (f == 400)
            {
                signals.BeginSpeaking(firstTurn);
            }
            else if (f == 650)
            {
                signals.BeginSpeaking(firstTurn + 1);
            }
            bool talking = f is (>= 50 and < 100) or (>= 150 and < 350) or (>= 450 and < 550) or (>= 700 and < 800);
            seen |= frontend.ProcessFrame(talking ? speech : silence);
        }
        signals.EndSpeaking(firstTurn + 1);
        return seen;
    }

    /// <summary>Pushes frames <paramref name="from"/>..<paramref name="count"/> of a 400 ms talk / 300 ms pause pattern in
    /// 20 ms pushes, as the gateway does, 25 at a time, waiting for the thread to drain each batch so it never falls far
    /// enough behind to trim its backlog.</summary>
    private static async Task PushAsync(VoiceAudioWorker worker, int from, int count)
    {
        float[] speech = new float[VoiceAudioFrontend.FrameSamples];
        float[] silence = new float[VoiceAudioFrontend.FrameSamples];
        Array.Fill(speech, VoiceHarness.SpeechLevel);
        long target = worker.ProcessedFrames;
        DateTime deadline = DateTime.UtcNow.AddSeconds(30);
        for (int f = from; f < from + count; f++)
        {
            worker.Push(f % 35 < 20 ? speech : silence);
            target++;
            if (target % 25 != 0 && f != from + count - 1)
            {
                continue;
            }
            while (worker.ProcessedFrames < target)
            {
                Assert.True(DateTime.UtcNow < deadline, $"the audio thread processed {worker.ProcessedFrames} of {target} frames.");
                await Task.Delay(1);
            }
        }
        Assert.Equal(0, worker.DroppedSamples);
    }
}
