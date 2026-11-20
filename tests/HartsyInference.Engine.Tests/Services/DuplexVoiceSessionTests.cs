using System.Diagnostics;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The live pacer around a duplex session, on a fake model with 20 ms frames (480 samples at 24 kHz) so the clock can be exercised quickly.</summary>
public sealed class DuplexVoiceSessionTests
{
    private const int Rate = 24000;
    private const int Frame = 480;

    private sealed class FakeSession(int computeMilliseconds = 0, int stepsBeforeDone = int.MaxValue, int onlyStep = 0) : IDuplexSpeechSession
    {
        private int _steps;
        public List<float[]> Heard { get; } = [];
        public bool Disposed { get; private set; }
        public int SampleRate => Rate;
        public int FrameSamples => Frame;
        public int RemainingFrames => int.MaxValue;

        public DuplexFrame Start() => new(new float[Frame], "", 0, 0, false);

        public DuplexFrame Step(ReadOnlySpan<float> userFrame)
        {
            lock (Heard)
            {
                Heard.Add(userFrame.ToArray());
            }
            _steps++;
            if (computeMilliseconds > 0 && (onlyStep == 0 || _steps == onlyStep))
            {
                Thread.Sleep(computeMilliseconds);
            }
            float[] reply = new float[Frame];
            Array.Fill(reply, _steps);
            return new DuplexFrame(reply, _steps == 3 ? "three" : "", _steps, computeMilliseconds / 1000.0, _steps >= stepsBeforeDone);
        }

        public void Dispose() => Disposed = true;
    }

    private static async Task<List<DuplexVoiceFrame>> ReadAsync(DuplexVoiceSession session, int count, TimeSpan timeout)
    {
        using CancellationTokenSource cts = new(timeout);
        List<DuplexVoiceFrame> frames = [];
        await foreach (DuplexVoiceFrame frame in session.Frames.ReadAllAsync(cts.Token))
        {
            frames.Add(frame);
            if (frames.Count == count)
            {
                break;
            }
        }
        return frames;
    }

    [Fact]
    public async Task Frames_ArriveInOrder_OnePerTick_AndFrameZeroIsImmediate()
    {
        using DuplexVoiceSession session = new(new FakeSession(), new DuplexVoiceOptions { LeadFrames = 8 });
        Stopwatch clock = Stopwatch.StartNew();
        session.Start();
        List<DuplexVoiceFrame> frames = await ReadAsync(session, 11, TimeSpan.FromSeconds(10));

        Assert.Equal(Enumerable.Range(0, 11), frames.Select(f => f.Index));
        Assert.Equal("three", frames[3].Text);
        Assert.All(frames, f => Assert.Equal(Frame, f.Audio.Length));
        // Ten ticks of 20 ms: not faster than the clock, and not wildly slower.
        Assert.InRange(clock.Elapsed.TotalMilliseconds, 190, 1500);
        Assert.Equal(20.0, session.FrameMs, 1);
    }

    [Fact]
    public async Task UserAudio_IsDeliveredFrameByFrame_AndSilenceFillsTheGaps()
    {
        FakeSession fake = new();
        using DuplexVoiceSession session = new(fake, new DuplexVoiceOptions { LeadFrames = 8 });
        float[] speech = Enumerable.Range(0, Frame * 2).Select(i => 0.001f * (i + 1)).ToArray();
        session.PushUserAudio(speech); // two whole frames before the clock starts
        session.Start();
        await ReadAsync(session, 5, TimeSpan.FromSeconds(10));

        float[][] heard;
        lock (fake.Heard)
        {
            heard = [.. fake.Heard];
        }
        Assert.Equal(speech[..Frame], heard[0]);
        Assert.Equal(speech[Frame..], heard[1]);
        Assert.All(heard[2], v => Assert.Equal(0f, v));
        Assert.True(session.Stats().SilenceFrames >= 2);
    }

    [Fact]
    public async Task APartialFrame_IsZeroPaddedNotHeld()
    {
        FakeSession fake = new();
        using DuplexVoiceSession session = new(fake, new DuplexVoiceOptions { LeadFrames = 8 });
        float[] part = Enumerable.Repeat(0.5f, 100).ToArray();
        session.PushUserAudio(part);
        session.Start();
        await ReadAsync(session, 3, TimeSpan.FromSeconds(10));

        float[] first;
        lock (fake.Heard)
        {
            first = fake.Heard[0];
        }
        Assert.All(first[..100], v => Assert.Equal(0.5f, v));
        Assert.All(first[100..], v => Assert.Equal(0f, v));
    }

    [Fact]
    public async Task ASlowFrame_IsLateForNoLead_AndOnTimeWithEnoughLead()
    {
        // One frame takes 100 ms on a 20 ms slot: five frame lengths over it, once.
        using DuplexVoiceSession tight = new(new FakeSession(computeMilliseconds: 100, onlyStep: 3), new DuplexVoiceOptions { LeadFrames = 0 });
        tight.Start();
        List<DuplexVoiceFrame> tightFrames = await ReadAsync(tight, 8, TimeSpan.FromSeconds(10));
        Assert.Contains(tightFrames, f => f.Late);

        using DuplexVoiceSession buffered = new(new FakeSession(computeMilliseconds: 100, onlyStep: 3), new DuplexVoiceOptions { LeadFrames = 8 });
        buffered.Start();
        List<DuplexVoiceFrame> bufferedFrames = await ReadAsync(buffered, 8, TimeSpan.FromSeconds(10));
        Assert.DoesNotContain(bufferedFrames, f => f.Late);
        Assert.True(buffered.Stats().MaxComputeMs >= 90);
    }

    [Fact]
    public async Task FallingBehindByMoreThanTheLead_Rebases_InsteadOfRunningFramesBackToBack()
    {
        using DuplexVoiceSession session = new(new FakeSession(computeMilliseconds: 100), new DuplexVoiceOptions { LeadFrames = 1 });
        Stopwatch clock = Stopwatch.StartNew();
        session.Start();
        List<DuplexVoiceFrame> frames = await ReadAsync(session, 6, TimeSpan.FromSeconds(10));
        Assert.True(session.Stats().Rebases >= 1);
        // Never faster than the compute itself: five steps of 100 ms, no bursting.
        Assert.True(clock.Elapsed.TotalMilliseconds >= 450, $"{clock.Elapsed.TotalMilliseconds} ms for {frames.Count} frames");
    }

    [Fact]
    public async Task ADoneConversation_CompletesTheChannel()
    {
        using DuplexVoiceSession session = new(new FakeSession(stepsBeforeDone: 3), new DuplexVoiceOptions { LeadFrames = 8 });
        session.Start();
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(10));
        List<DuplexVoiceFrame> all = [];
        await foreach (DuplexVoiceFrame frame in session.Frames.ReadAllAsync(cts.Token))
        {
            all.Add(frame);
        }
        Assert.Equal(4, all.Count); // frame 0 and three steps
    }

    [Fact]
    public async Task Dispose_StopsTheClock_ReleasesTheSession_AndCompletesTheChannel()
    {
        FakeSession fake = new();
        DuplexVoiceSession session = new(fake, new DuplexVoiceOptions { LeadFrames = 8 });
        session.Start();
        await ReadAsync(session, 3, TimeSpan.FromSeconds(10));
        session.Dispose();
        session.Dispose();
        Assert.True(fake.Disposed);
        int stepsAtDispose;
        lock (fake.Heard)
        {
            stepsAtDispose = fake.Heard.Count;
        }
        await Task.Delay(100);
        lock (fake.Heard)
        {
            Assert.Equal(stepsAtDispose, fake.Heard.Count);
        }
        Assert.Throws<ObjectDisposedException>(() => session.Start());
    }

    [Fact]
    public void Start_TwiceIsRefused_AndABadLeadOrRateIsRejected()
    {
        using DuplexVoiceSession session = new(new FakeSession(), new DuplexVoiceOptions { LeadFrames = 1 });
        session.Start();
        Assert.Throws<InvalidOperationException>(() => session.Start());
        Assert.Throws<ArgumentOutOfRangeException>(() => new DuplexVoiceSession(new FakeSession(), new DuplexVoiceOptions { LeadFrames = 9 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DuplexVoiceSession(new FakeSession(), new DuplexVoiceOptions { InputSampleRate = 12345 }));
    }

    [Fact]
    public async Task AnInputRateOtherThanTheModels_IsResampledToTheFrame()
    {
        FakeSession fake = new();
        using DuplexVoiceSession session = new(fake, new DuplexVoiceOptions { LeadFrames = 8, InputSampleRate = 16000 });
        // 60 ms of a tone at 16 kHz is 960 samples in, three chunks of 320; each becomes 480 samples at 24 kHz, one model frame (the filter holds one chunk back).
        float[] tone = Enumerable.Range(0, 960).Select(i => (float)Math.Sin(2 * Math.PI * 440 * i / 16000)).ToArray();
        session.PushUserAudio(tone);
        session.Start();
        await ReadAsync(session, 7, TimeSpan.FromSeconds(10));
        float[][] heard;
        lock (fake.Heard)
        {
            heard = [.. fake.Heard];
        }
        Assert.True(heard.Take(3).SelectMany(f => f).Max(Math.Abs) > 0.1f);
        Assert.All(heard[5], v => Assert.Equal(0f, v));
    }

    [Fact]
    public async Task AReaderThatFallsBehind_LosesTheOldestFrames_AndTheyAreCounted()
    {
        using DuplexVoiceSession session = new(new FakeSession(), new DuplexVoiceOptions { LeadFrames = 0 });
        session.Start();
        await Task.Delay(700); // about 35 frames against a queue of 16, with nobody reading
        DuplexVoiceStats stats = session.Stats();
        Assert.True(stats.DroppedReplyFrames > 5, $"dropped {stats.DroppedReplyFrames}");
        DuplexVoiceFrame oldest = await session.Frames.ReadAsync();
        Assert.True(oldest.Index > 0, "the queue should have lost frame 0");
    }

    [Fact]
    public async Task AudioThatPilesUpBeyondTheLead_IsTrimmedOldestFirst_AndCounted()
    {
        FakeSession fake = new();
        using DuplexVoiceSession session = new(fake, new DuplexVoiceOptions { LeadFrames = 1 });
        // Five frames of a ramp before the clock starts: one is read at the first tick, and of the other four only the newest two (the lead plus the frame
        // being read) are kept.
        float[] ramp = Enumerable.Range(0, Frame * 5).Select(i => (i + 1) * 1e-4f).ToArray();
        session.PushUserAudio(ramp);
        session.Start();
        await ReadAsync(session, 4, TimeSpan.FromSeconds(10));

        float[][] heard;
        lock (fake.Heard)
        {
            heard = [.. fake.Heard];
        }
        Assert.Equal(ramp[..Frame], heard[0]);
        Assert.Equal(ramp[(Frame * 3)..(Frame * 4)], heard[1]);
        Assert.Equal(ramp[(Frame * 4)..], heard[2]);
        Assert.True(session.Stats().DroppedUserSamples >= Frame * 2);
    }

    [Fact]
    public void ANegativeInputRate_IsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DuplexVoiceSession(new FakeSession(), new DuplexVoiceOptions { InputSampleRate = -16000 }));
    }
}
