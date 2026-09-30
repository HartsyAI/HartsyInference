using System.Diagnostics;
using HartsyInference.Audio.Io;
using HartsyInference.PhoneGateway.Media;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>The RTP clock is the one thing in the gateway that must hold a cadence, and a drift, a missing frame or a
/// per-tick allocation would never surface as an exception. A fake subscriber that only counts stands in for
/// sipsorcery's <c>SendAudio</c>, so the allocation assertion measures our tick path alone.</summary>
public sealed class ClockedAudioSourceCadenceTests
{
    private readonly ITestOutputHelper _output;
    public ClockedAudioSourceCadenceTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task ThreeSecondRun_FrameCountMatchesElapsedWithinOneFrame()
    {
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions { FifoPriority = 50 });
        CountingSubscriber subscriber = new();
        source.OnAudioSourceEncodedSample += subscriber.OnFrame;
        source.Start();
        Stopwatch clock = Stopwatch.StartNew();
        await Task.Delay(3000);
        source.Stop();
        clock.Stop();
        long frames = source.FramesSent;
        long expected = source.ElapsedNs / ClockedAudioSource.PeriodNs;
        long wall = (long)(clock.Elapsed.TotalMilliseconds / 20);
        _output.WriteLine($"frames={frames} expected(source clock)={expected} wall={wall} fifo={source.FifoActive} " +
            $"catchUp={source.CatchUpFrames} resyncs={source.Resyncs} lateness={source.Lateness.Snapshot()}");
        Assert.False(source.Faulted);
        Assert.Equal(frames, subscriber.Frames);
        Assert.InRange(frames, expected - 1, expected + 1);
        Assert.InRange(frames, wall - 2, wall + 3);
        Assert.Equal(0, source.Resyncs);
        Assert.Equal(160u, subscriber.LastDuration);
        Assert.Equal(160, subscriber.LastLength);
    }

    [Fact]
    public async Task TickThread_AllocatesNothingAfterWarmUp()
    {
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions());
        CountingSubscriber subscriber = new();
        source.OnAudioSourceEncodedSample += subscriber.OnFrame;
        short[] tone = new short[2000];
        for (int i = 0; i < tone.Length; i++)
        {
            tone[i] = (short)(Math.Sin(i * 0.2) * 8000);
        }
        source.Start();
        // Mix real audio and silence so both branches of the tick run.
        for (int i = 0; i < 4; i++)
        {
            source.WriteOutbound(tone);
            await Task.Delay(500);
        }
        await Task.Delay(500);
        source.Stop();
        _output.WriteLine($"ticks={source.Ticks} silence={source.SilenceFrames} allocated={source.TickThreadAllocatedBytes}");
        Assert.False(source.Faulted);
        Assert.True(source.Ticks > 100);
        Assert.True(source.SilenceFrames > 0);
        Assert.True(source.SilenceFrames < source.Ticks);
        Assert.Equal(0, source.TickThreadAllocatedBytes);
    }

    [Fact]
    public async Task Flush_DrainsTheRingWithinOneTick()
    {
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions());
        CountingSubscriber subscriber = new();
        source.OnAudioSourceEncodedSample += subscriber.OnFrame;
        source.Start();
        short[] audio = new short[16000];
        Array.Fill(audio, (short)1000);
        Assert.Equal(audio.Length, source.WriteOutbound(audio));
        await Task.Delay(100);
        int queued = source.Flush();
        Assert.True(queued > 10_000, $"queued={queued}");
        await Task.Delay(45);
        int remaining = source.Available;
        long flushed = source.FlushedSamples;
        source.Stop();
        _output.WriteLine($"queued={queued} flushed={flushed} remaining={remaining}");
        Assert.Equal(0, remaining);
        Assert.InRange(flushed, queued - 2 * ClockedAudioSource.FrameSamples, queued);
    }

    [Fact]
    public void Fifo_IsRefusedHereWithAnActionableReason()
    {
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions { FifoPriority = 50 });
        source.OnAudioSourceEncodedSample += new CountingSubscriber().OnFrame;
        source.Start();
        source.Stop();
        _output.WriteLine($"fifo={source.FifoActive} reason={source.FifoReason}");
        Assert.False(source.Faulted);
        if (source.FifoActive)
        {
            Assert.Equal("", source.FifoReason);
            return;
        }
        Assert.Contains("LimitRTPRIO", source.FifoReason, StringComparison.Ordinal);
        Assert.Contains("limits.d", source.FifoReason, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Pause_StopsRaisingButKeepsTheClock()
    {
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions());
        CountingSubscriber subscriber = new();
        source.OnAudioSourceEncodedSample += subscriber.OnFrame;
        source.Start();
        await Task.Delay(100);
        await source.PauseAudio();
        long atPause = subscriber.Frames;
        await Task.Delay(100);
        long ticksWhilePaused = source.Ticks;
        Assert.Equal(atPause, subscriber.Frames);
        Assert.True(ticksWhilePaused > atPause);
        await source.ResumeAudio();
        await Task.Delay(100);
        source.Stop();
        Assert.True(subscriber.Frames > atPause);
    }

    [Fact]
    public void SubscriberFault_RaisesTickFaultedOnceOnTheTickThreadAndEndsIt()
    {
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions());
        InvalidOperationException injected = new("injected tick fault");
        long frames = 0;
        source.OnAudioSourceEncodedSample += (_, _) =>
        {
            if (Interlocked.Increment(ref frames) == 5)
            {
                throw injected;
            }
        };
        int raised = 0;
        Exception? reported = null;
        string? thread = null;
        using ManualResetEventSlim faulted = new(false);
        source.TickFaulted += ex =>
        {
            Interlocked.Increment(ref raised);
            reported = ex;
            thread = Thread.CurrentThread.Name;
            faulted.Set();
        };
        source.Start();
        Assert.True(faulted.Wait(5000), "TickFaulted was never raised");
        source.Stop();
        Assert.Same(injected, reported);
        Assert.Equal("phone-rtp-tick", thread);
        Assert.Equal(1, raised);
        Assert.True(source.Faulted);
        Assert.False(source.IsRunning);
        Assert.Equal(5, Interlocked.Read(ref frames));
    }

    [Fact]
    public void StartFault_IsReportedAndStartStillReturns()
    {
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions());
        InvalidOperationException injected = new("injected start fault");
        source.InjectedStartFault = injected;
        Exception? reported = null;
        using ManualResetEventSlim faulted = new(false);
        source.TickFaulted += ex =>
        {
            reported = ex;
            faulted.Set();
        };
        source.Start();
        Assert.True(source.Faulted, "Faulted must be set before Start returns");
        Assert.True(faulted.Wait(5000));
        source.Stop();
        Assert.Same(injected, reported);
        Assert.Equal(0, source.Ticks);
        InvalidOperationException restart = Assert.Throws<InvalidOperationException>(source.Start);
        Assert.Contains("faulted", restart.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SetAudioSourceFormat_SwitchesTheLaw()
    {
        using ClockedAudioSource source = new(new ClockedAudioSourceOptions());
        Assert.Equal(G711Law.MuLaw, source.Law);
        source.SetAudioSourceFormat(new SIPSorceryMedia.Abstractions.AudioFormat(SIPSorceryMedia.Abstractions.SDPWellKnownMediaFormatsEnum.PCMA));
        Assert.Equal(G711Law.ALaw, source.Law);
        source.SetAudioSourceFormat(new SIPSorceryMedia.Abstractions.AudioFormat(SIPSorceryMedia.Abstractions.SDPWellKnownMediaFormatsEnum.G722));
        Assert.Equal(G711Law.ALaw, source.Law);
        Assert.Equal(2, source.GetAudioSourceFormats().Count);
    }

    private sealed class CountingSubscriber
    {
        private long _frames;
        public long Frames => Volatile.Read(ref _frames);
        public uint LastDuration;
        public int LastLength;

        public void OnFrame(uint durationRtpUnits, byte[] sample)
        {
            LastDuration = durationRtpUnits;
            LastLength = sample.Length;
            Volatile.Write(ref _frames, _frames + 1);
        }
    }
}
