using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneLink;
using HartsyInference.VoiceHost.Link;
using HartsyInference.VoiceHost.Tests.Support;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>The 20 ms sender: on an endless reply it moves full 20 ms frames at its period, starts the burst with the
/// prebuffer, and its audio path allocates nothing over more than 1000 frames once warm. The long run uses a 4 ms period
/// so the unit lane stays quick; the frames stay 20 ms of audio, and a one-second run at the real period checks the
/// production cadence.</summary>
public sealed class SenderCadenceTests
{
    private const int Frame = 320;

    private readonly ITestOutputHelper _output;

    public SenderCadenceTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task OverMoreThan1000FramesTheSenderKeepsItsPeriodAndAllocatesNothing()
    {
        const long period = 4_000_000L;
        const int frames = 1_200;
        await using HostRig rig = HostRig.Start(options => options with { SenderPeriodNs = period, SenderWarmUpTicks = 100 });
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.EndlessTurn = 1;
        long startNs = MonotonicClock.NowNs();
        Assert.True(gateway.WaitUntil(() => AudioFrames(gateway) >= frames, 30_000), $"only {AudioFrames(gateway)} frames arrived.");
        session.EndlessTurn = 0;
        long elapsedNs = MonotonicClock.NowNs() - startNs;

        LinkConnection connection = rig.Server.Current!;
        List<FakeGateway.RecordedFrame> audio = gateway.FramesOf(LinkMessageType.OutboundAudio);
        LatencyHistogram.Summary late = connection.Lateness.Snapshot();
        _output.WriteLine($"{audio.Count} frames in {elapsedNs / 1e6:F0} ms at a {period / 1e6} ms period; lateness p50={late.P50Us}us "
            + $"p99={late.P99Us}us max={late.MaxUs}us; catch-up {connection.CatchUpFrames}, resyncs {connection.Resyncs}; "
            + $"audio path allocated {connection.AudioPathAllocatedBytes} B");

        Assert.All(audio, f => Assert.Equal(Frame, f.Pcm.Length));
        Assert.All(audio, f => Assert.Equal(1u, f.TurnId));
        Assert.Equal(0, connection.AudioPathAllocatedBytes);
        // Three frames on the first tick (20 ms plus the 40 ms prebuffer), then one per period, catch-up frames included.
        double expected = elapsedNs / (double)period + 2;
        Assert.InRange(audio.Count, frames, expected + 4);
    }

    [Fact]
    public async Task AtTheRealPeriodABurstStartsWithThePrebufferThenKeeps20MsPerFrame()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.QueueReply(turnId: 1, samples: 16_000);
        Assert.True(gateway.WaitUntil(() => Samples(gateway) == 16_000, 5_000));

        List<FakeGateway.RecordedFrame> audio = gateway.FramesOf(LinkMessageType.OutboundAudio);
        Assert.Equal(50, audio.Count);
        // The first three frames go out together, on the first tick.
        Assert.InRange((audio[2].ReceivedNs - audio[0].ReceivedNs) / 1e6, 0, 5);
        // The other 47 follow one per 20 ms tick: 47 periods, give or take scheduler noise.
        double paced = (audio[^1].ReceivedNs - audio[2].ReceivedNs) / 1e6;
        _output.WriteLine($"47 paced frames over {paced:F1} ms");
        Assert.InRange(paced, 47 * 20 - 40, 47 * 20 + 40);
    }

    private static int AudioFrames(FakeGateway gateway) => gateway.FramesOf(LinkMessageType.OutboundAudio).Count;

    private static int Samples(FakeGateway gateway) => gateway.FramesOf(LinkMessageType.OutboundAudio).Sum(f => f.Pcm.Length);
}
