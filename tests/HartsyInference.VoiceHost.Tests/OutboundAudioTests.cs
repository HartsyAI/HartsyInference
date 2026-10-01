using HartsyInference.PhoneLink;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Tests.Support;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>Reply audio from the session to the gateway: frames carry the turn that produced them and at most 20 ms;
/// a barge-in becomes <c>Flush(turnId)</c> and nothing of that turn or an older one follows it on the link, even when the
/// session still hands such audio over; <c>OutboundEnd</c> follows a turn's last frame and precedes the next turn's
/// first, never for a flushed turn; a finished turn's latency goes out as <c>TurnLatency</c>.</summary>
public sealed class OutboundAudioTests
{
    private const int Frame = 320;

    [Fact]
    public async Task ReplyAudioGoesOutTaggedWithItsTurnInFramesOfAtMost20Ms()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.QueueReply(turnId: 1, samples: 10 * Frame + 100, level: 0.5f);

        Assert.True(gateway.WaitUntil(() => Samples(gateway, 1) == 10 * Frame + 100), $"got {Samples(gateway, 1)} samples.");
        List<FakeGateway.RecordedFrame> audio = gateway.FramesOf(LinkMessageType.OutboundAudio);
        Assert.All(audio, f => Assert.Equal(1u, f.TurnId));
        Assert.All(audio, f => Assert.InRange(f.Pcm.Length, 1, Frame));
        Assert.All(audio.SelectMany(f => f.Pcm), sample => Assert.Equal((short)(0.5f * 32767f), sample));
    }

    [Fact]
    public async Task ABargeInFlushCarriesTheTurnAndNothingOfThatTurnFollowsIt()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.QueueReply(turnId: 1, samples: 4 * Frame);
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 1);
        // A second of turn 2 that the session keeps handing over after the barge-in: the worst case for the host's own rule.
        session.QueueReply(turnId: 2, samples: 16_000);
        Assert.True(gateway.WaitUntil(() => Samples(gateway, 2) >= 8 * Frame));

        session.Raise(VoiceAgentEventKind.BargeIn, turnId: 2);
        FakeGateway.RecordedFrame flush = gateway.WaitFor(LinkMessageType.Flush)[0];
        Assert.Equal(2u, flush.TurnId);
        Assert.Equal(1u, flush.Header.CallId);
        session.QueueReply(turnId: 3, samples: 6 * Frame);
        Assert.True(gateway.WaitUntil(() => Samples(gateway, 3) == 6 * Frame), "the next turn's reply did not follow the flush.");

        List<FakeGateway.RecordedFrame> afterFlush = gateway.AllFrames().Where(f => f.Header.Sequence > flush.Header.Sequence).ToList();
        Assert.DoesNotContain(afterFlush, f => f.Header.Type is LinkMessageType.OutboundAudio or LinkMessageType.OutboundEnd && f.TurnId <= 2);
        Assert.True(rig.Call(1)!.StaleSamples > 0, "the session never handed the host stale audio; the test proved nothing.");
    }

    [Fact]
    public async Task OutboundEndFollowsATurnsLastFrameAndPrecedesTheNextTurn()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.QueueReply(turnId: 1, samples: 3 * Frame + 50);
        session.QueueReply(turnId: 2, samples: 2 * Frame);
        Assert.True(gateway.WaitUntil(() => Samples(gateway, 2) == 2 * Frame));
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 2);
        gateway.WaitFor(LinkMessageType.OutboundEnd, 2);

        List<(LinkMessageType Type, uint Turn)> sequence = gateway.AllFrames()
            .Where(f => f.Header.Type is LinkMessageType.OutboundAudio or LinkMessageType.OutboundEnd)
            .Select(f => (f.Header.Type, f.TurnId)).ToList();
        int endOfOne = sequence.IndexOf((LinkMessageType.OutboundEnd, 1u));
        Assert.True(endOfOne > 0);
        Assert.All(sequence.Take(endOfOne), item => Assert.Equal((LinkMessageType.OutboundAudio, 1u), item));
        Assert.All(sequence.Skip(endOfOne + 1).SkipLast(1), item => Assert.Equal((LinkMessageType.OutboundAudio, 2u), item));
        Assert.Equal((LinkMessageType.OutboundEnd, 2u), sequence[^1]);
    }

    [Fact]
    public async Task AFlushedTurnGetsNoOutboundEnd()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.QueueReply(turnId: 1, samples: 16_000);
        Assert.True(gateway.WaitUntil(() => Samples(gateway, 1) >= 4 * Frame));
        session.Raise(VoiceAgentEventKind.BargeIn, turnId: 1);
        gateway.WaitFor(LinkMessageType.Flush);
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 1, metrics: new VoiceTurnMetrics { TurnId = 1, Interrupted = true });
        await Task.Delay(200);

        Assert.Empty(gateway.FramesOf(LinkMessageType.OutboundEnd));
    }

    [Fact]
    public async Task AFinishedTurnsLatencyGoesOutWhenItPlayedAudio()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 4, metrics: new VoiceTurnMetrics { TurnId = 4, SttMs = 120.4, LlmTtftMs = 88.6, TotalMs = 912.6 });
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 5, metrics: new VoiceTurnMetrics { TurnId = 5, SttMs = 99 });

        LinkEventMessage latency = gateway.WaitFor(LinkMessageType.Event)[0].Event;
        Assert.Equal(LinkEventKind.TurnLatency, latency.Kind);
        Assert.Equal(4u, latency.TurnId);
        Assert.Equal(120, latency.Latency!.SttMs);
        Assert.Equal(89, latency.Latency.LlmFirstTokenMs);
        Assert.Equal(913, latency.Latency.TotalMs);
        Assert.Null(latency.Latency.TtsFirstChunkMs);
        await Task.Delay(100);
        Assert.Single(gateway.FramesOf(LinkMessageType.Event));
    }

    private static int Samples(FakeGateway gateway, uint turn) =>
        gateway.FramesOf(LinkMessageType.OutboundAudio).Where(f => f.TurnId == turn).Sum(f => f.Pcm.Length);
}
