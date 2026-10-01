using System.Text.Json;
using HartsyInference.Core.Runtime;
using HartsyInference.Engine.Requests;
using HartsyInference.PhoneLink;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Link;
using HartsyInference.VoiceHost.Tests.Support;
using HartsyInference.VoiceHost.Tools;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>The tools a call's model is offered: a telephony tool is a <c>ToolRequest</c> to the gateway answered by its
/// <c>ToolResult</c>, or a <c>Failed</c> result after the tool timeout or the end of the call; <c>hangup</c> answers at once
/// and goes to the gateway only once the turn that asked for it has ended and its goodbye is on the link, plus the
/// downstream margin, within a cap that a goodbye which never drains cannot outlast; a barge-in on the goodbye cuts it
/// short and the call still ends; <c>get_time</c> is answered on the host; only the enabled tools are offered.</summary>
public sealed class TelephonyToolTests
{
    private const int Frame = 320;

    [Fact]
    public async Task ATelephonyToolIsARoundTripToTheGateway()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        Task<string> invoked = Invoke(session, VoiceHostTools.SendDtmf, """{"digits":"12#"}""");

        FakeGateway.RecordedFrame request = gateway.WaitFor(LinkMessageType.ToolRequest)[0];
        ToolRequestMessage message = request.ToolRequest(out uint requestId);
        Assert.Equal(VoiceHostTools.SendDtmf, message.Name);
        Assert.Equal("12#", request.ToolArgument("digits"));
        Assert.False(invoked.IsCompleted);
        gateway.SendToolResult(1, requestId, LinkToolStatus.Ok);

        Assert.Equal("""{"status":"Ok"}""", await invoked.WaitAsync(TimeSpan.FromSeconds(5)));
    }

    [Fact]
    public async Task AFailedToolCarriesTheGatewaysReasonToTheModel()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        Task<string> invoked = Invoke(session, VoiceHostTools.Transfer, """{"target":"+15550199"}""");
        gateway.WaitFor(LinkMessageType.ToolRequest)[0].ToolRequest(out uint requestId);
        gateway.SendToolResult(1, requestId, LinkToolStatus.Failed, "transfer target: not in sip.destinationPrefixes");

        using JsonDocument result = JsonDocument.Parse(await invoked.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal("Failed", result.RootElement.GetProperty("status").GetString());
        Assert.Contains("destinationPrefixes", result.RootElement.GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AToolTheGatewayNeverAnswersFailsAfterTheTimeout()
    {
        await using HostRig rig = HostRig.Start(options => options with { ToolTimeoutMs = 300 });
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        long started = Environment.TickCount64;
        string answer = await Invoke(session, VoiceHostTools.Hold).WaitAsync(TimeSpan.FromSeconds(5));
        long waited = Environment.TickCount64 - started;

        using JsonDocument result = JsonDocument.Parse(answer);
        Assert.Equal("Failed", result.RootElement.GetProperty("status").GetString());
        Assert.Contains("300 ms", result.RootElement.GetProperty("message").GetString(), StringComparison.Ordinal);
        Assert.InRange(waited, 250, 4000);
        gateway.WaitFor(LinkMessageType.ToolRequest)[0].ToolRequest(out uint requestId);
        gateway.SendToolResult(1, requestId, LinkToolStatus.Ok);
        await Task.Delay(100);
        Assert.False(gateway.IsClosed, "a late ToolResult broke the link.");
    }

    [Fact]
    public async Task APendingToolFailsWhenTheCallEnds()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        Task<string> invoked = Invoke(session, VoiceHostTools.Unhold);
        gateway.WaitFor(LinkMessageType.ToolRequest);
        gateway.SendCallEnd(1, LinkCallEndReason.RemoteHangup);

        using JsonDocument result = JsonDocument.Parse(await invoked.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal("Failed", result.RootElement.GetProperty("status").GetString());
    }

    [Fact]
    public async Task HangupGoesOutOnlyAfterTheGoodbyesLastFrameAndTheMargin()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        string answer = await Invoke(session, VoiceHostTools.Hangup).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Contains("\"Ok\"", answer, StringComparison.Ordinal);
        session.QueueReply(turnId: 3, samples: 25 * Frame);
        await Task.Delay(100);
        Assert.Empty(gateway.FramesOf(LinkMessageType.ToolRequest));

        // The turn ends with most of its half-second goodbye still queued, as one whose synthesis failed midway does.
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 3, metrics: new VoiceTurnMetrics { TurnId = 3 });
        FakeGateway.RecordedFrame request = gateway.WaitFor(LinkMessageType.ToolRequest)[0];
        Assert.Equal(VoiceHostTools.Hangup, request.ToolRequest(out uint requestId).Name);

        List<FakeGateway.RecordedFrame> goodbye = gateway.FramesOf(LinkMessageType.OutboundAudio).Where(f => f.TurnId == 3).ToList();
        Assert.Equal(25 * Frame, goodbye.Sum(f => f.Pcm.Length));
        FakeGateway.RecordedFrame end = Assert.Single(gateway.FramesOf(LinkMessageType.OutboundEnd));
        Assert.Equal(3u, end.TurnId);
        Assert.True(goodbye[^1].Header.Sequence < end.Header.Sequence && end.Header.Sequence < request.Header.Sequence,
            "the hangup request went out before the goodbye's last frame and its OutboundEnd.");
        double gapMs = (request.ReceivedNs - end.ReceivedNs) / 1e6;
        int marginMs = new PhoneLinkServerOptions { SocketPath = "unused" }.HangupMarginMs;
        Assert.True(gapMs >= marginMs - 30, $"the request followed the goodbye's OutboundEnd by {gapMs:F0} ms, not the {marginMs} ms margin.");

        gateway.SendToolResult(1, requestId, LinkToolStatus.Ok);
        FakeGateway.RecordedFrame callEnd = gateway.WaitFor(LinkMessageType.CallEnd)[0];
        Assert.Equal(LinkCallEndReason.Completed, callEnd.AsFrame().ReadCallEnd());
        Assert.True(HostRig.Wait(() => session.Disposed && rig.Call(1) is null));
    }

    [Fact]
    public async Task AHangupWithNoGoodbyeAudioWaitsOnlyTheMargin()
    {
        await using HostRig rig = HostRig.Start(options => options with { HangupSlackMs = 5_000 });
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        await Invoke(session, VoiceHostTools.Hangup).WaitAsync(TimeSpan.FromSeconds(5));
        long ended = MonotonicClock.NowNs();
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 2, metrics: new VoiceTurnMetrics { TurnId = 2 });

        FakeGateway.RecordedFrame request = gateway.WaitFor(LinkMessageType.ToolRequest)[0];
        double waitedMs = (request.ReceivedNs - ended) / 1e6;
        Assert.True(waitedMs < 2_000, $"a turn with no audio waited {waitedMs:F0} ms, as if for the 5 s cap.");
        Assert.Empty(gateway.FramesOf(LinkMessageType.OutboundEnd));
    }

    [Theory]
    [InlineData(8_000, 10_000, 1_500)]
    [InlineData(960_000, 1_200, 1_200)]
    public async Task TheCapSendsTheHangupWhenTheGoodbyeNeverDrains(int queuedSamples, int maxWaitMs, int capMs)
    {
        // Half a second still queued: the cap is that plus the 1 s slack. A minute queued: the absolute maximum wins
        // (shortened here from its 10 s default).
        await using HostRig rig = HostRig.Start(options => options with { HangupMaxWaitMs = maxWaitMs });
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        await Invoke(session, VoiceHostTools.Hangup).WaitAsync(TimeSpan.FromSeconds(5));
        session.EndlessQueuedSamples = queuedSamples;
        session.EndlessTurn = 3;
        Assert.True(gateway.WaitUntil(() => Samples(gateway, 3) >= 5 * Frame));
        long ended = MonotonicClock.NowNs();
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 3, metrics: new VoiceTurnMetrics { TurnId = 3 });

        FakeGateway.RecordedFrame request = gateway.WaitFor(LinkMessageType.ToolRequest, timeoutMs: capMs + 5_000)[0];
        Assert.Equal(VoiceHostTools.Hangup, request.ToolRequest(out uint requestId).Name);
        double waitedMs = (request.ReceivedNs - ended) / 1e6;
        Assert.InRange(waitedMs, capMs - 20, capMs + 1_000);
        Assert.Empty(gateway.FramesOf(LinkMessageType.OutboundEnd));
        Assert.Contains(gateway.FramesOf(LinkMessageType.OutboundAudio), f => f.TurnId == 3 && f.ReceivedNs > request.ReceivedNs - 100_000_000L);

        gateway.SendToolResult(1, requestId, LinkToolStatus.Ok);
        Assert.Equal(LinkCallEndReason.Completed, gateway.WaitFor(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ABargeInOnTheGoodbyeStillHangsUpPromptly(bool turnEndsFirst)
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        await Invoke(session, VoiceHostTools.Hangup).WaitAsync(TimeSpan.FromSeconds(5));
        session.QueueReply(turnId: 3, samples: 5 * 16_000);
        if (turnEndsFirst)
        {
            session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 3, metrics: new VoiceTurnMetrics { TurnId = 3 });
        }
        Assert.True(gateway.WaitUntil(() => Samples(gateway, 3) >= 4 * Frame));
        Assert.Empty(gateway.FramesOf(LinkMessageType.ToolRequest));

        long bargedIn = MonotonicClock.NowNs();
        session.Raise(VoiceAgentEventKind.BargeIn, turnId: 3);
        FakeGateway.RecordedFrame flush = gateway.WaitFor(LinkMessageType.Flush)[0];
        if (!turnEndsFirst)
        {
            // The real session's order: the turn ends once its flush is applied, marked interrupted.
            session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 3, metrics: new VoiceTurnMetrics { TurnId = 3, Interrupted = true });
        }
        FakeGateway.RecordedFrame request = gateway.WaitFor(LinkMessageType.ToolRequest)[0];
        Assert.Equal(VoiceHostTools.Hangup, request.ToolRequest(out uint requestId).Name);
        double waitedMs = (request.ReceivedNs - bargedIn) / 1e6;
        Assert.True(waitedMs < 1_000, $"the hangup went out {waitedMs:F0} ms after the barge-in, with 5 s of goodbye queued.");
        Assert.True(flush.Header.Sequence < request.Header.Sequence);
        Assert.DoesNotContain(gateway.AllFrames(), f => f.Header.Sequence > flush.Header.Sequence
            && f.Header.Type is LinkMessageType.OutboundAudio or LinkMessageType.OutboundEnd);

        gateway.SendToolResult(1, requestId, LinkToolStatus.Ok);
        Assert.Equal(LinkCallEndReason.Completed, gateway.WaitFor(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
        Assert.True(HostRig.Wait(() => session.Disposed && rig.Call(1) is null));
    }

    [Fact]
    public async Task GetTimeIsAnsweredOnTheHost()
    {
        FixedClock clock = new(new DateTimeOffset(2026, 9, 30, 23, 59, 0, TimeSpan.Zero));
        await using HostRig rig = HostRig.Start(options => options with { Clock = clock });
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();

        using JsonDocument reading = JsonDocument.Parse(await Invoke(session, VoiceHostTools.GetTime).WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal("2026-09-30T23:59:00Z", reading.RootElement.GetProperty("utc").GetString());
        Assert.Equal("2026-09-30T23:59:00+00:00", reading.RootElement.GetProperty("local").GetString());
        Assert.Equal("Wednesday", reading.RootElement.GetProperty("dayOfWeek").GetString());
        Assert.Empty(gateway.FramesOf(LinkMessageType.ToolRequest));
    }

    [Fact]
    public async Task OnlyTheEnabledToolsAreOffered()
    {
        await using HostRig rig = HostRig.Start(options => options with { Tools = [VoiceHostTools.GetTime, VoiceHostTools.Hangup] });
        (_, FakeCallSession session) = await rig.StartCallAsync();

        Assert.Equal([VoiceHostTools.Hangup, VoiceHostTools.GetTime], session.Tools.Names);
        foreach (ToolDefinition definition in session.Tools.Definitions)
        {
            using JsonDocument schema = JsonDocument.Parse(definition.JsonSchema);
            Assert.Equal("object", schema.RootElement.GetProperty("type").GetString());
            Assert.False(string.IsNullOrWhiteSpace(definition.Description));
        }
    }

    private static Task<string> Invoke(FakeCallSession session, string tool, string arguments = "{}") =>
        session.Tools.InvokeAsync(new NativeToolCall { Id = "call-1", Name = tool, Arguments = arguments });

    private static int Samples(FakeGateway gateway, uint turn) =>
        gateway.FramesOf(LinkMessageType.OutboundAudio).Where(f => f.TurnId == turn).Sum(f => f.Pcm.Length);

    /// <summary>A clock stopped at one instant, in UTC.</summary>
    private sealed class FixedClock(DateTimeOffset now) : TimeProvider
    {
        public override DateTimeOffset GetUtcNow() => now;

        public override TimeZoneInfo LocalTimeZone => TimeZoneInfo.Utc;
    }
}
