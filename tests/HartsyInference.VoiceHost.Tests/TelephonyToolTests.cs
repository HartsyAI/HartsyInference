using System.Text.Json;
using HartsyInference.Engine.Requests;
using HartsyInference.PhoneLink;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Tests.Support;
using HartsyInference.VoiceHost.Tools;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>The tools a call's model is offered: a telephony tool is a <c>ToolRequest</c> to the gateway answered by its
/// <c>ToolResult</c>, or a <c>Failed</c> result after the tool timeout or the end of the call; <c>hangup</c> answers at once
/// and goes to the gateway only after the reply that asked for it has played, and not at all when the caller barged
/// into that goodbye; <c>get_time</c> is answered on the host; only the enabled tools are offered.</summary>
public sealed class TelephonyToolTests
{
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
    public async Task HangupWaitsForTheReplyToPlayThenEndsTheCall()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        string answer = await Invoke(session, VoiceHostTools.Hangup).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Contains("\"Ok\"", answer, StringComparison.Ordinal);
        await Task.Delay(100);
        Assert.Empty(gateway.FramesOf(LinkMessageType.ToolRequest));

        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 3, metrics: new VoiceTurnMetrics { TurnId = 3 });
        FakeGateway.RecordedFrame request = gateway.WaitFor(LinkMessageType.ToolRequest)[0];
        Assert.Equal(VoiceHostTools.Hangup, request.ToolRequest(out uint requestId).Name);
        gateway.SendToolResult(1, requestId, LinkToolStatus.Ok);

        FakeGateway.RecordedFrame end = gateway.WaitFor(LinkMessageType.CallEnd)[0];
        Assert.Equal(LinkCallEndReason.Completed, end.AsFrame().ReadCallEnd());
        Assert.True(HostRig.Wait(() => session.Disposed && rig.Call(1) is null));
    }

    [Fact]
    public async Task ACallerWhoBargesIntoTheGoodbyeKeepsTheCall()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        await Invoke(session, VoiceHostTools.Hangup).WaitAsync(TimeSpan.FromSeconds(5));
        session.Raise(VoiceAgentEventKind.TurnCompleted, turnId: 2, metrics: new VoiceTurnMetrics { TurnId = 2, Interrupted = true });
        await Task.Delay(200);

        Assert.Empty(gateway.FramesOf(LinkMessageType.ToolRequest));
        Assert.Empty(gateway.FramesOf(LinkMessageType.CallEnd));
        Assert.NotNull(rig.Call(1));
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

    /// <summary>A clock stopped at one instant, in UTC.</summary>
    private sealed class FixedClock(DateTimeOffset now) : TimeProvider
    {
        public override DateTimeOffset GetUtcNow() => now;

        public override TimeZoneInfo LocalTimeZone => TimeZoneInfo.Utc;
    }
}
