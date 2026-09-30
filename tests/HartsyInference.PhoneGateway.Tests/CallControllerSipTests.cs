using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Metrics;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneGateway.Tests.Support;
using HartsyInference.PhoneLink;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Call-control failures that are silent without a test: a call whose RTP clock died staying up with no audio
/// until the far end times it out, and a caller's INVITE retransmissions inflating the rejection counters. Real
/// sipsorcery peers on 127.0.0.1 and a fake host on a temporary socket; no audio timing is asserted, so these run in
/// the unit lane. The tick fault is injected the way a real one arrives: an exception out of the frame subscriber
/// (<c>SendAudio</c> in production), or out of the tick thread's start.</summary>
public sealed class CallControllerSipTests
{
    private const int WaitMs = GatewayLoopback.WaitMs;

    private readonly ITestOutputHelper _output;
    public CallControllerSipTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TickFault_OnAnActiveInboundCall_SendsByeCallEndAndCountsOnce()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        ClockedAudioSource? source = null;
        gateway.Controller.SourceCreated = s => source = s;
        using Softphone phone = new();
        Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        Assert.True(gateway.WaitUntil(() => phone.RtpFramesReceived >= 5, WaitMs), "the RTP clock never ran");

        source!.OnAudioSourceEncodedSample += (_, _) => throw new InvalidOperationException("injected tick fault");

        Assert.True(phone.HungUp.Wait(WaitMs), "the caller never got a BYE");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.CallEnd).Count == 1, WaitMs));
        Assert.Equal(LinkCallEndReason.Failed, gateway.Host.FramesOf(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
        Assert.True(source.Faulted);
        await Task.Delay(200);
        Assert.Single(gateway.Host.FramesOf(LinkMessageType.CallEnd));
        Assert.Equal(1, gateway.Metrics.CallsMediaFault);
        Assert.Contains("hartsy_phone_calls_media_fault_total 1\n", PrometheusTextWriter.Render(gateway.Metrics), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TickFault_OnAnActiveOutboundCall_SendsBye()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        ClockedAudioSource? source = null;
        gateway.Controller.SourceCreated = s => source = s;
        using Softphone phone = new();
        CallPlacementResult placed = await gateway.Controller.PlaceCallAsync($"sip:phone@127.0.0.1:{phone.Port}");
        Assert.True(placed.Placed, placed.Message);
        Assert.True(gateway.WaitUntil(() => phone.RtpFramesReceived >= 5, WaitMs), "the RTP clock never ran");

        source!.OnAudioSourceEncodedSample += (_, _) => throw new InvalidOperationException("injected tick fault");

        Assert.True(phone.HungUp.Wait(WaitMs), "the callee never got a BYE");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.CallEnd).Count == 1, WaitMs));
        Assert.Equal(LinkCallEndReason.Failed, gateway.Host.FramesOf(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
        Assert.Equal(1, gateway.Metrics.CallsMediaFault);
    }

    [Fact]
    public async Task TickFault_BeforeTheCallIsAnnounced_EndsItWithByeAndNeverTellsTheHost()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        gateway.Controller.SourceCreated = s => s.InjectedStartFault = new InvalidOperationException("injected start fault");
        using Softphone phone = new();

        // The 200 OK has gone out by the time the fault is seen, so the call is answered and then ended with a BYE.
        Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
        Assert.True(phone.HungUp.Wait(WaitMs), "the caller never got a BYE");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
        await Task.Delay(200);
        Assert.Empty(gateway.Host.FramesOf(LinkMessageType.CallStart));
        Assert.Empty(gateway.Host.FramesOf(LinkMessageType.CallEnd));
        Assert.Equal(1, gateway.Metrics.CallsMediaFault);
        Assert.Equal(0, gateway.Metrics.CallsInbound);
        Assert.Equal(0, gateway.Metrics.CallsActive);
    }

    [Fact]
    public async Task BusyInvite_RetransmittedThreeTimes_IsAnsweredEachTimeAndCountedOnce()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        using Softphone phone = new();
        Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        using RawSipClient caller = new();

        await caller.SendInviteCopiesAsync(gateway.Port, copies: 3, gapMs: 100);

        Assert.True(gateway.WaitUntil(() => caller.Statuses.Length >= 3, WaitMs), $"responses: {string.Join(",", caller.Statuses)}");
        _output.WriteLine($"responses to 3 copies: {string.Join(",", caller.Statuses)}");
        Assert.All(caller.Statuses, status => Assert.Equal(486, status));
        Assert.Equal(1, gateway.Metrics.CallsRejectedBusy);
        Assert.Contains("hartsy_phone_calls_rejected_total{reason=\"busy\"} 1\n", PrometheusTextWriter.Render(gateway.Metrics), StringComparison.Ordinal);
        Assert.Equal(CallState.Active, gateway.Controller.State);
        Assert.Single(gateway.Host.FramesOf(LinkMessageType.CallStart));
        phone.Agent.Hangup();
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
    }

    [Fact]
    public async Task DeclinedInvite_RetransmittedThreeTimes_IsAnsweredEachTimeAndCountedOnce()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions { InboundPolicy = InboundPolicy.Reject });
        using RawSipClient caller = new();

        await caller.SendInviteCopiesAsync(gateway.Port, copies: 3, gapMs: 100);

        Assert.True(gateway.WaitUntil(() => caller.Statuses.Length >= 3, WaitMs), $"responses: {string.Join(",", caller.Statuses)}");
        _output.WriteLine($"responses to 3 copies: {string.Join(",", caller.Statuses)}");
        Assert.All(caller.Statuses, status => Assert.Equal(603, status));
        Assert.Equal(1, gateway.Metrics.CallsDeclined);
        Assert.Equal(CallState.Idle, gateway.Controller.State);
        Assert.Empty(gateway.Host.FramesOf(LinkMessageType.CallStart));
    }

    [Fact]
    public async Task HostDownInvite_RetransmittedThreeTimes_IsAnsweredEachTimeAndCountedOnce()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        gateway.Host.Shutdown();
        Assert.True(gateway.WaitUntil(() => !gateway.Link.IsConnected, WaitMs));
        using RawSipClient caller = new();

        await caller.SendInviteCopiesAsync(gateway.Port, copies: 3, gapMs: 100);

        Assert.True(gateway.WaitUntil(() => caller.Statuses.Length >= 3, WaitMs), $"responses: {string.Join(",", caller.Statuses)}");
        _output.WriteLine($"responses to 3 copies: {string.Join(",", caller.Statuses)}");
        Assert.All(caller.Statuses, status => Assert.Equal(503, status));
        Assert.Equal(1, gateway.Metrics.CallsRejectedHostDown);
        Assert.Equal(CallState.Idle, gateway.Controller.State);
    }
}
