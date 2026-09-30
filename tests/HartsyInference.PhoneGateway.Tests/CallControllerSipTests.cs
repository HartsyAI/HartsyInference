using System.Net;
using System.Net.Sockets;
using HartsyInference.Core.Logging;
using HartsyInference.PhoneGateway.Admin;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Metrics;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneGateway.Tests.Support;
using HartsyInference.PhoneLink;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Call-control failures that are silent without a test: a call whose RTP clock or inbound pump died staying up
/// half-dead until the far end times it out, and a caller's INVITE retransmissions inflating the rejection counters. Real
/// sipsorcery peers on 127.0.0.1 and a fake host on a temporary socket; no audio timing is asserted, so these run in
/// the unit lane. The tick fault is injected the way a real one arrives: an exception out of the frame subscriber
/// (<c>SendAudio</c> in production), or out of the tick thread's start.</summary>
public sealed class CallControllerSipTests
{
    private const int WaitMs = GatewayLoopback.WaitMs;
    private const int AdminStartAttempts = 5;

    private readonly ITestOutputHelper _output;
    public CallControllerSipTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TickFault_OnAnActiveInboundCall_SendsByeCallEndAndCountsOnce()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        ClockedAudioSource? source = null;
        gateway.Controller.MediaCreated = (s, _) => source = s;
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
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions { AllowAnyDestination = true });
        ClockedAudioSource? source = null;
        gateway.Controller.MediaCreated = (s, _) => source = s;
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
    public async Task PumpFault_OnAnActiveCall_TakesTheSameTeardown()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        InboundAudioPath? pump = null;
        gateway.Controller.MediaCreated = (_, p) => pump = p;
        using Softphone phone = new();
        Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.AudioFramesReceived >= 5, WaitMs), "the pump never ran");

        pump!.InjectedFault = new InvalidOperationException("injected pump fault");

        Assert.True(phone.HungUp.Wait(WaitMs), "the caller never got a BYE");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.CallEnd).Count == 1, WaitMs));
        Assert.Equal(LinkCallEndReason.Failed, gateway.Host.FramesOf(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
        Assert.True(pump.Faulted);
        await Task.Delay(200);
        Assert.Single(gateway.Host.FramesOf(LinkMessageType.CallEnd));
        Assert.Equal(1, gateway.Metrics.CallsMediaFault);
        Assert.Contains("hartsy_phone_calls_media_fault_total 1\n", PrometheusTextWriter.Render(gateway.Metrics), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TickFault_BeforeTheCallIsAnnounced_EndsItWithByeAndNeverTellsTheHost()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        gateway.Controller.MediaCreated = (s, _) => s.InjectedStartFault = new InvalidOperationException("injected start fault");
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
    public async Task MediaSetupFailure_OnAnOutboundCall_ReturnsFailedAndLeavesTheGatewayUsable()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions { AllowAnyDestination = true });
        using Softphone phone = new();
        gateway.Controller.MediaCreated = (_, _) => throw new InvalidOperationException("injected setup failure");

        CallPlacementResult failed = await gateway.Controller.PlaceCallAsync($"sip:phone@127.0.0.1:{phone.Port}");

        Assert.Equal(CallPlacementStatus.Failed, failed.Status);
        Assert.Equal(CallState.Idle, gateway.Controller.State);
        gateway.Controller.MediaCreated = null;
        CallPlacementResult placed = await gateway.Controller.PlaceCallAsync($"sip:phone@127.0.0.1:{phone.Port}");
        Assert.True(placed.Placed, placed.Message);
        gateway.Controller.HangUp(LinkCallEndReason.Completed);
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
    }

    [Fact]
    public async Task MediaSetupFailure_OnAnInboundCall_Answers500AndLeavesTheGatewayUsable()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        gateway.Controller.MediaCreated = (_, _) => throw new InvalidOperationException("injected setup failure");
        using Softphone first = new();

        Assert.False(await first.CallAsync(gateway.Port));

        _output.WriteLine($"failed setup: {first.LastFailureStatus} {first.LastFailure}");
        Assert.Equal(500, first.LastFailureStatus);
        Assert.Equal(CallState.Idle, gateway.Controller.State);
        gateway.Controller.MediaCreated = null;
        using Softphone second = new();
        Assert.True(await second.CallAsync(gateway.Port), $"call failed: {second.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        second.Agent.Hangup();
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
    }

    /// <summary>The default dial plan is closed: with no <c>destinationPrefixes</c> and <c>allowAnyDestination</c> off,
    /// neither the host's <c>transfer</c> tool nor an operator's <c>POST /calls</c> reaches the SIP stack, and both say
    /// which setting to change.</summary>
    [Fact]
    public async Task NoDialPlan_RefusesATransferAndPostCallsWith403()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions());
        using Softphone phone = new();
        const string token = "test-admin-token";
        using AdminEndpoint admin = StartAdmin(gateway, token);
        using HttpClient http = new();
        using HttpRequestMessage post = new(HttpMethod.Post, admin.Prefix + "calls")
        {
            Content = new StringContent($"{{\"destination\":\"sip:phone@127.0.0.1:{phone.Port}\"}}", System.Text.Encoding.UTF8, "application/json"),
        };
        post.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

        using HttpResponseMessage answer = await http.SendAsync(post);

        string body = await answer.Content.ReadAsStringAsync();
        _output.WriteLine($"POST /calls: {(int)answer.StatusCode} {body}");
        Assert.Equal(System.Net.HttpStatusCode.Forbidden, answer.StatusCode);
        Assert.Contains("application/json", answer.Content.Headers.ContentType?.MediaType, StringComparison.Ordinal);
        Assert.Contains("sip.destinationPrefixes", body, StringComparison.Ordinal);
        Assert.Contains("sip.allowAnyDestination", body, StringComparison.Ordinal);
        Assert.Equal(CallState.Idle, gateway.Controller.State);
        Assert.Equal(0, phone.IncomingCalls);
        Assert.Equal(0, gateway.Metrics.CallsOutbound);

        Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        uint callId = gateway.Controller.Current!.CallId;
        using (System.Text.Json.JsonDocument arguments = System.Text.Json.JsonDocument.Parse("{\"target\":\"sip:+19005551234@premium.example\"}"))
        {
            gateway.Host.SendToolRequest(callId, 11, new ToolRequestMessage { Name = "transfer", Arguments = arguments.RootElement });
        }
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.ToolResult).Count == 1, WaitMs));
        ToolResultMessage result = gateway.Host.FramesOf(LinkMessageType.ToolResult)[0].AsFrame().ReadToolResult(out uint requestId);
        Assert.Equal(11u, requestId);
        Assert.Equal(LinkToolStatus.Failed, result.Status);
        Assert.Contains("sip.destinationPrefixes", result.Message, StringComparison.Ordinal);
        Assert.Contains("sip.allowAnyDestination", result.Message, StringComparison.Ordinal);
        Assert.Equal(CallState.Active, gateway.Controller.State);
        phone.Agent.Hangup();
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
    }

    /// <summary><c>allowAnyDestination</c> opens the dial plan for a LAN or development, and says so once, at start-up;
    /// placing a call does not repeat it.</summary>
    [Fact]
    public async Task AllowAnyDestination_DialsAnyDestinationAndWarnsOnceAtStart()
    {
        List<string> warnings = [];
        Logs.SetLogger((level, message) =>
        {
            if (level == LogLevel.Warning && message.Contains("toll-fraud protection is off", StringComparison.Ordinal))
            {
                lock (warnings)
                {
                    warnings.Add(message);
                }
            }
        });
        try
        {
            using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions { AllowAnyDestination = true });
            Assert.Single(WarningsSoFar());
            Assert.Contains("sip.allowAnyDestination", WarningsSoFar()[0], StringComparison.Ordinal);

            using Softphone phone = new();
            CallPlacementResult placed = await gateway.Controller.PlaceCallAsync($"sip:phone@127.0.0.1:{phone.Port}");
            Assert.True(placed.Placed, placed.Message);
            gateway.Controller.HangUp(LinkCallEndReason.Completed);
            Assert.True(phone.HungUp.Wait(WaitMs), "the callee never got a BYE");
            Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));

            Assert.Single(WarningsSoFar());
        }
        finally
        {
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }

        List<string> WarningsSoFar()
        {
            lock (warnings)
            {
                return [.. warnings];
            }
        }
    }

    [Fact]
    public async Task DestinationPrefixes_RefuseAnOutboundCallAndATransfer()
    {
        using GatewayLoopback gateway = GatewayLoopback.Start(new CallControllerOptions { DestinationPrefixes = ["+1555"] });
        using Softphone phone = new();

        CallPlacementResult refused = await gateway.Controller.PlaceCallAsync($"sip:phone@127.0.0.1:{phone.Port}");
        Assert.Equal(CallPlacementStatus.NotAllowed, refused.Status);
        Assert.Equal(CallState.Idle, gateway.Controller.State);

        Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        uint callId = gateway.Controller.Current!.CallId;
        using (System.Text.Json.JsonDocument arguments = System.Text.Json.JsonDocument.Parse("{\"target\":\"sip:+19005551234@premium.example\"}"))
        {
            gateway.Host.SendToolRequest(callId, 9, new ToolRequestMessage { Name = "transfer", Arguments = arguments.RootElement });
        }
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.ToolResult).Count == 1, WaitMs));
        ToolResultMessage result = gateway.Host.FramesOf(LinkMessageType.ToolResult)[0].AsFrame().ReadToolResult(out uint requestId);
        Assert.Equal(9u, requestId);
        Assert.Equal(LinkToolStatus.Failed, result.Status);
        Assert.Contains("destinationPrefixes", result.Message, StringComparison.Ordinal);
        Assert.Equal(CallState.Active, gateway.Controller.State);
        phone.Agent.Hangup();
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
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

    /// <summary>Starts an admin endpoint on a free loopback port wired to the gateway's controller, retrying when the
    /// port picked is taken before the listener binds it.</summary>
    private static AdminEndpoint StartAdmin(GatewayLoopback gateway, string token)
    {
        HealthStatus health = new()
        {
            Status = "ok",
            Version = "test",
            LinkConnected = true,
            LinkOutboundRate = 16000,
            Registered = true,
            RegistrationState = "none",
            TickFifo = false,
        };
        for (int attempt = 1; ; attempt++)
        {
            TcpListener probe = new(IPAddress.Loopback, 0);
            probe.Start();
            int port = ((IPEndPoint)probe.LocalEndpoint).Port;
            probe.Stop();
            AdminEndpoint admin = new(port, token, gateway.Metrics, () => health, gateway.Controller.PlaceCallAsync);
            try
            {
                admin.Start();
                return admin;
            }
            catch (HttpListenerException) when (attempt < AdminStartAttempts)
            {
                admin.Dispose();
            }
        }
    }
}
