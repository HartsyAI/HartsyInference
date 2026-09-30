using System.Net;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Metrics;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneGateway.Tests.Support;
using HartsyInference.PhoneGateway.Transport;
using HartsyInference.PhoneLink;
using SIPSorcery.Media;
using SIPSorcery.Net;
using SIPSorcery.SIP;
using SIPSorcery.SIP.App;
using SIPSorceryMedia.Abstractions;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Whole-gateway calls over loopback: a stock sipsorcery softphone with a tone source on one side, the real
/// <see cref="CallController"/> with a fake echo host on the other. What the unit tests cannot see is asserted here:
/// that sipsorcery really drives our source and packet handler, that DTMF crosses the link, that both hang-up
/// directions tear down once, and that a second INVITE is refused with 486.</summary>
[Trait("Category", "Integration")]
public sealed class LoopbackSipCallTests
{
    private const int WaitMs = 8000;
    private const int TalkMs = 3000;

    private readonly ITestOutputHelper _output;
    public LoopbackSipCallTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task InboundCall_AudioBothWays_DtmfAndRemoteHangup()
    {
        using Loopback gateway = Loopback.Start(new CallControllerOptions());
        using Softphone phone = new();
        Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.CallStart).Count == 1, WaitMs));
        CallStartMessage start = gateway.Host.FramesOf(LinkMessageType.CallStart)[0].AsFrame().ReadCallStart();
        Assert.Equal(LinkCallDirection.Inbound, start.Direction);
        Assert.False(start.Resume);
        await Task.Delay(TalkMs);
        MediaSnapshot media = gateway.Metrics.MediaProbe!()!;
        long toPhone = phone.RtpFramesReceived;
        int toHost = gateway.Host.AudioFramesReceived;
        _output.WriteLine($"rtp->phone={toPhone} rtp->host={toHost} jitter received={media.JitterReceived} lost={media.JitterLost} " +
            $"tick frames={media.TickFrames} silence={media.TickSilence} fifo={media.TickFifo} lateness={media.Lateness}");
        Assert.True(toPhone >= 100, $"phone got {toPhone} frames");
        Assert.True(toHost >= 100, $"host got {toHost} frames");
        Assert.True(media.JitterReceived >= 100);
        Assert.True(media.TickSilence < media.TickFrames, "echoed audio never reached the wire");
        await phone.Agent.SendDtmf(5);
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.DtmfEvent).Count >= 1, WaitMs));
        LinkDtmf dtmf = gateway.Host.FramesOf(LinkMessageType.DtmfEvent)[0].AsFrame().ReadDtmf();
        Assert.Equal('5', dtmf.Digit);
        phone.Agent.Hangup();
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.CallEnd).Count == 1, WaitMs));
        Assert.Equal(LinkCallEndReason.RemoteHangup, gateway.Host.FramesOf(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
        Assert.Equal(1, gateway.Metrics.CallsInbound);
        Assert.Equal(0, gateway.Metrics.CallsActive);
    }

    [Fact]
    public async Task HostHangupTool_EndsTheCallAtThePhone()
    {
        using Loopback gateway = Loopback.Start(new CallControllerOptions());
        using Softphone phone = new();
        Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        uint callId = gateway.Controller.Current!.CallId;
        gateway.Host.SendToolRequest(callId, 7, new ToolRequestMessage { Name = "hangup" });
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.ToolResult).Count == 1, WaitMs));
        ToolResultMessage result = gateway.Host.FramesOf(LinkMessageType.ToolResult)[0].AsFrame().ReadToolResult(out uint requestId);
        Assert.Equal(7u, requestId);
        Assert.Equal(LinkToolStatus.Ok, result.Status);
        Assert.True(phone.HungUp.Wait(WaitMs), "the phone never saw the BYE");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
        // The host owns CallEnd after its own hangup tool; the gateway must not send a second one.
        await Task.Delay(200);
        Assert.Empty(gateway.Host.FramesOf(LinkMessageType.CallEnd));
    }

    [Fact]
    public async Task SecondInvite_DuringACall_GetsBusyHere()
    {
        using Loopback gateway = Loopback.Start(new CallControllerOptions());
        using Softphone first = new();
        using Softphone second = new();
        Assert.True(await first.CallAsync(gateway.Port), $"call failed: {first.LastFailure}");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        Assert.False(await second.CallAsync(gateway.Port));
        _output.WriteLine($"second INVITE: {second.LastFailureStatus} {second.LastFailure}");
        Assert.Equal(486, second.LastFailureStatus);
        Assert.Equal(CallState.Active, gateway.Controller.State);
        Assert.Equal(1, gateway.Metrics.CallsRejectedBusy);
        first.Agent.Hangup();
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
    }

    [Fact]
    public async Task OutboundCall_PlacedByTheGateway_AndHungUpByIt()
    {
        using Loopback gateway = Loopback.Start(new CallControllerOptions());
        using Softphone phone = new();
        CallPlacementResult placed = await gateway.Controller.PlaceCallAsync($"sip:phone@127.0.0.1:{phone.Port}");
        Assert.True(placed.Placed, placed.Message);
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Active, WaitMs));
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.CallStart).Count == 1, WaitMs));
        Assert.Equal(LinkCallDirection.Outbound, gateway.Host.FramesOf(LinkMessageType.CallStart)[0].AsFrame().ReadCallStart().Direction);
        await Task.Delay(TalkMs);
        long toPhone = phone.RtpFramesReceived;
        int toHost = gateway.Host.AudioFramesReceived;
        _output.WriteLine($"outbound: rtp->phone={toPhone} rtp->host={toHost}");
        Assert.True(toPhone >= 100, $"phone got {toPhone} frames");
        Assert.True(toHost >= 100, $"host got {toHost} frames");
        gateway.Controller.HangUp(LinkCallEndReason.Completed);
        Assert.True(phone.HungUp.Wait(WaitMs), "the phone never saw the BYE");
        Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, WaitMs));
        Assert.True(gateway.Host.WaitUntil(() => gateway.Host.FramesOf(LinkMessageType.CallEnd).Count == 1, WaitMs));
        Assert.Equal(LinkCallEndReason.Completed, gateway.Host.FramesOf(LinkMessageType.CallEnd)[0].AsFrame().ReadCallEnd());
        Assert.Equal(1, gateway.Metrics.CallsOutbound);
    }

    [Fact]
    public async Task RejectPolicy_DeclinesWith603()
    {
        using Loopback gateway = Loopback.Start(new CallControllerOptions { InboundPolicy = InboundPolicy.Reject });
        using Softphone phone = new();
        Assert.False(await phone.CallAsync(gateway.Port));
        _output.WriteLine($"declined: {phone.LastFailureStatus} {phone.LastFailure}");
        Assert.Equal(603, phone.LastFailureStatus);
        Assert.Equal(CallState.Idle, gateway.Controller.State);
        Assert.Equal(1, gateway.Metrics.CallsDeclined);
        Assert.Empty(gateway.Host.FramesOf(LinkMessageType.CallStart));
    }

    /// <summary>The gateway side: fake echo host, link, SIP account on an ephemeral loopback port, controller.</summary>
    private sealed class Loopback : IDisposable
    {
        public required FakeLinkHost Host { get; init; }
        public required EngineLink Link { get; init; }
        public required SipAccount Account { get; init; }
        public required CallController Controller { get; init; }
        public required GatewayMetrics Metrics { get; init; }

        public int Port => Account.ListeningPort;

        public static Loopback Start(CallControllerOptions options)
        {
            FakeLinkHost host = new() { Echo = true, OutboundRate = 16000 };
            host.Start();
            EngineLink link = new(new EngineLinkOptions { SocketPath = host.SocketPath, ReconnectBaseMs = 10, ReconnectCapMs = 100 });
            SipAccount account = new(new SipAccountOptions { ListenAddress = "127.0.0.1", Port = 0, PublicAddress = PublicAddressResolver.Parse("none") });
            GatewayMetrics metrics = new();
            CallController controller = new(account, link, options with
            {
                RtpPortStart = 40000,
                RtpPortEnd = 40100,
                BindAddress = IPAddress.Loopback,
                Outage = new LinkOutageGuardOptions { OutageHangupMs = 5000 },
            }, new PromptPlayer(), metrics);
            controller.Start();
            link.Start();
            account.Start();
            Loopback loopback = new() { Host = host, Link = link, Account = account, Controller = controller, Metrics = metrics };
            if (!loopback.WaitUntil(() => link.IsConnected, WaitMs))
            {
                loopback.Dispose();
                throw new InvalidOperationException("The gateway never connected to the fake host.");
            }
            return loopback;
        }

        public bool WaitUntil(Func<bool> condition, int timeoutMs) => Host.WaitUntil(condition, timeoutMs);

        public void Dispose()
        {
            Controller.Dispose();
            Account.Dispose();
            Link.Dispose();
            Host.Dispose();
        }
    }

    /// <summary>A stock sipsorcery user agent with a sine-wave source; answers any inbound INVITE.</summary>
    private sealed class Softphone : IDisposable
    {
        private readonly SIPTransport _transport = new();
        private long _rtpFrames;

        public Softphone()
        {
            _transport.AddSIPChannel(new SIPUDPChannel(new IPEndPoint(IPAddress.Loopback, 0)));
            Agent = new SIPUserAgent(_transport, null);
            Agent.OnCallHungup += _ => HungUp.Set();
            Agent.ClientCallFailed += (_, message, response) =>
            {
                LastFailure = message;
                LastFailureStatus = response?.StatusCode;
            };
            Agent.OnIncomingCall += (ua, request) => _ = AnswerAsync(ua, request);
        }

        public SIPUserAgent Agent { get; }

        public ManualResetEventSlim HungUp { get; } = new(false);

        public string? LastFailure { get; private set; }

        public int? LastFailureStatus { get; private set; }

        public int Port => _transport.GetSIPChannels()[0].ListeningEndPoint.Port;

        public long RtpFramesReceived => Interlocked.Read(ref _rtpFrames);

        public Task<bool> CallAsync(int gatewayPort) => Agent.Call($"sip:agent@127.0.0.1:{gatewayPort}", null, null, CreateSession(), 10);

        private async Task AnswerAsync(SIPUserAgent ua, SIPRequest request)
        {
            SIPServerUserAgent uas = ua.AcceptCall(request);
            await ua.Answer(uas, CreateSession());
        }

        private VoIPMediaSession CreateSession()
        {
            AudioExtrasSource source = new(new AudioEncoder(), new AudioSourceOptions { AudioSource = AudioSourcesEnum.SineWave });
            VoIPMediaSession session = new(new VoIPMediaSessionConfig
            {
                MediaEndPoint = new MediaEndPoints { AudioSource = source },
                BindAddress = IPAddress.Loopback,
            });
            session.AcceptRtpFromAny = true;
            session.OnRtpPacketReceived += (_, mediaType, packet) =>
            {
                if (mediaType == SDPMediaTypesEnum.audio && packet.Header.PayloadType is 0 or 8)
                {
                    Interlocked.Increment(ref _rtpFrames);
                }
            };
            return session;
        }

        public void Dispose()
        {
            Agent.Dispose();
            _transport.Shutdown();
            _transport.Dispose();
            HungUp.Dispose();
        }
    }
}
