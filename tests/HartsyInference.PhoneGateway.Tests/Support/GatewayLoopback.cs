using System.Net;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Metrics;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneGateway.Transport;

namespace HartsyInference.PhoneGateway.Tests.Support;

/// <summary>The gateway side of a loopback call: a fake echo host on a temporary socket, the real link, a SIP account on an
/// ephemeral 127.0.0.1 port and the real <see cref="CallController"/>, started and connected.</summary>
internal sealed class GatewayLoopback : IDisposable
{
    public const int WaitMs = 8000;

    public required FakeLinkHost Host { get; init; }
    public required EngineLink Link { get; init; }
    public required SipAccount Account { get; init; }
    public required CallController Controller { get; init; }
    public required GatewayMetrics Metrics { get; init; }

    public int Port => Account.ListeningPort;

    public static GatewayLoopback Start(CallControllerOptions options)
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
        GatewayLoopback loopback = new() { Host = host, Link = link, Account = account, Controller = controller, Metrics = metrics };
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
