using System.Collections.Concurrent;
using System.Net;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Metrics;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneGateway.Transport;
using HartsyInference.PhoneLink;

namespace HartsyInference.VoiceHost.Tests.Loopback;

/// <summary>The real phone gateway in-process: its PhoneLink client dialling the host's socket, a SIP account on an
/// ephemeral 127.0.0.1 port and the real <see cref="CallController"/>, with every tick-thread frame recorded and every host
/// event and accepted reply frame noted with the time it arrived.</summary>
internal sealed class LoopbackGateway : IDisposable
{
    public const int WaitMs = 30_000;

    private readonly ConcurrentQueue<(long Ns, LinkEventMessage Event)> _events = new();
    private readonly ConcurrentQueue<(long Ns, uint Turn, int Samples)> _replyFrames = new();
    private readonly ConcurrentQueue<(long Ns, uint Turn)> _flushes = new();

    private LoopbackGateway(EngineLink link, SipAccount account, CallController controller, GatewayMetrics metrics, TickRecorder ticks)
    {
        Link = link;
        Account = account;
        Controller = controller;
        Metrics = metrics;
        Ticks = ticks;
    }

    public EngineLink Link { get; }

    public SipAccount Account { get; }

    public CallController Controller { get; }

    public GatewayMetrics Metrics { get; }

    public TickRecorder Ticks { get; }

    public int Port => Account.ListeningPort;

    /// <summary>Host events in arrival order.</summary>
    public (long Ns, LinkEventMessage Event)[] Events => [.. _events];

    /// <summary>Reply frames the gateway accepted (past its flush rule), in arrival order.</summary>
    public (long Ns, uint Turn, int Samples)[] ReplyFrames => [.. _replyFrames];

    /// <summary><c>Flush</c> frames that raised the gateway's epoch, with the time each arrived.</summary>
    public (long Ns, uint Turn)[] Flushes => [.. _flushes];

    public static LoopbackGateway Start(string socketPath, string token, int outageHangupMs)
    {
        EngineLink link = new(new EngineLinkOptions { SocketPath = socketPath, Token = token, ReconnectBaseMs = 50, ReconnectCapMs = 500 });
        SipAccount account = new(new SipAccountOptions { ListenAddress = "127.0.0.1", Port = 0, PublicAddress = PublicAddressResolver.Parse("none") });
        GatewayMetrics metrics = new();
        TickRecorder ticks = new();
        CallController controller = new(account, link, new CallControllerOptions
        {
            RtpPortStart = 41_000,
            RtpPortEnd = 41_200,
            BindAddress = IPAddress.Loopback,
            Outage = new LinkOutageGuardOptions { OutageHangupMs = outageHangupMs },
        }, new PromptPlayer(), metrics);
        controller.MediaCreated = (source, _) => ticks.Attach(source);
        LoopbackGateway gateway = new(link, account, controller, metrics, ticks);
        controller.Start();
        Action<uint, LinkEventMessage>? logEvent = link.Event;
        link.Event = (callId, item) =>
        {
            gateway._events.Enqueue((MonotonicClock.NowNs(), item));
            logEvent?.Invoke(callId, item);
        };
        OutboundAudioHandler? play = link.OutboundAudio;
        link.OutboundAudio = (callId, turnId, pcm) =>
        {
            gateway._replyFrames.Enqueue((MonotonicClock.NowNs(), turnId, pcm.Length));
            play?.Invoke(callId, turnId, pcm);
        };
        Func<uint, uint, uint>? flush = link.Flush;
        link.Flush = (callId, turnId) =>
        {
            gateway._flushes.Enqueue((MonotonicClock.NowNs(), turnId));
            return flush?.Invoke(callId, turnId) ?? 0;
        };
        link.Start();
        account.Start();
        return gateway;
    }

    public bool WaitUntil(Func<bool> condition, int timeoutMs = WaitMs)
    {
        long deadline = Environment.TickCount64 + timeoutMs;
        while (Environment.TickCount64 < deadline)
        {
            if (condition())
            {
                return true;
            }
            Thread.Sleep(5);
        }
        return condition();
    }

    public void Dispose()
    {
        Controller.Dispose();
        Account.Dispose();
        Link.Dispose();
    }
}
