using System.Net;
using System.Reflection;
using HartsyInference.Core.Logging;
using HartsyInference.PhoneGateway.Admin;
using HartsyInference.PhoneGateway.Config;
using HartsyInference.PhoneGateway.Logging;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Metrics;
using HartsyInference.PhoneGateway.Runtime;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneGateway.Transport;

namespace HartsyInference.PhoneGateway;

/// <summary>Composition root: builds the link, the SIP account, the call controller and the admin endpoint from
/// <see cref="GatewaySettings"/>, starts them in dependency order and stops them in reverse.</summary>
public sealed class GatewayHost : IDisposable
{
    private readonly GatewaySettings _settings;
    private readonly GatewayMetrics _metrics = new();
    private readonly EngineLink _link;
    private readonly SipAccount _account;
    private readonly CallController _controller;
    private readonly AdminEndpoint? _admin;
    private int _started;
    private int _stopped;

    public GatewayHost(GatewaySettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        _settings = settings;
        GatewayConfig config = settings.Config;
        _link = new EngineLink(new EngineLinkOptions { SocketPath = config.Link.SocketPath, Token = settings.LinkToken });
        _account = new SipAccount(new SipAccountOptions
        {
            ListenAddress = config.Sip.ListenAddress,
            Port = config.Sip.Port,
            Transport = config.Sip.Transport,
            Registrar = config.Sip.Registrar,
            Username = config.Sip.Username,
            Password = settings.SipPassword,
            RegistrationExpirySeconds = config.Sip.RegistrationExpirySeconds,
            PublicAddress = PublicAddressResolver.Parse(config.Sip.PublicAddress),
        });
        _controller = new CallController(_account, _link, new CallControllerOptions
        {
            InboundPolicy = config.Sip.InboundPolicy,
            Allowlist = config.Sip.Allowlist,
            GreetingPromptFile = config.Sip.GreetingPromptFile,
            Codec = config.Sip.Codec,
            RtpPortStart = config.Sip.RtpPortStart,
            RtpPortEnd = config.Sip.RtpPortEnd,
            BindAddress = config.Sip.ListenAddress == "0.0.0.0" ? null : IPAddress.Parse(config.Sip.ListenAddress),
            RingTimeoutSeconds = config.Sip.RingTimeoutSeconds,
            Tick = new ClockedAudioSourceOptions
            {
                FifoPriority = config.Media.FifoPriority,
                TickCpu = config.Media.TickCpu,
                WarmUpTicks = config.Media.WarmUpTicks,
                Codec = config.Sip.Codec,
            },
            Outage = new LinkOutageGuardOptions { OutageHangupMs = config.Link.OutageHangupSeconds * 1000 },
            Recording = new RecordingOptions { Enabled = config.Recording.Enabled, Directory = config.Recording.Directory },
        }, new PromptPlayer(), _metrics);
        _metrics.LinkProbe = () => new LinkSnapshot(
            _link.IsConnected, _link.OutboundRate, _link.Reconnects, _link.LastRttNs / 1_000_000.0,
            _link.AudioLaneDropped, _link.InboundDroppedWhileDown, _link.StaleOutboundDropped, _link.FramesSent, _link.FramesReceived);
        _metrics.RegisteredProbe = () => _account.Registrar.Length == 0 || _account.IsRegistered;
        if (config.Admin.Port > 0)
        {
            _admin = new AdminEndpoint(config.Admin.Port, settings.AdminToken, _metrics, Health, _controller.PlaceCallAsync);
        }
    }

    public static string Version =>
        Assembly.GetExecutingAssembly().GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";

    public void Start()
    {
        if (Interlocked.Exchange(ref _started, 1) != 0)
        {
            throw new InvalidOperationException("GatewayHost is already started.");
        }
        RuntimeTuning.Apply();
        SipLogBridge.Install(_settings.Config.Logging.SipDebug);
        if (_settings.Config.Recording.Enabled)
        {
            Logs.Warning($"[PhoneGateway] Call recording is ON ({_settings.Config.Recording.Directory}); make sure callers are told where the law requires it.");
        }
        _controller.Start();
        _link.Start();
        _account.Start();
        _admin?.Start();
        Logs.Info($"[PhoneGateway] Gateway {Version} started.");
    }

    public void Stop()
    {
        if (Interlocked.Exchange(ref _stopped, 1) != 0)
        {
            return;
        }
        Logs.Info("[PhoneGateway] Stopping.");
        _admin?.Stop();
        _controller.Dispose();
        _account.Dispose();
        _link.Dispose();
        _admin?.Dispose();
    }

    private HealthStatus Health()
    {
        bool registered = _account.Registrar.Length == 0 || _account.IsRegistered;
        CallSummary? call = _controller.Current;
        MediaSnapshot? media = _metrics.MediaProbe?.Invoke();
        return new HealthStatus
        {
            Status = _link.IsConnected && registered ? "ok" : "degraded",
            Version = Version,
            LinkConnected = _link.IsConnected,
            LinkOutboundRate = _link.OutboundRate,
            Registered = registered,
            RegistrationState = _account.RegistrationState,
            TickFifo = media?.TickFifo ?? false,
            Call = call,
        };
    }

    public void Dispose() => Stop();
}
