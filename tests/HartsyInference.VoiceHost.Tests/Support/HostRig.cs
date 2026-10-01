using HartsyInference.PhoneLink;
using HartsyInference.VoiceHost.Calls;
using HartsyInference.VoiceHost.Link;

namespace HartsyInference.VoiceHost.Tests.Support;

/// <summary>A started <see cref="PhoneLinkServer"/> on a socket in a fresh temporary directory, with scripted sessions, and
/// helpers to connect a <see cref="FakeGateway"/> and to wait on the host's view of a call.</summary>
internal sealed class HostRig : IAsyncDisposable
{
    public const string Token = "test-link-token";

    private readonly string _directory;
    private readonly List<FakeGateway> _gateways = [];

    private HostRig(string directory, PhoneLinkServer server, FakeSessionFactory factory)
    {
        _directory = directory;
        Server = server;
        Factory = factory;
    }

    public PhoneLinkServer Server { get; }

    public FakeSessionFactory Factory { get; }

    public string SocketPath => Server.SocketPath;

    public static HostRig Start(Func<PhoneLinkServerOptions, PhoneLinkServerOptions>? configure = null)
    {
        string directory = NewSocketDirectory();
        PhoneLinkServerOptions options = new() { SocketPath = Path.Combine(directory, "phone.sock"), Token = Token, HandshakeTimeoutMs = 2000 };
        options = configure?.Invoke(options) ?? options;
        FakeSessionFactory factory = new();
        PhoneLinkServer server = new(options, factory);
        server.Start();
        return new HostRig(directory, server, factory);
    }

    /// <summary>A short, private directory for a socket (Unix socket paths are limited to about 100 bytes).</summary>
    public static string NewSocketDirectory()
    {
        string directory = Path.Combine(Path.GetTempPath(), "hvh-" + Guid.NewGuid().ToString("N")[..10]);
        Directory.CreateDirectory(directory);
        return directory;
    }

    /// <summary>Connects a gateway and completes the handshake with <paramref name="token"/>.</summary>
    public FakeGateway Connect(string token = Token)
    {
        FakeGateway gateway = FakeGateway.Connect(SocketPath);
        _gateways.Add(gateway);
        FakeGateway.RecordedFrame answer = gateway.Hello(token);
        if (answer.Header.Type != LinkMessageType.HelloAck)
        {
            throw new InvalidOperationException($"The host answered Hello with {answer.Header.Type}.");
        }
        return gateway;
    }

    /// <summary>Connects, announces call <paramref name="callId"/> and waits until the host is ready for its audio.</summary>
    public async Task<(FakeGateway Gateway, FakeCallSession Session)> StartCallAsync(uint callId = 1, int sessionIndex = 0)
    {
        FakeGateway gateway = Connect();
        gateway.SendCallStart(callId);
        FakeCallSession session = await Factory.WaitForSessionAsync(sessionIndex);
        WaitUntilReady(callId);
        return (gateway, session);
    }

    /// <summary>The host's call <paramref name="callId"/> on the current connection, or null.</summary>
    public VoiceCall? Call(uint callId) => Server.Current?.Calls.FirstOrDefault(c => c.CallId == callId);

    public void WaitUntilReady(uint callId)
    {
        if (!Wait(() => Call(callId) is { AudioReady: true }))
        {
            throw new TimeoutException($"Call {callId} never became ready on the host.");
        }
    }

    public static bool Wait(Func<bool> condition, int timeoutMs = FakeGateway.WaitMs)
    {
        long deadline = Environment.TickCount64 + timeoutMs;
        while (Environment.TickCount64 < deadline)
        {
            if (condition())
            {
                return true;
            }
            Thread.Sleep(2);
        }
        return condition();
    }

    public async ValueTask DisposeAsync()
    {
        await Server.StopAsync(LinkCallEndReason.LocalHangup);
        foreach (FakeGateway gateway in _gateways)
        {
            gateway.Dispose();
        }
        Directory.Delete(_directory, recursive: true);
    }
}
