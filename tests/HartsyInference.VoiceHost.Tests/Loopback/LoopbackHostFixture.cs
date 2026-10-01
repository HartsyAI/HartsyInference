using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneLink;
using HartsyInference.Tests.Common;
using HartsyInference.VoiceHost.Calls;
using HartsyInference.VoiceHost.Config;
using HartsyInference.VoiceHost.Tests.Support;
using Xunit;

namespace HartsyInference.VoiceHost.Tests.Loopback;

/// <summary>One real voice host for a class of loopback calls: Kokoro and Whisper small.en loaded once on the RTX 3060, a
/// scripted language model the tests queue replies on, the PhoneLink socket with a token, and the real gateway connected to
/// it. <see cref="SkipReason"/> is set, and nothing is started, when the weights or the card are missing.</summary>
public sealed class LoopbackHostFixture : IAsyncLifetime
{
    public const string Token = "loopback-link-token";
    public const string Greeting = "Hello, how can I help you?";

    private readonly List<string> _log = [];
    private string? _directory;

    /// <summary>Why the class is skipped; null when the host is up.</summary>
    public string? SkipReason { get; private set; }

    internal ScriptedRounds Text { get; } = new();

    internal VoiceHostService? Host { get; private set; }

    internal LoopbackGateway? Gateway { get; private set; }

    /// <summary>What the fixture logged while starting, for the test output.</summary>
    public IReadOnlyList<string> Log => _log;

    /// <summary>The host's current call.</summary>
    internal VoiceCall? Call => Host?.Server?.Current?.Calls.FirstOrDefault();

    public async Task InitializeAsync()
    {
        if (!RealWeightGate.Require(_log.Add, LoopbackAssets.All()))
        {
            SkipReason = string.Join("; ", _log);
            return;
        }
        string? device = LoopbackAssets.AudioDevice(_log.Add);
        if (device is null)
        {
            SkipReason = string.Join("; ", _log);
            return;
        }
        _directory = HostRig.NewSocketDirectory();
        string socket = Path.Combine(_directory, "phone.sock");
        string token = Path.Combine(_directory, "link-token");
        await File.WriteAllTextAsync(token, Token + "\n");
        File.SetUnixFileMode(token, UnixFileMode.UserRead | UnixFileMode.UserWrite);
        VoiceHostConfig config = new()
        {
            Link = new HostLinkConfig { SocketPath = socket, TokenFile = token },
            Models = new HostModelsConfig { AudioDevice = device, LlmDevice = "cpu", WakeModelRoot = LoopbackAssets.WakeRoot },
            Agent = new HostAgentConfig { Greeting = Greeting },
        };
        Host = new VoiceHostService(VoiceHostConfigLoader.Resolve(config), _ => Text);
        await Host.StartAsync(CancellationToken.None);
        Gateway = LoopbackGateway.Start(socket, Token, outageHangupMs: 3_000);
        if (!Gateway.WaitUntil(() => Gateway.Link.IsConnected, 10_000))
        {
            throw new InvalidOperationException("The gateway never connected to the voice host.");
        }
        _log.Add($"host up on {device}, gateway SIP port {Gateway.Port}");
    }

    public async Task DisposeAsync()
    {
        Gateway?.Dispose();
        if (Host is not null)
        {
            await Host.StopAsync(CancellationToken.None);
        }
        if (_directory is not null)
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    /// <summary>Hangs up whatever call is left and waits for both sides to forget it.</summary>
    internal void EndCall(LoopbackPhone phone)
    {
        if (Gateway is null)
        {
            return;
        }
        if (Gateway.Controller.State != CallState.Idle)
        {
            phone.Agent.Hangup();
        }
        Gateway.WaitUntil(() => Gateway.Controller.State == CallState.Idle && Call is null, 10_000);
    }

    /// <summary>Waits for the next host event after <paramref name="sinceNs"/> that matches.</summary>
    internal LinkEventMessage? WaitForEvent(long sinceNs, Func<LinkEventMessage, bool> match, int timeoutMs)
    {
        LinkEventMessage? found = null;
        Gateway!.WaitUntil(() =>
        {
            found = Gateway.Events.Where(e => e.Ns >= sinceNs).Select(e => e.Event).FirstOrDefault(match);
            return found is not null;
        }, timeoutMs);
        return found;
    }
}
