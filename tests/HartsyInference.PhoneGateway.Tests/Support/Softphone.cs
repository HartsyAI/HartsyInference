using System.Net;
using SIPSorcery.Media;
using SIPSorcery.Net;
using SIPSorcery.SIP;
using SIPSorcery.SIP.App;
using SIPSorceryMedia.Abstractions;

namespace HartsyInference.PhoneGateway.Tests.Support;

/// <summary>A stock sipsorcery user agent on an ephemeral 127.0.0.1 port with a sine-wave source; answers any INVITE and
/// counts the INVITEs and G.711 RTP packets it receives.</summary>
internal sealed class Softphone : IDisposable
{
    private readonly SIPTransport _transport = new();
    private long _rtpFrames;
    private int _incomingCalls;

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
        Agent.OnIncomingCall += (ua, request) =>
        {
            Interlocked.Increment(ref _incomingCalls);
            _ = AnswerAsync(ua, request);
        };
    }

    public SIPUserAgent Agent { get; }

    public ManualResetEventSlim HungUp { get; } = new(false);

    public string? LastFailure { get; private set; }

    public int? LastFailureStatus { get; private set; }

    public int Port => _transport.GetSIPChannels()[0].ListeningEndPoint.Port;

    public long RtpFramesReceived => Interlocked.Read(ref _rtpFrames);

    /// <summary>INVITEs offered to this phone.</summary>
    public int IncomingCalls => Volatile.Read(ref _incomingCalls);

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
