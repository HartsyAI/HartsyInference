using System.Net;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Runtime;
using SIPSorcery.Media;
using SIPSorcery.Net;
using SIPSorcery.SIP;
using SIPSorcery.SIP.App;
using SIPSorceryMedia.Abstractions;

namespace HartsyInference.VoiceHost.Tests.Loopback;

/// <summary>The caller: a stock sipsorcery user agent on an ephemeral 127.0.0.1 port that sends silence until told to
/// speak a clip, and records the level of every G.711 packet it receives with the time it arrived.</summary>
internal sealed class LoopbackPhone : IDisposable
{
    private readonly SIPTransport _transport = new();
    private readonly object _framesLock = new();
    private readonly List<(long Ns, int Peak)> _frames = [];
    private readonly short[] _pcm = new short[1024];
    private AudioExtrasSource? _source;
    private long _hungUpNs;

    public LoopbackPhone()
    {
        _transport.AddSIPChannel(new SIPUDPChannel(new IPEndPoint(IPAddress.Loopback, 0)));
        Agent = new SIPUserAgent(_transport, null);
        Agent.OnCallHungup += _ =>
        {
            Volatile.Write(ref _hungUpNs, MonotonicClock.NowNs());
            HungUp.Set();
        };
        Agent.ClientCallFailed += (_, message, response) =>
        {
            LastFailure = message;
            LastFailureStatus = response?.StatusCode;
        };
    }

    public SIPUserAgent Agent { get; }

    public ManualResetEventSlim HungUp { get; } = new(false);

    /// <summary>When the far end's BYE arrived, in monotonic nanoseconds; 0 before.</summary>
    public long HungUpNs => Volatile.Read(ref _hungUpNs);

    public string? LastFailure { get; private set; }

    public int? LastFailureStatus { get; private set; }

    /// <summary>Calls the gateway; true when the call was answered.</summary>
    public Task<bool> CallAsync(int gatewayPort)
    {
        HungUp.Reset();
        Volatile.Write(ref _hungUpNs, 0);
        return Agent.Call($"sip:agent@127.0.0.1:{gatewayPort}", null, null, CreateSession(), 10);
    }

    /// <summary>Sends <paramref name="pcm8k"/> (8 kHz PCM16, the line rate) into the call, then goes back to silence.</summary>
    public Task SpeakAsync(byte[] pcm8k) =>
        _source?.SendAudioFromStream(new MemoryStream(pcm8k), AudioSamplingRatesEnum.Rate8KHz)
            ?? throw new InvalidOperationException("No call is up.");

    /// <summary>Received packets, oldest first.</summary>
    public (long Ns, int Peak)[] Frames()
    {
        lock (_framesLock)
        {
            return [.. _frames];
        }
    }

    /// <summary>Received packets at or above <paramref name="peak"/> since <paramref name="fromNs"/>.</summary>
    public int AudibleSince(long fromNs, int peak) => Frames().Count(f => f.Ns >= fromNs && f.Peak >= peak);

    private VoIPMediaSession CreateSession()
    {
        AudioExtrasSource source = new(new AudioEncoder(), new AudioSourceOptions { AudioSource = AudioSourcesEnum.Silence });
        _source = source;
        VoIPMediaSession session = new(new VoIPMediaSessionConfig
        {
            MediaEndPoint = new MediaEndPoints { AudioSource = source },
            BindAddress = IPAddress.Loopback,
        });
        session.AcceptRtpFromAny = true;
        session.OnRtpPacketReceived += (_, mediaType, packet) =>
        {
            if (mediaType != SDPMediaTypesEnum.audio || packet.Header.PayloadType is not (0 or 8))
            {
                return;
            }
            G711Law law = packet.Header.PayloadType == 0 ? G711Law.MuLaw : G711Law.ALaw;
            int peak = 0;
            lock (_framesLock)
            {
                int count = Math.Min(packet.Payload.Length, _pcm.Length);
                G711.Decode(packet.Payload.AsSpan(0, count), _pcm.AsSpan(0, count), law);
                for (int i = 0; i < count; i++)
                {
                    peak = Math.Max(peak, Math.Abs((int)_pcm[i]));
                }
                _frames.Add((MonotonicClock.NowNs(), peak));
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
