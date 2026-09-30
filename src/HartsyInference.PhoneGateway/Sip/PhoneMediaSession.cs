using System.Net;
using HartsyInference.Core.Logging;
using SIPSorcery.Media;
using SIPSorcery.Net;
using SIPSorcery.Sys;
using SIPSorceryMedia.Abstractions;

namespace HartsyInference.PhoneGateway.Sip;

/// <summary>A <see cref="VoIPMediaSession"/> that advertises the public address in its SDP, accepts RTP from any
/// source and latches its send destination onto the first packet's origin (symmetric RTP, what providers' comedia
/// mode expects behind NAT). The source and sink come from the gateway's own media classes.</summary>
public sealed class PhoneMediaSession : VoIPMediaSession
{
    private readonly IPAddress? _publicAddress;
    private int _latched;

    public PhoneMediaSession(MediaEndPoints endPoints, IPAddress? bindAddress, PortRange? rtpPortRange, IPAddress? publicAddress)
        : base(new VoIPMediaSessionConfig { MediaEndPoint = endPoints, BindAddress = bindAddress, RtpPortRange = rtpPortRange })
    {
        _publicAddress = publicAddress;
        AcceptRtpFromAny = true;
        OnRtpPacketReceived += LatchRemote;
    }

    /// <summary>True once the send destination has been re-pointed at where the far end's RTP came from.</summary>
    public bool RemoteLatched => Volatile.Read(ref _latched) == 2;

    public override SDP CreateOffer(IPAddress? connectionAddress = null) => base.CreateOffer(_publicAddress ?? connectionAddress);

    public override SDP CreateAnswer(IPAddress? connectionAddress = null) => base.CreateAnswer(_publicAddress ?? connectionAddress);

    private void LatchRemote(IPEndPoint remote, SDPMediaTypesEnum mediaType, RTPPacket packet)
    {
        if (mediaType != SDPMediaTypesEnum.audio || Interlocked.CompareExchange(ref _latched, 1, 0) != 0)
        {
            return;
        }
        IPEndPoint? expected = AudioDestinationEndPoint;
        if (expected is null || expected.Equals(remote))
        {
            return;
        }
        SetDestination(SDPMediaTypesEnum.audio, remote, new IPEndPoint(remote.Address, remote.Port + 1));
        Volatile.Write(ref _latched, 2);
        Logs.Info($"[PhoneGateway] RTP destination latched to {remote} (SDP said {expected}).");
    }
}
