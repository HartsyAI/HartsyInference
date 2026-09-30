using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Sip;

namespace HartsyInference.PhoneGateway.Config;

/// <summary>SIP transport, registration, screening and RTP settings (<c>sip</c> section).</summary>
public sealed record SipConfig
{
    public string ListenAddress { get; set; } = "0.0.0.0";

    public int Port { get; set; } = 5060;

    /// <summary><c>udp</c> or <c>tcp</c>.</summary>
    public string Transport { get; set; } = "udp";

    /// <summary>Registrar host[:port]; empty for no registration.</summary>
    public string Registrar { get; set; } = "";

    public string Username { get; set; } = "";

    /// <summary>Absolute path of the file holding the SIP password (mode 0600/0400); required when a registrar is set.</summary>
    public string PasswordFile { get; set; } = "";

    public int RegistrationExpirySeconds { get; set; } = 60;

    /// <summary><c>none</c>, <c>stun:host[:port]</c> or an IP literal.</summary>
    public string PublicAddress { get; set; } = "none";

    public int RtpPortStart { get; set; } = 20000;

    public int RtpPortEnd { get; set; } = 20100;

    public AudioCodecPreference Codec { get; set; } = AudioCodecPreference.Any;

    public InboundPolicy InboundPolicy { get; set; } = InboundPolicy.AllowAll;

    public string[] Allowlist { get; set; } = [];

    /// <summary>Number prefixes outbound calls and transfers may dial (e.g. <c>+1555</c>); empty allows any. The agent
    /// can be talked into dialling by its caller, so set this on a real trunk.</summary>
    public string[] DestinationPrefixes { get; set; } = [];

    /// <summary>Raw 8 kHz PCM16 file played to every answered inbound call; null for none.</summary>
    public string? GreetingPromptFile { get; set; }

    public int RingTimeoutSeconds { get; set; } = 45;
}
