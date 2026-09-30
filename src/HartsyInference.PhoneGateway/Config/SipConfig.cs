using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Sip;

namespace HartsyInference.PhoneGateway.Config;

/// <summary>SIP transport, registration, screening and RTP settings (<c>sip</c> section).</summary>
public sealed record SipConfig
{
    public string ListenAddress { get; init; } = "0.0.0.0";

    public int Port { get; init; } = 5060;

    /// <summary><c>udp</c> or <c>tcp</c>.</summary>
    public string Transport { get; init; } = "udp";

    /// <summary>Registrar host[:port]; empty for no registration.</summary>
    public string Registrar { get; init; } = "";

    public string Username { get; init; } = "";

    /// <summary>Name of the environment variable holding the SIP password; required when a registrar is set.</summary>
    public string PasswordEnv { get; init; } = "HARTSY_SIP_PASSWORD";

    public int RegistrationExpirySeconds { get; init; } = 60;

    /// <summary><c>none</c>, <c>stun:host[:port]</c> or an IP literal.</summary>
    public string PublicAddress { get; init; } = "none";

    public int RtpPortStart { get; init; } = 20000;

    public int RtpPortEnd { get; init; } = 20100;

    public AudioCodecPreference Codec { get; init; } = AudioCodecPreference.Any;

    public InboundPolicy InboundPolicy { get; init; } = InboundPolicy.AllowAll;

    public string[] Allowlist { get; init; } = [];

    /// <summary>Raw 8 kHz PCM16 file played to every answered inbound call; null for none.</summary>
    public string? GreetingPromptFile { get; init; }

    public int RingTimeoutSeconds { get; init; } = 45;
}
