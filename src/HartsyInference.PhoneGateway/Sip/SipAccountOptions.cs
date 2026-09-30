namespace HartsyInference.PhoneGateway.Sip;

/// <summary>Transport and registration settings for <see cref="SipAccount"/>. The password is already resolved from
/// its environment variable by the config loader; it is never logged.</summary>
public sealed record SipAccountOptions
{
    /// <summary>Shortest registration expiry accepted.</summary>
    public const int MinExpirySeconds = 60;

    /// <summary>Longest registration expiry accepted.</summary>
    public const int MaxExpirySeconds = 120;

    public string ListenAddress { get; init; } = "0.0.0.0";

    public int Port { get; init; } = 5060;

    /// <summary><c>udp</c> or <c>tcp</c>.</summary>
    public string Transport { get; init; } = "udp";

    /// <summary>Registrar host (with optional port); empty means no registration, the LAN/IP-auth case.</summary>
    public string Registrar { get; init; } = "";

    public string Username { get; init; } = "";

    public string Password { get; init; } = "";

    /// <summary>Registration expiry, clamped to 60..120 s; sipsorcery refreshes five seconds early.</summary>
    public int RegistrationExpirySeconds { get; init; } = MinExpirySeconds;

    public required PublicAddressResolver PublicAddress { get; init; }
}
