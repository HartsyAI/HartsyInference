namespace HartsyInference.PhoneGateway.Config;

/// <summary>The PhoneLink socket to the voice host (<c>link</c> section).</summary>
public sealed record LinkConfig
{
    public string SocketPath { get; set; } = "/run/hartsyinference/phone.sock";

    /// <summary>Name of the environment variable holding the shared link token; unset means no token.</summary>
    public string TokenEnv { get; set; } = "HARTSY_PHONE_LINK_TOKEN";

    /// <summary>How long a live call waits for the host before the gateway says goodbye and hangs up.</summary>
    public int OutageHangupSeconds { get; set; } = 20;
}
