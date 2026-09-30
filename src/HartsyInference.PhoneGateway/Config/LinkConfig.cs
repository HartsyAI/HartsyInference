namespace HartsyInference.PhoneGateway.Config;

/// <summary>The PhoneLink socket to the voice host (<c>link</c> section).</summary>
public sealed record LinkConfig
{
    public string SocketPath { get; set; } = "/run/hartsyinference/phone.sock";

    /// <summary>Absolute path of the file holding the shared link token (mode 0600/0400); empty means no token.</summary>
    public string TokenFile { get; set; } = "";

    /// <summary>How long a live call waits for the host before the gateway says goodbye and hangs up.</summary>
    public int OutageHangupSeconds { get; set; } = 20;
}
