namespace HartsyInference.VoiceHost.Config;

/// <summary>The PhoneLink socket the gateway dials (<c>link</c> section).</summary>
public sealed record HostLinkConfig
{
    /// <summary>Unix socket to listen on; under systemd it lives in the host unit's <c>RuntimeDirectory</c>.</summary>
    public string SocketPath { get; set; } = "/run/hartsyinference/phone.sock";

    /// <summary>Octal mode of the socket file: owner read and write at least, nothing beyond 0660.</summary>
    public string SocketMode { get; set; } = "0660";

    /// <summary>Absolute path of the file holding the shared link token (mode 0600 or 0400). Empty means no token, and
    /// the gateway must then send none either.</summary>
    public string TokenFile { get; set; } = "";

    /// <summary>Reply audio sent ahead of the 20 ms pace when a burst starts, so the gateway's clock never waits on the
    /// host's.</summary>
    public int PrebufferMs { get; set; } = 40;

    /// <summary>A connection that has received nothing, or has had one write stuck, for this long is closed.</summary>
    public int LivenessTimeoutSeconds { get; set; } = 20;
}
