namespace HartsyInference.PhoneGateway.Config;

/// <summary>The loopback admin endpoint (<c>admin</c> section).</summary>
public sealed record AdminConfig
{
    /// <summary>Port on 127.0.0.1; zero disables the endpoint.</summary>
    public int Port { get; set; } = 9280;

    /// <summary>Absolute path of the file holding the bearer token for <c>POST /calls</c> (mode 0600/0400); empty
    /// disables that endpoint.</summary>
    public string TokenFile { get; set; } = "";
}
