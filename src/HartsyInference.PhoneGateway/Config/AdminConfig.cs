namespace HartsyInference.PhoneGateway.Config;

/// <summary>The loopback admin endpoint (<c>admin</c> section).</summary>
public sealed record AdminConfig
{
    /// <summary>Port on 127.0.0.1; zero disables the endpoint.</summary>
    public int Port { get; set; } = 9280;

    /// <summary>Name of the environment variable holding the bearer token for <c>POST /calls</c>; unset disables it.</summary>
    public string TokenEnv { get; set; } = "HARTSY_PHONE_ADMIN_TOKEN";
}
