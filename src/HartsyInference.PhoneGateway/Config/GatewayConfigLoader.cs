using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.PhoneGateway.Sip;

namespace HartsyInference.PhoneGateway.Config;

/// <summary>Reads <c>phone.json</c>, validates it and resolves the secrets it names from the environment. A missing
/// variable is reported by name; the value is never echoed.</summary>
public static class GatewayConfigLoader
{
    /// <summary>Loads and validates the file at <paramref name="path"/>.</summary>
    /// <exception cref="GatewayConfigException">The file is missing, malformed or names a variable that is not set.</exception>
    public static GatewaySettings Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new GatewayConfigException($"Config file {path} not found. Copy phone.example.json next to the gateway and pass --config <path>.");
        }
        GatewayConfig config;
        try
        {
            using FileStream stream = File.OpenRead(path);
            config = JsonSerializer.Deserialize(stream, GatewayJsonContext.Default.GatewayConfig)
                ?? throw new GatewayConfigException($"Config file {path} is JSON null.");
        }
        catch (JsonException ex)
        {
            throw new GatewayConfigException($"Config file {path} is not valid: {ex.Message}", ex);
        }
        return Resolve(config);
    }

    /// <summary>Validates <paramref name="config"/> and resolves its secrets from the environment.</summary>
    public static GatewaySettings Resolve(GatewayConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        Validate(config);
        string sipPassword = "";
        if (config.Sip.Registrar.Length > 0)
        {
            sipPassword = RequireSecret(config.Sip.PasswordEnv, "sip.passwordEnv", "the SIP password for the registrar");
        }
        string linkToken = OptionalSecret(config.Link.TokenEnv, "link.tokenEnv") ?? "";
        string? adminToken = OptionalSecret(config.Admin.TokenEnv, "admin.tokenEnv");
        return new GatewaySettings { Config = config, SipPassword = sipPassword, LinkToken = linkToken, AdminToken = adminToken };
    }

    /// <summary>Parses <c>logging.level</c>.</summary>
    public static LogLevel ParseLogLevel(string level) => level.Trim().ToLowerInvariant() switch
    {
        "verbose" => LogLevel.Verbose,
        "debug" => LogLevel.Debug,
        "info" or "information" => LogLevel.Info,
        "warning" or "warn" => LogLevel.Warning,
        "error" => LogLevel.Error,
        _ => throw new GatewayConfigException($"logging.level '{level}' must be Verbose, Debug, Info, Warning or Error."),
    };

    private static void Validate(GatewayConfig config)
    {
        SipConfig sip = config.Sip;
        if (!System.Net.IPAddress.TryParse(sip.ListenAddress, out _))
        {
            throw new GatewayConfigException($"sip.listenAddress '{sip.ListenAddress}' is not an IP address.");
        }
        if (sip.Port is < 1 or > 65535)
        {
            throw new GatewayConfigException($"sip.port {sip.Port} is out of range.");
        }
        if (sip.Transport is not ("udp" or "tcp"))
        {
            throw new GatewayConfigException($"sip.transport '{sip.Transport}' must be udp or tcp.");
        }
        if (sip.Registrar.Length > 0 && sip.Username.Length == 0)
        {
            throw new GatewayConfigException("sip.username is required when sip.registrar is set.");
        }
        if (sip.RegistrationExpirySeconds < SipAccountOptions.MinExpirySeconds || sip.RegistrationExpirySeconds > SipAccountOptions.MaxExpirySeconds)
        {
            throw new GatewayConfigException(
                $"sip.registrationExpirySeconds {sip.RegistrationExpirySeconds} must be {SipAccountOptions.MinExpirySeconds}..{SipAccountOptions.MaxExpirySeconds}.");
        }
        try
        {
            PublicAddressResolver.Parse(sip.PublicAddress);
        }
        catch (ArgumentException ex)
        {
            throw new GatewayConfigException("sip." + ex.Message, ex);
        }
        if (sip.RtpPortStart < 1024 || sip.RtpPortEnd > 65534 || sip.RtpPortEnd <= sip.RtpPortStart)
        {
            throw new GatewayConfigException($"sip.rtpPortStart/rtpPortEnd {sip.RtpPortStart}-{sip.RtpPortEnd} is not a valid range above 1023.");
        }
        if (sip.InboundPolicy == InboundPolicy.Allowlist && sip.Allowlist.Length == 0)
        {
            throw new GatewayConfigException("sip.inboundPolicy is Allowlist but sip.allowlist is empty.");
        }
        if (sip.GreetingPromptFile is not null && !File.Exists(sip.GreetingPromptFile))
        {
            throw new GatewayConfigException($"sip.greetingPromptFile '{sip.GreetingPromptFile}' does not exist.");
        }
        if (sip.RingTimeoutSeconds is < 5 or > 300)
        {
            throw new GatewayConfigException($"sip.ringTimeoutSeconds {sip.RingTimeoutSeconds} must be 5..300.");
        }
        if (string.IsNullOrWhiteSpace(config.Link.SocketPath))
        {
            throw new GatewayConfigException("link.socketPath is required.");
        }
        if (config.Link.OutageHangupSeconds is < 1 or > 300)
        {
            throw new GatewayConfigException($"link.outageHangupSeconds {config.Link.OutageHangupSeconds} must be 1..300.");
        }
        if (config.Admin.Port is < 0 or > 65535)
        {
            throw new GatewayConfigException($"admin.port {config.Admin.Port} is out of range.");
        }
        if (config.Media.FifoPriority is < 0 or > 99)
        {
            throw new GatewayConfigException($"media.fifoPriority {config.Media.FifoPriority} must be 0..99.");
        }
        if (config.Media.WarmUpTicks is < 0 or > 10_000)
        {
            throw new GatewayConfigException($"media.warmUpTicks {config.Media.WarmUpTicks} must be 0..10000.");
        }
        if (config.Recording.Enabled && string.IsNullOrWhiteSpace(config.Recording.Directory))
        {
            throw new GatewayConfigException("recording.directory is required when recording.enabled is true.");
        }
        ParseLogLevel(config.Logging.Level);
    }

    private static string RequireSecret(string variable, string setting, string purpose)
    {
        if (string.IsNullOrWhiteSpace(variable))
        {
            throw new GatewayConfigException($"{setting} must name the environment variable holding {purpose}.");
        }
        return Environment.GetEnvironmentVariable(variable)
            ?? throw new GatewayConfigException($"Environment variable {variable} ({setting}) is not set; it must hold {purpose}.");
    }

    private static string? OptionalSecret(string variable, string setting)
    {
        if (string.IsNullOrWhiteSpace(variable))
        {
            return null;
        }
        string? value = Environment.GetEnvironmentVariable(variable);
        if (value is null)
        {
            Logs.Info($"[PhoneGateway] Environment variable {variable} ({setting}) is not set; that feature runs without a secret.");
        }
        return value;
    }
}
