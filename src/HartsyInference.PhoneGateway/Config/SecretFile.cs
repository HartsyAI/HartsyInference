using CoreSecretFile = HartsyInference.Core.Configuration.SecretFile;

namespace HartsyInference.PhoneGateway.Config;

/// <summary>The gateway's secrets (SIP password, link and admin tokens) through the shared
/// <see cref="CoreSecretFile"/> rules, reported as <see cref="GatewayConfigException"/>.</summary>
internal static class SecretFile
{
    /// <summary>Reads the secret at <paramref name="path"/>; <paramref name="setting"/> names the config field in errors.</summary>
    /// <exception cref="GatewayConfigException">The path is relative, or the file is missing, open to group or others,
    /// too large, unreadable or empty.</exception>
    public static string Read(string path, string setting) =>
        CoreSecretFile.Read(path, setting, static (message, cause) => cause is null
            ? new GatewayConfigException(message)
            : new GatewayConfigException(message, cause));
}
