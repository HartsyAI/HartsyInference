using System.Text;

namespace HartsyInference.PhoneGateway.Config;

/// <summary>Reads a deployment secret (SIP password, link or admin token) from a file, once, at start-up.</summary>
/// <remarks>Secrets live in files rather than in the config or the environment, which leaks through
/// <c>/proc/&lt;pid&gt;/environ</c>, child processes and crash dumps. The ssh rule applies: a file that group or others
/// can access is refused (0600 and 0400 are fine). Under systemd the file is a credential: <c>LoadCredential=</c>
/// places it at <c>/run/credentials/&lt;unit&gt;/&lt;name&gt;</c>, readable only by the service. One trailing line
/// ending is trimmed, since editors add one; the value itself is never logged.</remarks>
internal static class SecretFile
{
    /// <summary>Largest secret file accepted; anything bigger is almost certainly the wrong path.</summary>
    public const int MaxBytes = 4096;

    private const UnixFileMode GroupOrOther =
        UnixFileMode.GroupRead | UnixFileMode.GroupWrite | UnixFileMode.GroupExecute
        | UnixFileMode.OtherRead | UnixFileMode.OtherWrite | UnixFileMode.OtherExecute;

    /// <summary>Reads the secret at <paramref name="path"/>; <paramref name="setting"/> names the config field in errors.</summary>
    /// <exception cref="GatewayConfigException">The path is relative, or the file is missing, open to group or others,
    /// too large, unreadable or empty.</exception>
    public static string Read(string path, string setting)
    {
        ArgumentNullException.ThrowIfNull(path);
        if (!Path.IsPathFullyQualified(path))
        {
            throw new GatewayConfigException($"{setting} '{path}' must be an absolute path.");
        }
        FileInfo info = new(path);
        if (!info.Exists)
        {
            throw new GatewayConfigException($"{setting}: secret file {path} does not exist.");
        }
        if (!OperatingSystem.IsWindows())
        {
            UnixFileMode mode = File.GetUnixFileMode(path);
            if ((mode & GroupOrOther) != 0)
            {
                throw new GatewayConfigException(
                    $"{setting}: secret file {path} is accessible by group or others (mode 0{Convert.ToString((int)mode, 8)}); "
                    + $"restrict it to its owner, e.g. chmod 600 {path}.");
            }
        }
        if (info.Length > MaxBytes)
        {
            throw new GatewayConfigException($"{setting}: secret file {path} is {info.Length} bytes; a secret is at most {MaxBytes}.");
        }
        string value;
        try
        {
            value = File.ReadAllText(path, Encoding.UTF8);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new GatewayConfigException($"{setting}: secret file {path} cannot be read: {ex.Message}", ex);
        }
        value = TrimOneLineEnding(value);
        if (value.Length == 0)
        {
            throw new GatewayConfigException($"{setting}: secret file {path} is empty.");
        }
        return value;
    }

    /// <summary>Drops one trailing LF or CRLF and nothing else, since a secret may end in other whitespace.</summary>
    internal static string TrimOneLineEnding(string value) =>
        value.EndsWith("\r\n", StringComparison.Ordinal) ? value[..^2]
        : value.EndsWith('\n') ? value[..^1]
        : value;
}
