using System.Text;

namespace HartsyInference.Core.Configuration;

/// <summary>Reads a deployment secret (a password or a shared token) from a file, once, at start-up.</summary>
/// <remarks>Secrets live in files rather than in a config file or the environment, which leaks through
/// <c>/proc/&lt;pid&gt;/environ</c>, child processes and crash dumps. The ssh rule applies: a file that group or others
/// can access is refused (0600 and 0400 are fine). Under systemd the file is a credential: <c>LoadCredential=</c>
/// places it at <c>/run/credentials/&lt;unit&gt;/&lt;name&gt;</c>, readable only by the service. One trailing line
/// ending is trimmed, since editors add one; the value itself is never logged or put in an error message. Errors are
/// built by the caller's factory, so each host reports them as its own configuration error type.</remarks>
public static class SecretFile
{
    /// <summary>Largest secret file accepted; anything bigger is almost certainly the wrong path.</summary>
    public const int MaxBytes = 4096;

    private const UnixFileMode GroupOrOther =
        UnixFileMode.GroupRead | UnixFileMode.GroupWrite | UnixFileMode.GroupExecute
        | UnixFileMode.OtherRead | UnixFileMode.OtherWrite | UnixFileMode.OtherExecute;

    /// <summary>Reads the secret at <paramref name="path"/>; <paramref name="setting"/> names the config field in errors.</summary>
    /// <param name="path">Absolute path of the secret file.</param>
    /// <param name="setting">The configuration field that named the file, for error messages.</param>
    /// <param name="error">Builds the exception thrown for an unusable file from a message and an optional cause.</param>
    /// <returns>The secret, with one trailing LF or CRLF removed.</returns>
    public static string Read(string path, string setting, Func<string, Exception?, Exception> error)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentException.ThrowIfNullOrWhiteSpace(setting);
        ArgumentNullException.ThrowIfNull(error);
        if (!Path.IsPathFullyQualified(path))
        {
            throw error($"{setting} '{path}' must be an absolute path.", null);
        }
        if (!File.Exists(path))
        {
            throw error($"{setting}: secret file {path} does not exist.", null);
        }
        string value;
        try
        {
            // One handle for the mode check, the size check and the read, so the file checked is the file read.
            using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            if (!OperatingSystem.IsWindows())
            {
                UnixFileMode mode = File.GetUnixFileMode(stream.SafeFileHandle);
                if ((mode & GroupOrOther) != 0)
                {
                    throw error(
                        $"{setting}: secret file {path} is accessible by group or others (mode 0{Convert.ToString((int)mode, 8)}); "
                        + $"restrict it to its owner, e.g. chmod 600 {path}.", null);
                }
            }
            if (stream.Length > MaxBytes)
            {
                throw error($"{setting}: secret file {path} is {stream.Length} bytes; a secret is at most {MaxBytes}.", null);
            }
            using StreamReader reader = new(stream, Encoding.UTF8);
            value = reader.ReadToEnd();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw error($"{setting}: secret file {path} cannot be read: {ex.Message}", ex);
        }
        value = TrimOneLineEnding(value);
        if (value.Length == 0)
        {
            throw error($"{setting}: secret file {path} is empty.", null);
        }
        return value;
    }

    /// <summary>Drops one trailing LF or CRLF and nothing else, since a secret may end in other whitespace.</summary>
    public static string TrimOneLineEnding(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return value.EndsWith("\r\n", StringComparison.Ordinal) ? value[..^2]
            : value.EndsWith('\n') ? value[..^1]
            : value;
    }
}
