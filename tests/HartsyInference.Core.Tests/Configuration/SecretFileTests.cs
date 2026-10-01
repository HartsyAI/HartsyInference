using HartsyInference.Core.Configuration;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>The shared secret-file rules both phone processes read their tokens with: owner-only files, one trailing line
/// ending trimmed, and every refusal built by the caller's own exception factory with the setting named and the secret
/// never quoted.</summary>
public sealed class SecretFileTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), "hartsy-secret-" + Guid.NewGuid().ToString("N"));

    public void Dispose() => File.Delete(_path);

    [Fact]
    public void AnOwnerOnlyFileIsReadWithOneLineEndingTrimmed()
    {
        Write("token-value\n\n", UnixFileMode.UserRead | UnixFileMode.UserWrite);
        Assert.Equal("token-value\n", SecretFile.Read(_path, "link.tokenFile", Error));
    }

    [Fact]
    public void AFileOthersCanReadIsRefusedThroughTheCallersFactory()
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        Write("token-value", UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.OtherRead);
        ConfigError ex = Assert.Throws<ConfigError>(() => SecretFile.Read(_path, "link.tokenFile", Error));
        Assert.Contains("link.tokenFile", ex.Message, StringComparison.Ordinal);
        Assert.Contains("chmod 600", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("token-value", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingEmptyAndRelativePathsAreRefused()
    {
        Assert.Contains("does not exist", Assert.Throws<ConfigError>(() => SecretFile.Read(_path, "link.tokenFile", Error)).Message,
            StringComparison.Ordinal);
        Write("\n", UnixFileMode.UserRead | UnixFileMode.UserWrite);
        Assert.Contains("is empty", Assert.Throws<ConfigError>(() => SecretFile.Read(_path, "link.tokenFile", Error)).Message,
            StringComparison.Ordinal);
        Assert.Contains("absolute", Assert.Throws<ConfigError>(() => SecretFile.Read("relative/token", "link.tokenFile", Error)).Message,
            StringComparison.Ordinal);
    }

    private static Exception Error(string message, Exception? cause) => new ConfigError(message, cause);

    private void Write(string content, UnixFileMode mode)
    {
        File.WriteAllText(_path, content);
        if (!OperatingSystem.IsWindows())
        {
            File.SetUnixFileMode(_path, mode);
        }
    }

    private sealed class ConfigError(string message, Exception? cause) : Exception(message, cause);
}
