using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.PhoneGateway.Config;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.PhoneGateway.Tests.Support;
using Xunit;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Secrets reach the gateway only as files named in its config, and a mistake in that plumbing is silent until
/// a registration fails in production, a secret sits world-readable on disk, or a password lands in a log line.</summary>
public sealed class GatewayConfigTests
{
    [Fact]
    public void ExampleFile_ParsesAndPointsAtSystemdCredentials()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "phone.example.json");
        Assert.True(File.Exists(path), $"missing {path}");
        GatewayConfig config = GatewayConfigLoader.Parse(path);
        Assert.Equal(5060, config.Sip.Port);
        Assert.Equal("none", config.Sip.PublicAddress);
        Assert.Equal(InboundPolicy.AllowAll, config.Sip.InboundPolicy);
        Assert.False(config.Recording.Enabled);
        Assert.False(config.Logging.SipDebug);
        Assert.NotEmpty(config.Sip.Registrar);
        Assert.NotEmpty(config.Sip.DestinationPrefixes);
        Assert.False(config.Sip.AllowAnyDestination);
        const string credentials = "/run/credentials/hartsyinference-phone-gateway.service/";
        Assert.StartsWith(credentials, config.Sip.PasswordFile, StringComparison.Ordinal);
        Assert.StartsWith(credentials, config.Link.TokenFile, StringComparison.Ordinal);
        Assert.StartsWith(credentials, config.Admin.TokenFile, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyObject_IsAValidLanConfigWithNoSecrets()
    {
        GatewayConfig config = JsonSerializer.Deserialize("{}", GatewayJsonContext.Default.GatewayConfig)!;
        GatewaySettings settings = GatewayConfigLoader.Resolve(config);
        Assert.Equal("/run/hartsyinference/phone.sock", settings.Config.Link.SocketPath);
        Assert.Equal(60, settings.Config.Sip.RegistrationExpirySeconds);
        Assert.Equal("", settings.SipPassword);
        Assert.Equal("", settings.LinkToken);
        Assert.Null(settings.AdminToken);
        Assert.Empty(settings.Config.Sip.DestinationPrefixes);
        Assert.False(settings.Config.Sip.AllowAnyDestination);
    }

    [Fact]
    public void AllowAnyDestination_IsReadFromTheFile()
    {
        GatewayConfig config = JsonSerializer.Deserialize("{\"sip\":{\"allowAnyDestination\":true}}", GatewayJsonContext.Default.GatewayConfig)!;
        Assert.True(GatewayConfigLoader.Resolve(config).Config.Sip.AllowAnyDestination);
    }

    [Fact]
    public void AllowAnyDestination_WithPrefixes_IsAConfigError()
    {
        GatewayConfig config = new()
        {
            Sip = new SipConfig { Registrar = "sip.example.net", Username = "u", DestinationPrefixes = ["+1555"], AllowAnyDestination = true },
        };
        GatewayConfigException ex = Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.Resolve(config));
        Assert.Contains("sip.allowAnyDestination", ex.Message, StringComparison.Ordinal);
        Assert.Contains("sip.destinationPrefixes", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SecretFiles_AreReadTrimmedAndNeverSerialized()
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        using TempSecret password = new("s3cret-value\n");
        using TempSecret link = new("link-token\r\n");
        using TempSecret admin = new("admin-token");
        GatewayConfig config = new()
        {
            Sip = new SipConfig { Registrar = "sip.example.net", Username = "u", PasswordFile = password.Path },
            Link = new LinkConfig { TokenFile = link.Path },
            Admin = new AdminConfig { TokenFile = admin.Path },
        };
        GatewaySettings settings = GatewayConfigLoader.Resolve(config);
        Assert.Equal("s3cret-value", settings.SipPassword);
        Assert.Equal("link-token", settings.LinkToken);
        Assert.Equal("admin-token", settings.AdminToken);
        string json = JsonSerializer.Serialize(settings.Config, GatewayJsonContext.Default.GatewayConfig);
        Assert.Contains(password.Path, json, StringComparison.Ordinal);
        Assert.DoesNotContain("s3cret-value", json, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("secret\n", "secret")]
    [InlineData("secret\r\n", "secret")]
    [InlineData("secret\n\n", "secret\n")]
    [InlineData("secret ", "secret ")]
    [InlineData("secret", "secret")]
    public void SecretFile_TrimsExactlyOneTrailingLineEnding(string content, string expected)
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        using TempSecret secret = new(content);
        Assert.Equal(expected, SecretFile.Read(secret.Path, "test.secretFile"));
    }

    [Theory]
    [InlineData(UnixFileMode.GroupRead)]
    [InlineData(UnixFileMode.OtherRead)]
    [InlineData(UnixFileMode.GroupWrite)]
    public void SecretFile_OpenToGroupOrOthers_IsRefused(UnixFileMode extra)
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        using TempSecret secret = new("secret", UnixFileMode.UserRead | UnixFileMode.UserWrite | extra);
        GatewayConfigException ex = Assert.Throws<GatewayConfigException>(() => SecretFile.Read(secret.Path, "admin.tokenFile"));
        Assert.Contains("admin.tokenFile", ex.Message, StringComparison.Ordinal);
        Assert.Contains("group or others", ex.Message, StringComparison.Ordinal);
        Assert.Contains("chmod 600", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("secret\n", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SecretFile_OwnerReadOnly_IsAccepted()
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        using TempSecret secret = new("secret", UnixFileMode.UserRead);
        Assert.Equal("secret", SecretFile.Read(secret.Path, "test.secretFile"));
    }

    [Fact]
    public void MissingSecretFile_IsAConfigErrorNamingTheSettingAndPath()
    {
        string missing = Path.Combine(Path.GetTempPath(), "hartsy-missing-" + Guid.NewGuid().ToString("N"));
        GatewayConfig config = new() { Sip = new SipConfig { Registrar = "sip.example.net", Username = "u", PasswordFile = missing } };
        GatewayConfigException ex = Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.Resolve(config));
        Assert.Contains("sip.passwordFile", ex.Message, StringComparison.Ordinal);
        Assert.Contains(missing, ex.Message, StringComparison.Ordinal);
        Assert.Contains("does not exist", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DestinationPrefixes_WithoutARegistrar_IsAConfigError()
    {
        GatewayConfig config = new() { Sip = new SipConfig { DestinationPrefixes = ["+1555"] } };
        GatewayConfigException ex = Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.Resolve(config));
        Assert.Contains("sip.destinationPrefixes", ex.Message, StringComparison.Ordinal);
        Assert.Contains("sip.registrar", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Registrar_WithoutAPasswordFile_IsAConfigError()
    {
        GatewayConfig config = new() { Sip = new SipConfig { Registrar = "sip.example.net", Username = "u" } };
        GatewayConfigException ex = Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.Resolve(config));
        Assert.Contains("sip.passwordFile", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyOrRelativeSecretFile_IsRefused()
    {
        GatewayConfig relative = new() { Link = new LinkConfig { TokenFile = "secrets/link-token" } };
        GatewayConfigException ex = Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.Resolve(relative));
        Assert.Contains("link.tokenFile", ex.Message, StringComparison.Ordinal);
        Assert.Contains("absolute", ex.Message, StringComparison.Ordinal);
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        using TempSecret empty = new("\n");
        GatewayConfigException emptyEx = Assert.Throws<GatewayConfigException>(() => SecretFile.Read(empty.Path, "link.tokenFile"));
        Assert.Contains("is empty", emptyEx.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("stun:example.com:99999")]
    [InlineData("not-an-address")]
    public void BadPublicAddress_IsRejectedWithTheSettingName(string value)
    {
        GatewayConfig config = new() { Sip = new SipConfig { PublicAddress = value } };
        GatewayConfigException ex = Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.Resolve(config));
        Assert.Contains("publicAddress", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PublicAddressResolver_ParsesTheThreeForms()
    {
        Assert.Equal(PublicAddressMode.None, PublicAddressResolver.Parse("none").Mode);
        Assert.Equal(PublicAddressMode.None, PublicAddressResolver.Parse("").Mode);
        PublicAddressResolver literal = PublicAddressResolver.Parse("203.0.113.4");
        Assert.Equal(PublicAddressMode.Literal, literal.Mode);
        Assert.Equal("203.0.113.4", literal.Resolve()!.ToString());
        Assert.Equal(PublicAddressMode.Stun, PublicAddressResolver.Parse("stun:stun.example.net:3478").Mode);
        Assert.Equal(PublicAddressMode.Stun, PublicAddressResolver.Parse("stun:stun.example.net").Mode);
    }

    [Fact]
    public void RegistrationExpiry_OutsideSixtyToOneTwenty_IsRejected()
    {
        GatewayConfig config = new() { Sip = new SipConfig { RegistrationExpirySeconds = 30 } };
        Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.Resolve(config));
    }

    [Fact]
    public void LogLevel_Parses()
    {
        Assert.Equal(LogLevel.Debug, GatewayConfigLoader.ParseLogLevel("debug"));
        Assert.Equal(LogLevel.Warning, GatewayConfigLoader.ParseLogLevel("Warn"));
        Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.ParseLogLevel("loud"));
    }
}
