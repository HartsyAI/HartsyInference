using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.PhoneGateway.Config;
using HartsyInference.PhoneGateway.Sip;
using Xunit;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Secrets reach the gateway by environment-variable name only, and a mistake in that plumbing is silent
/// until a registration fails in production or a password lands in a log line.</summary>
public sealed class GatewayConfigTests
{
    [Fact]
    public void ExampleFile_LoadsAndValidates()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "phone.example.json");
        Assert.True(File.Exists(path), $"missing {path}");
        GatewaySettings settings = GatewayConfigLoader.Load(path);
        Assert.Equal(5060, settings.Config.Sip.Port);
        Assert.Equal("none", settings.Config.Sip.PublicAddress);
        Assert.Equal(InboundPolicy.AllowAll, settings.Config.Sip.InboundPolicy);
        Assert.Equal("", settings.SipPassword);
        Assert.False(settings.Config.Recording.Enabled);
        Assert.False(settings.Config.Logging.SipDebug);
    }

    [Fact]
    public void EmptyObject_IsAValidLanConfig()
    {
        GatewayConfig config = JsonSerializer.Deserialize("{}", GatewayJsonContext.Default.GatewayConfig)!;
        GatewaySettings settings = GatewayConfigLoader.Resolve(config);
        Assert.Equal("/run/hartsyinference/phone.sock", settings.Config.Link.SocketPath);
        Assert.Equal(60, settings.Config.Sip.RegistrationExpirySeconds);
    }

    [Fact]
    public void Registrar_WithoutThePasswordVariable_FailsNamingTheVariable()
    {
        string variable = "HARTSY_TEST_SIP_PW_" + Guid.NewGuid().ToString("N")[..8];
        GatewayConfig config = new() { Sip = new SipConfig { Registrar = "sip.example.net", Username = "u", PasswordEnv = variable } };
        GatewayConfigException ex = Assert.Throws<GatewayConfigException>(() => GatewayConfigLoader.Resolve(config));
        Assert.Contains(variable, ex.Message, StringComparison.Ordinal);
        Assert.Contains("sip.passwordEnv", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Registrar_WithThePasswordVariable_ResolvesItAndNeverSerializesIt()
    {
        string variable = "HARTSY_TEST_SIP_PW_" + Guid.NewGuid().ToString("N")[..8];
        Environment.SetEnvironmentVariable(variable, "s3cret-value");
        try
        {
            GatewayConfig config = new() { Sip = new SipConfig { Registrar = "sip.example.net", Username = "u", PasswordEnv = variable } };
            GatewaySettings settings = GatewayConfigLoader.Resolve(config);
            Assert.Equal("s3cret-value", settings.SipPassword);
            string json = JsonSerializer.Serialize(settings.Config, GatewayJsonContext.Default.GatewayConfig);
            Assert.Contains("passwordEnv", json, StringComparison.Ordinal);
            Assert.DoesNotContain("s3cret-value", json, StringComparison.Ordinal);
        }
        finally
        {
            Environment.SetEnvironmentVariable(variable, null);
        }
    }

    [Fact]
    public void UnsetOptionalTokens_DisableTheFeaturesInsteadOfFailing()
    {
        string variable = "HARTSY_TEST_TOKEN_" + Guid.NewGuid().ToString("N")[..8];
        GatewayConfig config = new() { Link = new LinkConfig { TokenEnv = variable }, Admin = new AdminConfig { TokenEnv = variable } };
        GatewaySettings settings = GatewayConfigLoader.Resolve(config);
        Assert.Equal("", settings.LinkToken);
        Assert.Null(settings.AdminToken);
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
