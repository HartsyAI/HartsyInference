using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Config;
using HartsyInference.VoiceHost.Tools;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary><c>voice.json</c>: an empty file is the session defaults, the shipped example loads, the link token comes from
/// an owner-only secret file and is never printed, a file others can read is refused naming the setting and the fix, and
/// every invalid or misspelt setting fails at load naming its JSON path.</summary>
public sealed class VoiceHostConfigTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "hvh-config-" + Guid.NewGuid().ToString("N")[..10]);

    public VoiceHostConfigTests() => Directory.CreateDirectory(_directory);

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void AnEmptyFileIsTheSessionDefaultsWithNoToken()
    {
        VoiceHostSettings settings = VoiceHostConfigLoader.Load(WriteConfig("{}"));
        VoiceAgentOptions defaults = new();

        Assert.Equal(defaults, settings.Agent);
        Assert.Equal("", settings.LinkToken);
        Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.GroupRead | UnixFileMode.GroupWrite, settings.SocketMode);
        Assert.Equal("/run/hartsyinference/phone.sock", settings.Config.Link.SocketPath);
        Assert.Equal(VoiceHostTools.Names, settings.Config.Tools.Enabled);
        Assert.Equal(LogLevel.Info, VoiceHostConfigLoader.ParseLogLevel(settings.Config.Logging.Level));
    }

    [Fact]
    public void TheShippedExampleParsesAndMapsOntoTheSessionOptions()
    {
        string example = Path.Combine(AppContext.BaseDirectory, "voice.example.json");
        VoiceHostConfig config = VoiceHostConfigLoader.Parse(example);
        config.Link.TokenFile = "";
        VoiceHostSettings settings = VoiceHostConfigLoader.Resolve(config);

        Assert.Equal(14, settings.Agent.CpuThreadCap);
        Assert.Equal("cuda:1", settings.Agent.AudioDevice);
        Assert.Equal("cuda:0", settings.Agent.LlmDevice);
        Assert.Equal("qwen3", settings.Agent.LlmModel);
        Assert.Equal(16_000, settings.Agent.OutboundSampleRate);
        Assert.Equal(VoiceAgentOptions.DefaultSystemPrompt, settings.Agent.SystemPrompt);
        Assert.Equal("Hello, how can I help you?", config.Agent.Greeting);
        // The template must not drift from VoiceAgentOptions' own default: a hard-coded false here silently runs
        // every install without RNNoise (voice.endpoint.ms loses the denoiser's ~40 ms) regardless of what the
        // code default is. Compared against the default, not a literal true/false, so this still catches drift
        // if the code default itself ever changes.
        Assert.Equal(new VoiceAgentOptions().Denoise, settings.Agent.Denoise);
    }

    [Fact]
    public void TheTokenComesFromAnOwnerOnlyFileAndIsNeverPrinted()
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        string token = WriteSecret("s3cret-link-token\n", UnixFileMode.UserRead | UnixFileMode.UserWrite);
        VoiceHostSettings settings = VoiceHostConfigLoader.Load(WriteConfig(TokenConfig(token)));

        Assert.Equal("s3cret-link-token", settings.LinkToken);
        Assert.DoesNotContain("s3cret", settings.ToString(), StringComparison.Ordinal);
        Assert.Contains("(set)", settings.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain("s3cret", JsonSerializer.Serialize(settings.Config, VoiceHostJsonContext.Default.VoiceHostConfig), StringComparison.Ordinal);
    }

    [Fact]
    public void ATokenFileOthersCanReadIsRefused()
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }
        string token = WriteSecret("s3cret-link-token", UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.GroupRead | UnixFileMode.OtherRead);
        VoiceHostConfigException ex = Assert.Throws<VoiceHostConfigException>(() =>
            VoiceHostConfigLoader.Load(WriteConfig(TokenConfig(token))));

        Assert.Contains("link.tokenFile", ex.Message, StringComparison.Ordinal);
        Assert.Contains("chmod 600", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("s3cret", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingTokenFileIsRefused()
    {
        string missing = Path.Combine(_directory, "no-such-token");
        VoiceHostConfigException ex = Assert.Throws<VoiceHostConfigException>(() =>
            VoiceHostConfigLoader.Load(WriteConfig(TokenConfig(missing))));
        Assert.Contains("does not exist", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"link":{"socketPath":"phone.sock"}}""", "link.socketPath")]
    [InlineData("""{"link":{"socketPath":"/run/hartsyinference/a-socket-path-that-is-far-too-long-for-sun-path-on-linux/which-holds-108-bytes/phone.sock"}}""", "link.socketPath")]
    [InlineData("""{"link":{"socketMode":"0666"}}""", "link.socketMode")]
    [InlineData("""{"link":{"socketMode":"0400"}}""", "link.socketMode")]
    [InlineData("""{"link":{"socketMode":"0680"}}""", "link.socketMode")]
    [InlineData("""{"link":{"tokenFile":"secrets/token"}}""", "link.tokenFile")]
    [InlineData("""{"link":{"prebufferMs":500}}""", "link.prebufferMs")]
    [InlineData("""{"models":{"audioDevice":"auto"}}""", "models.audioDevice")]
    [InlineData("""{"models":{"llmDevice":"tpu:0"}}""", "models.llmDevice")]
    [InlineData("""{"models":{"sttModel":" "}}""", "models.sttModel")]
    [InlineData("""{"models":{"wakeModelRoot":"audio/wake"}}""", "models.wakeModelRoot")]
    [InlineData("""{"agent":{"outboundSampleRate":11025}}""", "agent.outboundSampleRate")]
    [InlineData("""{"agent":{"endOfTurnSilenceMs":50}}""", "agent.endOfTurnSilenceMs")]
    [InlineData("""{"agent":{"bargeInProbability":1.5}}""", "agent.bargeInProbability")]
    [InlineData("""{"tools":{"enabled":["hangup","dial"]}}""", "tools.enabled")]
    [InlineData("""{"tools":{"enabled":["hangup","hangup"]}}""", "tools.enabled")]
    [InlineData("""{"tools":{"timeoutMs":5}}""", "tools.timeoutMs")]
    [InlineData("""{"engine":{"cpuThreadCap":-1}}""", "engine.cpuThreadCap")]
    [InlineData("""{"engine":{"settingsFile":"/no/such/settings.json"}}""", "engine.settingsFile")]
    [InlineData("""{"logging":{"level":"loud"}}""", "logging.level")]
    [InlineData("""{"link":{"socketPathh":"/run/x.sock"}}""", "socketPathh")]
    public void AnInvalidSettingFailsAtLoadNamingIt(string json, string setting)
    {
        VoiceHostConfigException ex = Assert.Throws<VoiceHostConfigException>(() => VoiceHostConfigLoader.Load(WriteConfig(json)));
        Assert.Contains(setting, ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("0600", UnixFileMode.UserRead | UnixFileMode.UserWrite)]
    [InlineData("660", UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.GroupRead | UnixFileMode.GroupWrite)]
    [InlineData("0640", UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.GroupRead)]
    public void TheSocketModeIsOctalBetween0600And0660(string mode, UnixFileMode expected) =>
        Assert.Equal(expected, VoiceHostConfigLoader.ParseSocketMode(mode));

    [Fact]
    public void AMissingFileSaysWhereTheTemplateIs()
    {
        VoiceHostConfigException ex = Assert.Throws<VoiceHostConfigException>(() => VoiceHostConfigLoader.Load(Path.Combine(_directory, "absent.json")));
        Assert.Contains("voice.example.json", ex.Message, StringComparison.Ordinal);
    }

    private static string TokenConfig(string tokenFile) => "{\"link\":{\"tokenFile\":\"" + tokenFile + "\"}}";

    private string WriteConfig(string json)
    {
        string path = Path.Combine(_directory, "voice-" + Guid.NewGuid().ToString("N")[..6] + ".json");
        File.WriteAllText(path, json);
        return path;
    }

    private string WriteSecret(string content, UnixFileMode mode)
    {
        string path = Path.Combine(_directory, "token-" + Guid.NewGuid().ToString("N")[..6]);
        File.WriteAllText(path, content);
        File.SetUnixFileMode(path, mode);
        return path;
    }
}
