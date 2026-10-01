using System.Globalization;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.PhoneLink;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Tools;

namespace HartsyInference.VoiceHost.Config;

/// <summary>Reads <c>voice.json</c>, validates it, reads the link token from the secret file it names
/// (<see cref="SecretFile"/>) and builds the session options. The token is read once, here, and never logged; a problem
/// with any setting is reported by its JSON path.</summary>
public static class VoiceHostConfigLoader
{
    /// <summary>Where the systemd unit's host reads its configuration.</summary>
    public const string DefaultPath = "/etc/hartsyinference/voice.json";

    // sun_path is 108 bytes on Linux, one of them the terminating NUL.
    private const int MaxSocketPathBytes = 107;
    private const UnixFileMode OwnerReadWrite = UnixFileMode.UserRead | UnixFileMode.UserWrite;
    private const UnixFileMode WidestSocketMode = OwnerReadWrite | UnixFileMode.GroupRead | UnixFileMode.GroupWrite;

    /// <summary>Loads, validates and resolves the file at <paramref name="path"/>.</summary>
    /// <exception cref="VoiceHostConfigException">The file, or the token file it names, is missing, malformed or open to
    /// group or others.</exception>
    public static VoiceHostSettings Load(string path) => Resolve(Parse(path));

    /// <summary>Reads and validates the file at <paramref name="path"/> without opening the token file.</summary>
    /// <exception cref="VoiceHostConfigException">The file is missing, malformed or holds an invalid setting.</exception>
    public static VoiceHostConfig Parse(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new VoiceHostConfigException($"Config file {path} not found. Copy voice.example.json next to the host and pass --config <path>.");
        }
        VoiceHostConfig config;
        try
        {
            using FileStream stream = File.OpenRead(path);
            config = JsonSerializer.Deserialize(stream, VoiceHostJsonContext.Default.VoiceHostConfig)
                ?? throw new VoiceHostConfigException($"Config file {path} is JSON null.");
        }
        catch (JsonException ex)
        {
            throw new VoiceHostConfigException($"Config file {path} is not valid: {ex.Message}", ex);
        }
        Validate(config);
        return config;
    }

    /// <summary>Validates <paramref name="config"/>, reads its token file and builds the session options.</summary>
    public static VoiceHostSettings Resolve(VoiceHostConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        VoiceAgentOptions agent = Validate(config);
        string token = config.Link.TokenFile.Length == 0
            ? ""
            : SecretFile.Read(config.Link.TokenFile, "link.tokenFile", static (message, cause) => cause is null
                ? new VoiceHostConfigException(message)
                : new VoiceHostConfigException(message, cause));
        return new VoiceHostSettings { Config = config, LinkToken = token, Agent = agent, SocketMode = ParseSocketMode(config.Link.SocketMode) };
    }

    /// <summary>Parses <c>logging.level</c>: Verbose, Debug, Info, Warning or Error, any case.</summary>
    public static LogLevel ParseLogLevel(string level)
    {
        ArgumentNullException.ThrowIfNull(level);
        string trimmed = level.Trim();
        if (trimmed.Length > 0 && char.IsLetter(trimmed[0]) && Enum.TryParse(trimmed, ignoreCase: true, out LogLevel parsed))
        {
            return parsed;
        }
        throw new VoiceHostConfigException($"logging.level '{level}' must be Verbose, Debug, Info, Warning or Error.");
    }

    /// <summary>Parses <c>link.socketMode</c>, an octal mode between 0600 and 0660.</summary>
    internal static UnixFileMode ParseSocketMode(string mode)
    {
        ArgumentNullException.ThrowIfNull(mode);
        if (!int.TryParse(mode.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out int digits) || !IsOctal(digits))
        {
            throw new VoiceHostConfigException($"link.socketMode '{mode}' is not an octal file mode such as 0660.");
        }
        UnixFileMode parsed = (UnixFileMode)Convert.ToInt32(digits.ToString(CultureInfo.InvariantCulture), 8);
        if ((parsed & ~WidestSocketMode) != 0 || (parsed & OwnerReadWrite) != OwnerReadWrite)
        {
            throw new VoiceHostConfigException($"link.socketMode '{mode}' must give the owner read and write and nothing beyond 0660.");
        }
        return parsed;
    }

    private static bool IsOctal(int digits)
    {
        for (int rest = digits; rest > 0; rest /= 10)
        {
            if (rest % 10 > 7)
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>Checks every section and returns the session options the models and agent sections describe.</summary>
    private static VoiceAgentOptions Validate(VoiceHostConfig config)
    {
        HostLinkConfig link = config.Link ?? throw new VoiceHostConfigException("link must be an object.");
        HostModelsConfig models = config.Models ?? throw new VoiceHostConfigException("models must be an object.");
        HostAgentConfig agent = config.Agent ?? throw new VoiceHostConfigException("agent must be an object.");
        HostToolsConfig tools = config.Tools ?? throw new VoiceHostConfigException("tools must be an object.");
        HostEngineConfig engine = config.Engine ?? throw new VoiceHostConfigException("engine must be an object.");
        HostLoggingConfig logging = config.Logging ?? throw new VoiceHostConfigException("logging must be an object.");
        if (string.IsNullOrWhiteSpace(link.SocketPath) || !Path.IsPathFullyQualified(link.SocketPath))
        {
            throw new VoiceHostConfigException($"link.socketPath '{link.SocketPath}' must be an absolute path.");
        }
        Require(Encoding.UTF8.GetByteCount(link.SocketPath) <= MaxSocketPathBytes,
            $"link.socketPath '{link.SocketPath}' is longer than the {MaxSocketPathBytes} bytes a Unix socket path can hold.");
        ParseSocketMode(link.SocketMode);
        RequireAbsoluteOrEmpty(link.TokenFile, "link.tokenFile");
        Require(link.PrebufferMs is >= 0 and <= 200, $"link.prebufferMs {link.PrebufferMs} must be 0..200.");
        Require(link.LivenessTimeoutSeconds is >= 5 and <= 300, $"link.livenessTimeoutSeconds {link.LivenessTimeoutSeconds} must be 5..300.");
        Require(BackendFactory.IsValidSelector(models.AudioDevice), $"models.audioDevice '{models.AudioDevice}' is not a device such as cuda:1.");
        Require(BackendFactory.IsValidSelector(models.LlmDevice), $"models.llmDevice '{models.LlmDevice}' is not a device such as cuda:0.");
        Require(!BackendFactory.Kind(models.AudioDevice).Equals("auto", StringComparison.Ordinal),
            "models.audioDevice must name a device (cuda:1, cpu), not auto: the model set checks the engine runs exactly there.");
        Require(models.WakeModelRoot is null || Path.IsPathFullyQualified(models.WakeModelRoot),
            $"models.wakeModelRoot '{models.WakeModelRoot}' must be an absolute path, or null for the models root's audio/wake.");
        Require(LinkProtocol.IsOutboundSampleRate((uint)Math.Max(0, agent.OutboundSampleRate)),
            $"agent.outboundSampleRate {agent.OutboundSampleRate} must be one of 8000, 16000, 22050, 24000, 48000.");
        string[] enabled = tools.Enabled ?? throw new VoiceHostConfigException("tools.enabled must be a list of tool names.");
        foreach (string name in enabled)
        {
            Require(VoiceHostTools.Names.Contains(name, StringComparer.Ordinal),
                $"tools.enabled names '{name}'; the host offers {string.Join(", ", VoiceHostTools.Names)}.");
        }
        Require(enabled.Distinct(StringComparer.Ordinal).Count() == enabled.Length, "tools.enabled lists a tool twice.");
        Require(tools.TimeoutMs is >= 100 and <= 120_000, $"tools.timeoutMs {tools.TimeoutMs} must be 100..120000.");
        Require(engine.CpuThreadCap is >= 0 and <= 1024, $"engine.cpuThreadCap {engine.CpuThreadCap} must be 0..1024.");
        RequireAbsoluteOrEmpty(engine.SettingsFile, "engine.settingsFile");
        Require(engine.SettingsFile.Length == 0 || File.Exists(engine.SettingsFile), $"engine.settingsFile '{engine.SettingsFile}' does not exist.");
        Require(agent.ResumeApology is not null, "agent.resumeApology must be a string; empty says nothing.");
        ParseLogLevel(logging.Level);
        VoiceAgentOptions options = new()
        {
            LlmModel = models.LlmModel,
            LlmDevice = models.LlmDevice,
            AudioDevice = models.AudioDevice,
            SttModel = models.SttModel,
            TtsModel = models.TtsModel,
            Denoise = models.Denoise,
            OutboundSampleRate = agent.OutboundSampleRate,
            EndOfTurnSilenceMs = agent.EndOfTurnSilenceMs,
            MaxUtteranceMs = agent.MaxUtteranceMs,
            BargeInEnabled = agent.BargeInEnabled,
            BargeInProbability = agent.BargeInProbability,
            BargeInMinMs = agent.BargeInMinMs,
            BargeInHoldoffMs = agent.BargeInHoldoffMs,
            SystemPrompt = agent.SystemPrompt ?? VoiceAgentOptions.DefaultSystemPrompt,
            MaxToolRoundsPerTurn = agent.MaxToolRoundsPerTurn,
            MaxHistoryTokens = agent.MaxHistoryTokens,
            MaxReplyTokens = agent.MaxReplyTokens,
            FirstSentenceMinChars = agent.FirstSentenceMinChars,
            MaxSentenceChars = agent.MaxSentenceChars,
            CpuThreadCap = engine.CpuThreadCap,
        };
        try
        {
            options.Validate();
        }
        catch (ArgumentException ex)
        {
            throw new VoiceHostConfigException($"{SettingOf(ex.ParamName)}: {ex.Message}", ex);
        }
        return options;
    }

    /// <summary>The JSON path of the <see cref="VoiceAgentOptions"/> property a validation error names.</summary>
    private static string SettingOf(string? option)
    {
        if (string.IsNullOrEmpty(option))
        {
            return "agent";
        }
        string section = option switch
        {
            nameof(VoiceAgentOptions.LlmModel) or nameof(VoiceAgentOptions.LlmDevice) or nameof(VoiceAgentOptions.AudioDevice)
                or nameof(VoiceAgentOptions.SttModel) or nameof(VoiceAgentOptions.TtsModel) => "models",
            nameof(VoiceAgentOptions.CpuThreadCap) => "engine",
            _ => "agent",
        };
        return section + "." + char.ToLowerInvariant(option[0]) + option[1..];
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new VoiceHostConfigException(message);
        }
    }

    private static void RequireAbsoluteOrEmpty(string path, string setting)
    {
        if (path is null)
        {
            throw new VoiceHostConfigException($"{setting} must be a string; empty means none.");
        }
        if (path.Length > 0 && !Path.IsPathFullyQualified(path))
        {
            throw new VoiceHostConfigException($"{setting} '{path}' must be an absolute path.");
        }
    }
}
