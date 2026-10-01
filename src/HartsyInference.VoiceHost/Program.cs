using HartsyInference.Core.Logging;
using HartsyInference.VoiceHost.Config;

namespace HartsyInference.VoiceHost;

/// <summary>Entry point: <c>HartsyInference.VoiceHost --config /etc/hartsyinference/voice.json</c>. Runs until SIGTERM or
/// Ctrl+C. Exit code 2 means the configuration could not be used (the message names the setting), 1 that start-up or
/// the host failed.</summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        string configPath = VoiceHostConfigLoader.DefaultPath;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--config" when i + 1 < args.Length:
                    configPath = args[++i];
                    break;
                case "--help" or "-h":
                    Console.WriteLine("usage: HartsyInference.VoiceHost [--config <voice.json>]");
                    Console.WriteLine($"  default config: {VoiceHostConfigLoader.DefaultPath}; a template is voice.example.json next to the binary.");
                    return 0;
                default:
                    Console.Error.WriteLine($"unknown argument '{args[i]}'; see --help");
                    return 2;
            }
        }
        VoiceHostSettings settings;
        try
        {
            settings = VoiceHostConfigLoader.Load(configPath);
        }
        catch (VoiceHostConfigException ex)
        {
            Console.Error.WriteLine($"config error: {ex.Message}");
            return 2;
        }
        Logs.MinLevel = VoiceHostConfigLoader.ParseLogLevel(settings.Config.Logging.Level);
        return await VoiceHostProgram.RunAsync(settings, engine => engine.Text).ConfigureAwait(false);
    }
}
