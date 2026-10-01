using HartsyInference.Core.Logging;
using HartsyInference.VoiceHost;
using HartsyInference.VoiceHost.Config;

namespace HartsyInference.VoiceHost.TestHost;

/// <summary>The shipping voice host composition (engine, speech models, socket, sender) with <see cref="ScriptedReplyService"/>
/// as the language model: <c>--config &lt;voice.json&gt; --reply &lt;text&gt;</c>. The loopback tests run it as a process so they
/// can kill it with SIGKILL, which no in-process host can stand in for.</summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        string? config = null;
        string reply = "Okay.";
        for (int i = 0; i + 1 < args.Length; i += 2)
        {
            switch (args[i])
            {
                case "--config":
                    config = args[i + 1];
                    break;
                case "--reply":
                    reply = args[i + 1];
                    break;
                default:
                    Console.Error.WriteLine($"unknown argument '{args[i]}'");
                    return 2;
            }
        }
        if (config is null)
        {
            Console.Error.WriteLine("usage: HartsyInference.VoiceHost.TestHost --config <voice.json> [--reply <text>]");
            return 2;
        }
        VoiceHostSettings settings = VoiceHostConfigLoader.Load(config);
        Logs.MinLevel = VoiceHostConfigLoader.ParseLogLevel(settings.Config.Logging.Level);
        return await VoiceHostProgram.RunAsync(settings, _ => new ScriptedReplyService(reply)).ConfigureAwait(false);
    }
}
