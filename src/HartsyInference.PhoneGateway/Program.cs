using System.Runtime.InteropServices;
using HartsyInference.Core.Logging;
using HartsyInference.PhoneGateway.Config;

namespace HartsyInference.PhoneGateway;

/// <summary>Entry point: <c>hartsy-phone-gateway --config /etc/hartsyinference/phone.json</c>. Runs until SIGTERM or
/// Ctrl+C. Exit code 2 means the configuration could not be used (the message names the setting or variable).</summary>
public static class Program
{
    private const string DefaultConfigPath = "/etc/hartsyinference/phone.json";

    public static int Main(string[] args)
    {
        string configPath = DefaultConfigPath;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--config" when i + 1 < args.Length:
                    configPath = args[++i];
                    break;
                case "--help" or "-h":
                    Console.WriteLine("usage: HartsyInference.PhoneGateway [--config <phone.json>]");
                    Console.WriteLine($"  default config: {DefaultConfigPath}; a template is phone.example.json next to the binary.");
                    return 0;
                default:
                    Console.Error.WriteLine($"unknown argument '{args[i]}'; see --help");
                    return 2;
            }
        }
        GatewaySettings settings;
        try
        {
            settings = GatewayConfigLoader.Load(configPath);
        }
        catch (GatewayConfigException ex)
        {
            Console.Error.WriteLine($"config error: {ex.Message}");
            return 2;
        }
        Logs.MinLevel = GatewayConfigLoader.ParseLogLevel(settings.Config.Logging.Level);
        using ManualResetEventSlim exit = new(false);
        using PosixSignalRegistration term = PosixSignalRegistration.Create(PosixSignal.SIGTERM, ctx =>
        {
            ctx.Cancel = true;
            exit.Set();
        });
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            exit.Set();
        };
        using GatewayHost host = new(settings);
        try
        {
            host.Start();
        }
        catch (Exception ex)
        {
            Logs.Error("[PhoneGateway] Start failed", ex);
            return 1;
        }
        exit.Wait();
        host.Stop();
        return 0;
    }
}
