using HartsyInference.Cli.Commands;
using HartsyInference.Core.Configuration;
using Spectre.Console.Cli;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary><c>hartsy settings get</c> and <c>set</c> on a removed id warn and succeed without touching the settings file,
/// which the engine inside SwarmUI shares.</summary>
public sealed class SettingsRemovedIdTests
{
    private static CommandApp App()
    {
        CommandApp app = new();
        app.Configure(config =>
        {
            config.PropagateExceptions();
            config.Settings.StrictParsing = true;
            config.AddBranch("settings", settings =>
            {
                settings.AddCommand<SettingsGetCommand>("get");
                settings.AddCommand<SettingsSetCommand>("set");
            });
        });
        return app;
    }

    [Fact]
    public void GetAndSet_OnARemovedId_WarnAndSucceed()
    {
        string[] removed = [.. KnobDeprecations.All.Where(e => e.NewId is null).Select(e => e.OldId)];
        Assert.NotEmpty(removed);
        string dir = Path.Combine(Path.GetTempPath(), "hartsy-cli-removed-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string file = Path.Combine(dir, "settings.json");
        const string Contents = """{"settings":{}}""";
        File.WriteAllText(file, Contents);
        string? previous = KnobFile.ExplicitPath;
        KnobFile.ExplicitPath = file;
        try
        {
            foreach (string id in removed)
            {
                Assert.Equal(0, App().Run(["settings", "get", id]));
                Assert.Equal(0, App().Run(["settings", "set", id, "true"]));
            }
            Assert.Equal(Contents, File.ReadAllText(file));
        }
        finally
        {
            KnobFile.ExplicitPath = previous;
            Directory.Delete(dir, recursive: true);
        }
    }
}
