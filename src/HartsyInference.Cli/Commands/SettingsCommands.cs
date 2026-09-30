using System.ComponentModel;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Configuration;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Prints the settings file this process reads and writes, and whether it exists yet.</summary>
public sealed class SettingsPathCommand : Command<SettingsPathCommand.Settings>
{
    /// <summary>Options for <c>hartsy settings path</c>.</summary>
    public sealed class Settings : CommandSettings
    {
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        string path = KnobFile.Path;
        AnsiConsole.MarkupLine($"[#2ea5e0]{Markup.Escape(path)}[/]");
        AnsiConsole.MarkupLine(File.Exists(path)
            ? $"[grey]exists — {KnobFile.LoadedCount} setting(s) applied[/]"
            : "[grey]does not exist yet — 'hartsy settings set <id> <value>' creates it[/]");
        return 0;
    }
}

/// <summary>Prints one setting's effective value and where that value came from.</summary>
public sealed class SettingsGetCommand : Command<SettingsGetCommand.Settings>
{
    /// <summary>Options for <c>hartsy settings get</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Dotted setting id, e.g. <c>paths.modelsRoot</c>.</summary>
        [CommandArgument(0, "<id>")]
        [Description("Dotted setting id, e.g. paths.modelsRoot.")]
        public string Id { get; init; } = "";
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (KnobRegistry.Find(settings.Id) is not { } knob)
        {
            AnsiConsole.MarkupLine($"[red]Unknown setting '{Markup.Escape(settings.Id)}'.[/] Run 'hartsy settings list'.");
            return 1;
        }
        (string id, string type, object? declared, KnobScope scope, KnobDomain domain, string summary) = KnobRegistry.Describe(knob);
        (object? effective, string source) = KnobCli.Effective(id);

        AnsiConsole.MarkupLine($"[bold]{Markup.Escape(id)}[/]  [grey]{Markup.Escape(type)} · {domain} · "
            + $"{(scope == KnobScope.Construction ? "applies at load" : "applies per run")}[/]");
        AnsiConsole.MarkupLine($"  value    [#2ea5e0]{Markup.Escape(Format(effective))}[/]  [grey](from {source})[/]");
        AnsiConsole.MarkupLine($"  default  [grey]{Markup.Escape(Format(declared))}[/]");
        AnsiConsole.MarkupLine($"  {Markup.Escape(summary)}");
        return 0;
    }

    private static string Format(object? value) => value?.ToString() ?? "—";
}

/// <summary>Writes one setting to the settings file so it survives a restart.</summary>
public sealed class SettingsSetCommand : Command<SettingsSetCommand.Settings>
{
    /// <summary>Options for <c>hartsy settings set</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Dotted setting id, e.g. <c>paths.modelsRoot</c>.</summary>
        [CommandArgument(0, "<id>")]
        [Description("Dotted setting id, e.g. paths.modelsRoot.")]
        public string Id { get; init; } = "";

        /// <summary>The value to store. Quoted as-is; parsed with the same rules the settings file uses.</summary>
        [CommandArgument(1, "<value>")]
        [Description("Value to store, parsed exactly as the settings file would parse it.")]
        public string Value { get; init; } = "";
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        object? stored;
        try
        {
            stored = KnobFile.Save(settings.Id, settings.Value);
        }
        catch (InvalidOperationException ex)
        {
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(ex.Message)}[/]");
            return 1;
        }

        AnsiConsole.MarkupLine($"[green]{Markup.Escape(settings.Id)}[/] = "
            + $"[#2ea5e0]{Markup.Escape(stored?.ToString() ?? "null")}[/]");
        AnsiConsole.MarkupLine($"[grey]written to {Markup.Escape(KnobFile.Path)}[/]");

        if (KnobRegistry.Find(settings.Id) is { } knob
            && KnobRegistry.Describe(knob).Scope == KnobScope.Construction)
        {
            AnsiConsole.MarkupLine("[yellow]This setting is read when the engine is built — restart to apply it.[/]");
        }
        return 0;
    }
}

/// <summary>Prints every declared setting, grouped by domain.</summary>
public sealed class SettingsListCommand : Command<SettingsListCommand.Settings>
{
    /// <summary>Options for <c>hartsy settings list</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Include the diagnostic and dump-directory settings, which exist for debugging rather than for tuning.</summary>
        [CommandOption("--all")]
        [Description("Include diagnostics and dump-directory settings (hidden by default — they are debugging hooks, not tuning).")]
        public bool All { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        KnobCli.ListSettings(settings.All);
        return 0;
    }
}
