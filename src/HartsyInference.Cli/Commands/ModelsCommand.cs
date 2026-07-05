using HartsyInference.Cli.Infra;
using HartsyInference.ModelHandler.Registry;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Shows the models that have been downloaded into the local cache.</summary>
public sealed class ModelsCommand : Command<ModelsCommand.Settings>
{
    /// <summary>Options for <c>hartsy models</c> (none yet).</summary>
    public sealed class Settings : CommandSettings
    {
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        ModelCacheStore cache = new ModelCacheStore();
        AnsiConsole.MarkupLine($"[grey]cache:[/] [{CliTheme.Accent}]{Markup.Escape(cache.CacheDirectory)}[/]");

        List<string> ids = cache.CachedModelIds.OrderBy(id => id, StringComparer.OrdinalIgnoreCase).ToList();
        if (ids.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No models cached yet.[/] Download one with [mediumpurple2]hartsy pull <repo-or-path>[/].");
            return 0;
        }

        Table table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
        table.AddColumn($"[{CliTheme.Accent}]model[/]");
        table.AddColumn("[grey]architecture[/]");
        table.AddColumn("[grey]size[/]");
        table.AddColumn("[grey]path[/]");

        foreach (string id in ids)
        {
            ModelInfo? info = cache.Get(id);
            string arch = info?.Architecture is { Length: > 0 } a ? Markup.Escape(a) : "[grey]?[/]";
            string size = info is not null ? FormatBytes(info.FileSize) : "[grey]?[/]";
            string path = info?.LocalPath is { Length: > 0 } p ? Markup.Escape(p) : "[grey]?[/]";
            table.AddRow($"[{CliTheme.Accent}]{Markup.Escape(id)}[/]", arch, size, $"[grey]{path}[/]");
        }

        AnsiConsole.Write(table);
        return 0;
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes <= 0)
            return "[grey]?[/]";
        string[] units = { "B", "KB", "MB", "GB", "TB" };
        double value = bytes;
        int unit = 0;
        while (value >= 1024 && unit < units.Length - 1)
        {
            value /= 1024;
            unit++;
        }
        return $"{value:0.#} {units[unit]}";
    }
}
