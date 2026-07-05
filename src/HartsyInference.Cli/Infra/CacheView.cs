using HartsyInference.ModelHandler.Registry;
using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Renders the local model cache, shared by <c>hartsy models</c> and the REPL's <c>/models</c>.</summary>
public static class CacheView
{
    /// <summary>Prints the cached models table (or an empty-state hint) and returns the count.</summary>
    public static int Render()
    {
        ModelCacheStore cache = new ModelCacheStore();
        AnsiConsole.MarkupLine($"[grey]cache:[/] [{CliTheme.Accent}]{Markup.Escape(cache.CacheDirectory)}[/]");

        List<string> ids = cache.CachedModelIds.OrderBy(id => id, StringComparer.OrdinalIgnoreCase).ToList();
        if (ids.Count == 0)
        {
            AnsiConsole.MarkupLine($"[yellow]No models cached yet.[/] Download one with [{CliTheme.Accent}]hartsy pull <repo-or-path>[/].");
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
        return ids.Count;
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
