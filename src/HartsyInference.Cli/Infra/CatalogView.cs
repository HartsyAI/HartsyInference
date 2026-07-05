using Spectre.Console;

namespace HartsyInference.Cli.Infra;

/// <summary>Renders the model catalog as a Spectre table, shared by <c>hartsy list</c> and the REPL's <c>/list</c>.</summary>
public static class CatalogView
{
    /// <summary>Prints the catalog filtered by <paramref name="filter"/> (null = all) and optionally to verified-only.
    /// Returns the number of rows shown.</summary>
    public static int Render(Modality? filter, bool verifiedOnly)
    {
        IEnumerable<CatalogEntry> rows = ModelCatalog.All;
        if (filter is not null)
            rows = rows.Where(e => e.Modality == filter.Value);
        if (verifiedOnly)
            rows = rows.Where(e => e.Status == ModelStatus.Verified);

        List<CatalogEntry> entries = rows.ToList();
        if (entries.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No models match that filter.[/]");
            return 0;
        }

        Table table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
        if (filter is null)
            table.AddColumn("[grey]modality[/]");
        table.AddColumn($"[{CliTheme.Accent}]model[/]");
        table.AddColumn("[grey]name[/]");
        table.AddColumn("[grey]architecture[/]");
        table.AddColumn("[grey]status[/]");

        foreach (CatalogEntry e in entries)
        {
            string id = $"[{CliTheme.Accent}]{Markup.Escape(e.Id)}[/]";
            string name = Markup.Escape(e.DisplayName);
            string arch = $"[grey]{Markup.Escape(e.Architecture)}[/]";
            string status = CliTheme.StatusMarkup(e.Status);
            if (filter is null)
                table.AddRow($"[grey]{Modalities.ToCliName(e.Modality)}[/]", id, name, arch, status);
            else
                table.AddRow(id, name, arch, status);
        }

        AnsiConsole.Write(table);
        return entries.Count;
    }
}
