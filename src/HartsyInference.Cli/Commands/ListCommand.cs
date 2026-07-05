using System.ComponentModel;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Prints the model catalog as a table, optionally filtered to one modality.</summary>
public sealed class ListCommand : Command<ListCommand.Settings>
{
    /// <summary>Options for <c>hartsy list</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Optional modality filter (image, text, speech, music, transcribe, vision, video, 3d, interactive).</summary>
        [CommandArgument(0, "[modality]")]
        [Description("Filter to one modality: image, text, speech, music, transcribe, vision, video, 3d, interactive.")]
        public string? Modality { get; init; }

        /// <summary>Show only verified models.</summary>
        [CommandOption("--verified")]
        [Description("Show only real-weight-verified models.")]
        public bool VerifiedOnly { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        Modality? filter = null;
        if (!string.IsNullOrWhiteSpace(settings.Modality))
        {
            if (!Modalities.TryParse(settings.Modality, out Modality parsed))
            {
                AnsiConsole.MarkupLine($"[red]Unknown modality '{Markup.Escape(settings.Modality)}'.[/] Valid: {string.Join(", ", Modalities.All.Select(Modalities.ToCliName))}.");
                return 1;
            }
            filter = parsed;
        }

        IEnumerable<CatalogEntry> rows = ModelCatalog.All;
        if (filter is not null)
            rows = rows.Where(e => e.Modality == filter.Value);
        if (settings.VerifiedOnly)
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
        AnsiConsole.MarkupLine($"[grey]{entries.Count} model(s). Use[/] [{CliTheme.Accent}]hartsy list <modality>[/] [grey]to filter, or[/] [{CliTheme.Accent}]--verified[/][grey].[/]");
        return 0;
    }
}
