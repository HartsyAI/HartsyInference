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

        int count = CatalogView.Render(filter, settings.VerifiedOnly);
        if (count > 0)
            AnsiConsole.MarkupLine($"[#9aa4af]{count} model(s). Use[/] [{CliTheme.Accent}]hartsy list <modality>[/] [#9aa4af]to filter, or[/] [{CliTheme.Accent}]--verified[/][#9aa4af].[/]");
        return 0;
    }
}
