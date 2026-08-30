using System.ComponentModel;
using System.Globalization;
using HartsyInference.ModelAssets.MiniMaxH3;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Locally rebases the official MiniMax-H3 Fun ControlNet branch onto a pruned FL2VA base.</summary>
public sealed class H3ControlNetConvertCommand : Command<H3ControlNetConvertCommand.Settings>
{
    /// <summary>Inputs for <c>hartsy convert h3-controlnet</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Official full-width Fun ControlNet branch.</summary>
        [CommandOption("--control <PATH>")]
        [Description("Official MiniMax-H3 Fun ControlNet-Union safetensors file.")]
        public string Control { get; init; } = "";

        /// <summary>Matching dense FL2VA base.</summary>
        [CommandOption("--full-base <PATH>")]
        [Description("Matching full FL2VA base used to reconstruct the dense AdaLN curve.")]
        public string FullBase { get; init; } = "";

        /// <summary>Target pruned FL2VA base.</summary>
        [CommandOption("--target-pruned-base <PATH>")]
        [Description("Target pruned FL2VA base containing adaln_t_table.")]
        public string TargetPrunedBase { get; init; } = "";

        /// <summary>Converted control checkpoint path.</summary>
        [CommandOption("-o|--output <PATH>")]
        [Description("Output safetensors path; inputs are never overwritten.")]
        public string Output { get; init; } = "";
    }

    /// <inheritdoc/>
    public override ValidationResult Validate(CommandContext context, Settings settings)
    {
        return string.IsNullOrWhiteSpace(settings.Control) || string.IsNullOrWhiteSpace(settings.FullBase)
            || string.IsNullOrWhiteSpace(settings.TargetPrunedBase) || string.IsNullOrWhiteSpace(settings.Output)
            ? ValidationResult.Error("--control, --full-base, --target-pruned-base, and --output are required.")
            : ValidationResult.Success();
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        MiniMaxH3ControlNetConversionSummary result = MiniMaxH3ControlNetPrunedConverter.Convert(
            settings.Control, settings.FullBase, settings.TargetPrunedBase, settings.Output);
        AnsiConsole.MarkupLine($"[green]Wrote[/] [#2ea5e0]{Markup.Escape(Path.GetFullPath(settings.Output))}[/]");
        AnsiConsole.MarkupLine($"residual [#2ea5e0]{result.RelativeResidual.ToString("E6", CultureInfo.InvariantCulture)}[/], "
            + $"rebased blocks [#2ea5e0]{result.RebasedBlocks.ToString(CultureInfo.InvariantCulture)}[/]");
        AnsiConsole.MarkupLine($"control SHA-256 [#9aa4af]{result.ControlSha256}[/]");
        AnsiConsole.MarkupLine($"full base SHA-256 [#9aa4af]{result.FullBaseSha256}[/]");
        AnsiConsole.MarkupLine($"target base SHA-256 [#9aa4af]{result.TargetBaseSha256}[/]");
        return 0;
    }
}
