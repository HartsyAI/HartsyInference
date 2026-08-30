using System.ComponentModel;
using System.Globalization;
using HartsyInference.ModelAssets.MiniMaxH3;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Locally rebases an official MiniMax-H3 PDD adapter from a matching dense base onto a pruned base.</summary>
public sealed class H3PddConvertCommand : Command<H3PddConvertCommand.Settings>
{
    /// <summary>Inputs for <c>hartsy convert h3-pdd</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Official PDD adapter file.</summary>
        [CommandOption("--adapter <PATH>")]
        [Description("Official MiniMax-H3 PDD adapter safetensors file.")]
        public string Adapter { get; init; } = "";

        /// <summary>Matching full H3 base.</summary>
        [CommandOption("--full-base <PATH>")]
        [Description("Matching full (non-pruned) H3 base used to reconstruct the dense AdaLN curve.")]
        public string FullBase { get; init; } = "";

        /// <summary>Target pruned H3 base.</summary>
        [CommandOption("--target-pruned-base <PATH>")]
        [Description("Target pruned H3 base containing adaln_t_table.")]
        public string TargetPrunedBase { get; init; } = "";

        /// <summary>Output adapter path.</summary>
        [CommandOption("-o|--output <PATH>")]
        [Description("Output safetensors path; inputs are never overwritten.")]
        public string Output { get; init; } = "";

        /// <summary>Task family binding.</summary>
        [CommandOption("--task <TASK>")]
        [Description("Adapter/base task: fl2va or ref2va.")]
        public string Task { get; init; } = "";
    }

    /// <inheritdoc/>
    public override ValidationResult Validate(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.Adapter) || string.IsNullOrWhiteSpace(settings.FullBase)
            || string.IsNullOrWhiteSpace(settings.TargetPrunedBase) || string.IsNullOrWhiteSpace(settings.Output))
        {
            return ValidationResult.Error("--adapter, --full-base, --target-pruned-base, and --output are required.");
        }
        return TryTask(settings.Task, out _)
            ? ValidationResult.Success()
            : ValidationResult.Error("--task must be fl2va or ref2va.");
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        TryTask(settings.Task, out MiniMaxH3PddTask task);
        MiniMaxH3PddConversionSummary result = MiniMaxH3PddPrunedConverter.Convert(
            settings.Adapter, settings.FullBase, settings.TargetPrunedBase, settings.Output, task);
        AnsiConsole.MarkupLine($"[green]Wrote[/] [#2ea5e0]{Markup.Escape(Path.GetFullPath(settings.Output))}[/]");
        AnsiConsole.MarkupLine($"residual [#2ea5e0]{result.RelativeResidual.ToString("E6", CultureInfo.InvariantCulture)}[/], "
            + $"rebased modules [#2ea5e0]{result.RebasedModules.ToString(CultureInfo.InvariantCulture)}[/]");
        AnsiConsole.MarkupLine($"adapter SHA-256 [#9aa4af]{result.AdapterSha256}[/]");
        AnsiConsole.MarkupLine($"full base SHA-256 [#9aa4af]{result.FullBaseSha256}[/]");
        AnsiConsole.MarkupLine($"target base SHA-256 [#9aa4af]{result.TargetBaseSha256}[/]");
        return 0;
    }

    private static bool TryTask(string value, out MiniMaxH3PddTask task)
    {
        string normalized = value.Trim().Replace("-", "", StringComparison.Ordinal)
            .Replace("_", "", StringComparison.Ordinal).ToLowerInvariant();
        task = normalized switch
        {
            "fl2va" => MiniMaxH3PddTask.Fl2Va,
            "ref2va" => MiniMaxH3PddTask.Ref2Va,
            _ => MiniMaxH3PddTask.Unknown,
        };
        return task != MiniMaxH3PddTask.Unknown;
    }
}
