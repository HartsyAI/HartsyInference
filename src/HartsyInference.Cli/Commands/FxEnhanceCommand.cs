using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Denoises and enhances a recording with Resemble-Enhance, saving a WAV.</summary>
public sealed class FxEnhanceCommand : Command<FxEnhanceCommand.Settings>
{
    /// <summary>Options for <c>hartsy fx enhance</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the audio WAV to enhance.</summary>
        [CommandArgument(0, "<audio>")]
        [Description("Path to the audio WAV to enhance.")]
        public string Audio { get; init; } = "";

        /// <summary>Model id.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id (resemble-enhance).")]
        public string Model { get; init; } = "resemble-enhance";

        /// <summary>Denoise/enhance blend; model default when unset.</summary>
        [CommandOption("--lambda")]
        [Description("Denoise/enhance blend; model default when unset.")]
        public float? Lambda { get; init; }

        /// <summary>CFM temperature; model default when unset.</summary>
        [CommandOption("--tau")]
        [Description("CFM temperature; model default when unset.")]
        public float? Tau { get; init; }

        /// <summary>RNG seed; negative uses 0.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative uses 0.")]
        public int Seed { get; init; } = -1;

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Directory to save the WAV to.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the WAV (defaults to the output root).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress output.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress output.")]
        public bool Quiet { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.Audio))
        {
            AnsiConsole.MarkupLine("[red]An audio file path is required.[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Fx) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("mode", "enhance");
        if (settings.Lambda is { } lambda)
            parameters.Put("lambda", lambda.ToString(CultureInfo.InvariantCulture));
        if (settings.Tau is { } tau)
            parameters.Put("tau", tau.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, null, Modality.Fx);
        string label = spec.Catalog?.Id ?? settings.Model;
        string outputDir = settings.Output ?? RepoPaths.OutputRoot();

        return CommandRunner.Run(Modality.Fx, spec, settings.Audio, parameters, settings.Backend, settings.Quiet,
            outputDir, label, showResponseRule: false);
    }
}
