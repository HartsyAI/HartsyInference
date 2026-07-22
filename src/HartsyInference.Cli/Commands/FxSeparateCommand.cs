using System.ComponentModel;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Splits a mixed audio clip into stems with Demucs, saving one WAV per stem.</summary>
public sealed class FxSeparateCommand : Command<FxSeparateCommand.Settings>
{
    /// <summary>Options for <c>hartsy fx separate</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the mixed-audio WAV to separate.</summary>
        [CommandArgument(0, "<audio>")]
        [Description("Path to the mixed-audio WAV to separate.")]
        public string Audio { get; init; } = "";

        /// <summary>Model id, optionally with a variant (e.g. demucs:htdemucs_6s).</summary>
        [CommandOption("-m|--model")]
        [Description("Model: demucs, optionally 'demucs:<variant>' (htdemucs, htdemucs_6s, htdemucs_ft).")]
        public string Model { get; init; } = "demucs";

        /// <summary>Path to a local htdemucs checkpoint (.th/.safetensors).</summary>
        [CommandOption("--model-path")]
        [Description("Path to a local htdemucs checkpoint (.th/.safetensors).")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Directory to write the stems into.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to write the stem WAVs into (defaults to the output root).")]
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
        parameters.Put("mode", "separate");

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Fx);
        string label = spec.Catalog?.Id ?? settings.Model;
        string outputDir = settings.Output ?? RepoPaths.OutputRoot();

        return CommandRunner.Run(Modality.Fx, spec, settings.Audio, parameters, settings.Backend, settings.Quiet,
            outputDir, label, showResponseRule: false);
    }
}
