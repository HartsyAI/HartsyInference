using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates music from a text prompt with MusicGen, saving a WAV.</summary>
public sealed class MusicCommand : Command<MusicCommand.Settings>
{
    /// <summary>Options for <c>hartsy music</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The music description.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The music description (e.g. \"upbeat electronic dance music\").")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id (musicgen).")]
        public string Model { get; init; } = "musicgen";

        /// <summary>Path to the MusicGen model.safetensors.</summary>
        [CommandOption("--model-path")]
        [Description("Path to a musicgen-small model.safetensors.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Duration in seconds.</summary>
        [CommandOption("-d|--duration")]
        [Description("Duration in seconds.")]
        public int Duration { get; init; } = 10;

        /// <summary>RNG seed; &lt; 0 uses 0.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed.")]
        public int Seed { get; init; } = -1;

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
        if (string.IsNullOrWhiteSpace(settings.Prompt))
        {
            AnsiConsole.MarkupLine("[red]A prompt is required.[/]");
            return 1;
        }

        if (string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            AnsiConsole.MarkupLine("[red]A model is required via[/] [mediumpurple2]--model-path[/][red].[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Music) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("duration", settings.Duration.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Music);
        string label = spec.Catalog?.Id ?? settings.Model;
        string outputDir = settings.Output ?? RepoPaths.OutputRoot();

        return CommandRunner.Run(Modality.Music, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            outputDir, label, showResponseRule: false);
    }
}
