using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates an image from a prompt with a diffusion checkpoint (SDXL today), saving a BMP.</summary>
public sealed class ImageCommand : Command<ImageCommand.Settings>
{
    /// <summary>Options for <c>hartsy image</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The positive prompt.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The image prompt.")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id (catalog) or path. Optional when <c>--model-path</c> is given.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id (e.g. sdxl) or a local path. Optional when --model-path is given.")]
        public string Model { get; init; } = "";

        /// <summary>Path to an SDXL .safetensors checkpoint.</summary>
        [CommandOption("--model-path")]
        [Description("Path to an SDXL .safetensors checkpoint.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Negative prompt.</summary>
        [CommandOption("-n|--negative")]
        [Description("Negative prompt.")]
        public string Negative { get; init; } = "";

        /// <summary>Image width in pixels.</summary>
        [CommandOption("--width")]
        [Description("Image width in pixels.")]
        public int Width { get; init; } = 1024;

        /// <summary>Image height in pixels.</summary>
        [CommandOption("--height")]
        [Description("Image height in pixels.")]
        public int Height { get; init; } = 1024;

        /// <summary>Number of denoising steps.</summary>
        [CommandOption("--steps")]
        [Description("Number of denoising steps.")]
        public int Steps { get; init; } = 20;

        /// <summary>Classifier-free guidance scale.</summary>
        [CommandOption("--cfg")]
        [Description("Classifier-free guidance scale.")]
        public float Cfg { get; init; } = 7.5f;

        /// <summary>RNG seed; &lt; 0 randomizes.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative randomizes.")]
        public int Seed { get; init; } = -1;

        /// <summary>Directory to save the image to.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the image (defaults to the output root).")]
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

        if (string.IsNullOrWhiteSpace(settings.Model) && string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            AnsiConsole.MarkupLine("[red]Specify a model with[/] [mediumpurple2]--model[/] [red]or[/] [mediumpurple2]--model-path[/][red].[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Image) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("negative", settings.Negative);
        parameters.Put("width", settings.Width.ToString(CultureInfo.InvariantCulture));
        parameters.Put("height", settings.Height.ToString(CultureInfo.InvariantCulture));
        parameters.Put("steps", settings.Steps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("cfg", settings.Cfg.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Image);
        string label = spec.Catalog?.Id ?? (settings.ModelPath is { Length: > 0 } mp ? Path.GetFileName(mp) : settings.Model);
        // Images always persist; default to the output root when -o is omitted.
        string outputDir = settings.Output ?? RepoPaths.OutputRoot();

        return CommandRunner.Run(Modality.Image, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            outputDir, label, showResponseRule: false, present: (artifact, quiet) =>
            {
                if (!quiet)
                    AnsiConsole.MarkupLine($"[grey]{Markup.Escape(artifact.Text ?? "image")}[/]");
            });
    }
}
