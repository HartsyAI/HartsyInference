using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Runs a vision model on an image: CLIP embedding or YOLO detection, chosen by the model id.</summary>
public sealed class VisionCommand : Command<VisionCommand.Settings>
{
    /// <summary>Options for <c>hartsy vision</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the input PNG image.</summary>
        [CommandArgument(0, "<image>")]
        [Description("Path to the input PNG image.")]
        public string Image { get; init; } = "";

        /// <summary>Model id: clip (embed) or yolo8 / yolo11 (detect).</summary>
        [CommandOption("-m|--model")]
        [Description("Model id: clip (embed) or yolo8 / yolo11 (detect).")]
        public string Model { get; init; } = "clip";

        /// <summary>Path to the model .safetensors.</summary>
        [CommandOption("--model-path")]
        [Description("Path to the model .safetensors checkpoint.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Detection confidence threshold.</summary>
        [CommandOption("-c|--confidence")]
        [Description("Detection confidence threshold (YOLO only).")]
        public float Confidence { get; init; } = 0.25f;

        /// <summary>Directory to save the result text to.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the result (.txt).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress output.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress output; print only the result.")]
        public bool Quiet { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.Image))
        {
            AnsiConsole.MarkupLine("[red]An input image path is required.[/]");
            return 1;
        }

        if (string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            AnsiConsole.MarkupLine("[red]A model is required via[/] [mediumpurple2]--model-path[/][red].[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Vision) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("confidence", settings.Confidence.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Vision);
        string label = spec.Catalog?.Id ?? settings.Model;

        return CommandRunner.Run(Modality.Vision, spec, settings.Image, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
