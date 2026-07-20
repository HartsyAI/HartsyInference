using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates a 3D mesh (GLB) from an input image with TripoSR or Hunyuan3D.</summary>
public sealed class ThreeDCommand : Command<ThreeDCommand.Settings>
{
    /// <summary>Options for <c>hartsy 3d</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the input PNG image.</summary>
        [CommandArgument(0, "<image>")]
        [Description("Path to the input PNG image.")]
        public string Image { get; init; } = "";

        /// <summary>Model id: triposr or hunyuan3d.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id: triposr or hunyuan3d.")]
        public string Model { get; init; } = "triposr";

        /// <summary>Path to the model directory or checkpoint.</summary>
        [CommandOption("--model-path")]
        [Description("Path to the model directory or checkpoint.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Marching-cubes grid resolution (0 = model default).</summary>
        [CommandOption("--grid")]
        [Description("Marching-cubes grid resolution (0 = model default).")]
        public int Grid { get; init; }

        /// <summary>Denoising steps (Hunyuan3D only; 0 = default).</summary>
        [CommandOption("--steps")]
        [Description("Denoising steps (Hunyuan3D only; 0 = default).")]
        public int Steps { get; init; }

        /// <summary>RNG seed; &lt; 0 randomizes.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative randomizes.")]
        public int Seed { get; init; } = -1;

        /// <summary>Directory to save the GLB to.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the GLB (defaults to the output root).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress output.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress output.")]
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
            AnsiConsole.MarkupLine("[red]A model is required via[/] [#2ea5e0]--model-path[/][red].[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Mesh) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("grid", settings.Grid.ToString(CultureInfo.InvariantCulture));
        parameters.Put("steps", settings.Steps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Mesh);
        string label = spec.Catalog?.Id ?? settings.Model;
        string outputDir = settings.Output ?? RepoPaths.OutputRoot();

        return CommandRunner.Run(Modality.Mesh, spec, settings.Image, parameters, settings.Backend, settings.Quiet,
            outputDir, label, showResponseRule: false);
    }
}
