using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates 3D output from an input: a mesh (GLB) or Gaussian-splat cloud (PLY) from an image with TripoSR, Hunyuan3D or TRELLIS, or one GLB mesh per decomposed object from a scene mesh with CoLa3D.</summary>
public sealed class ThreeDCommand : Command<ThreeDCommand.Settings>
{
    /// <summary>Options for <c>hartsy 3d</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the input: a PNG image, or a .obj/.glb scene mesh for cola3d.</summary>
        [CommandArgument(0, "<input>")]
        [Description("Input file: a PNG image (triposr, hunyuan3d, trellis) or a .obj/.glb scene mesh (cola3d).")]
        public string Input { get; init; } = "";

        /// <summary>Model id: triposr, hunyuan3d, trellis, or cola3d.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id: triposr, hunyuan3d, trellis, or cola3d.")]
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
        [Description("Grid resolution (0 = model default; cola3d: dual-grid voxels per axis, a multiple of 16).")]
        public int Grid { get; init; }

        /// <summary>Mask PNGs for cola3d, one per object (repeatable); all the same size, red channel is the coverage.</summary>
        [CommandOption("--prompt-mask")]
        [Description("cola3d: a mask PNG per object, all the same size (repeatable). Red channel is the coverage. Excludes --prompt-point.")]
        public string[]? PromptMasks { get; init; }

        /// <summary>Scene points for cola3d, one per object as x,y,z in scene units (repeatable).</summary>
        [CommandOption("--prompt-point")]
        [Description("cola3d: a point per object as x,y,z in the scene mesh's units (repeatable). Excludes --prompt-mask.")]
        public string[]? PromptPoints { get; init; }

        /// <summary>Denoising steps (Hunyuan3D and TRELLIS only; 0 = default).</summary>
        [CommandOption("--steps")]
        [Description("Denoising steps (Hunyuan3D and TRELLIS only; 0 = default).")]
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
        if (!CommandRunner.RequireNonEmpty(settings.Input, "An input path is required (an image, or a scene mesh for cola3d).", out int exitCode))
            return exitCode;

        if (string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            AnsiConsole.MarkupLine("[red]A model is required via[/] [#2ea5e0]--model-path[/][red].[/]");
            return 1;
        }

        if (settings.PromptMasks is { Length: > 0 } && settings.PromptPoints is { Length: > 0 })
        {
            AnsiConsole.MarkupLine("[red]Give[/] [#2ea5e0]--prompt-mask[/] [red]or[/] [#2ea5e0]--prompt-point[/][red], not both.[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Mesh) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        if (settings.PromptMasks is { Length: > 0 })
            parameters.Put("prompt-mask", string.Join('\n', settings.PromptMasks));
        if (settings.PromptPoints is { Length: > 0 })
            parameters.Put("prompt-point", string.Join('\n', settings.PromptPoints));
        parameters.Put("grid", settings.Grid.ToString(CultureInfo.InvariantCulture));
        parameters.Put("steps", settings.Steps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Mesh);
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        return CommandRunner.Run(Modality.Mesh, spec, settings.Input, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
