using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates a 3D mesh (GLB) or Gaussian-splat cloud (PLY) from an input image with TripoSR, Hunyuan3D, or TRELLIS.</summary>
public sealed class ThreeDCommand : Command<ThreeDCommand.Settings>
{
    /// <summary>Options for <c>hartsy 3d</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the input PNG image.</summary>
        [CommandArgument(0, "<image>")]
        [Description("Path to the input PNG image.")]
        public string Image { get; init; } = "";

        /// <summary>Model id: triposr, hunyuan3d, trellis, or hairdeck.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id: triposr, hunyuan3d, trellis, or hairdeck.")]
        public string Model { get; init; } = "triposr";

        /// <summary>Samples to generate (HairDeck only; this build accepts 1).</summary>
        [CommandOption("-n|--samples")]
        [Description("Samples to generate (hairdeck only; this build accepts 1).")]
        public int Samples { get; init; } = 1;

        /// <summary>Rigged (VRM) output (HairDeck only; not implemented yet).</summary>
        [CommandOption("--rigged")]
        [Description("Rigged output (hairdeck only; not implemented yet).")]
        public bool Rigged { get; init; }

        /// <summary>Skip the colour model (HairDeck only).</summary>
        [CommandOption("--no-colors")]
        [Description("Skip the colour model (hairdeck only).")]
        public bool NoColors { get; init; }

        /// <summary>Skip the spring-bone chains (HairDeck only).</summary>
        [CommandOption("--no-chains")]
        [Description("Skip the spring-bone chains (hairdeck only).")]
        public bool NoChains { get; init; }

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
        if (!CommandRunner.RequireNonEmpty(settings.Image, "An input image path is required.", out int exitCode))
            return exitCode;

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Mesh);
        if (spec.LocalPath is null)
        {
            AnsiConsole.MarkupLine("[red]A model is required via[/] [#2ea5e0]--model-path[/][red], or download its catalog assets.[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Mesh) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("grid", settings.Grid.ToString(CultureInfo.InvariantCulture));
        parameters.Put("steps", settings.Steps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        parameters.Put("samples", settings.Samples.ToString(CultureInfo.InvariantCulture));
        parameters.Put("rigged", settings.Rigged ? "true" : "false");
        parameters.Put("no-colors", settings.NoColors ? "true" : "false");
        parameters.Put("no-chains", settings.NoChains ? "true" : "false");
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        return CommandRunner.Run(Modality.Mesh, spec, settings.Image, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
