using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Rolls out an Oasis world model from a first-frame image with a canned "move forward" action plan.</summary>
public sealed class InteractiveCommand : Command<InteractiveCommand.Settings>
{
    /// <summary>Options for <c>hartsy world</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the first-frame image (Oasis expects 640x360).</summary>
        [CommandArgument(0, "<image>")]
        [Description("Path to the first-frame PNG (Oasis expects 640x360).")]
        public string Image { get; init; } = "";

        /// <summary>Model id.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id (oasis).")]
        public string Model { get; init; } = "oasis";

        /// <summary>Path to the Oasis DiT .safetensors.</summary>
        [CommandOption("--model-path")]
        [Description("Path to the Oasis DiT .safetensors.")]
        public string? ModelPath { get; init; }

        /// <summary>Path to the Oasis ViT-VAE .safetensors.</summary>
        [CommandOption("--vae-path")]
        [Description("Path to the Oasis ViT-VAE .safetensors.")]
        public string? VaePath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan (cuda recommended).")]
        public string Backend { get; init; } = "cuda";

        /// <summary>Number of frames to roll out.</summary>
        [CommandOption("--frames")]
        [Description("Number of frames to roll out.")]
        public int Frames { get; init; } = 16;

        /// <summary>DDIM steps per frame.</summary>
        [CommandOption("--steps")]
        [Description("DDIM steps per frame.")]
        public int Steps { get; init; } = 10;

        /// <summary>RNG seed; &lt; 0 randomizes.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative randomizes.")]
        public int Seed { get; init; } = -1;

        /// <summary>Directory to write the frame sequence into.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to write the frame sequence into (defaults to the output root).")]
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
            AnsiConsole.MarkupLine("[red]A first-frame image path is required.[/]");
            return 1;
        }

        if (string.IsNullOrWhiteSpace(settings.ModelPath) || string.IsNullOrWhiteSpace(settings.VaePath))
        {
            AnsiConsole.MarkupLine("[red]Oasis needs[/] [#2ea5e0]--model-path[/] [red](DiT) and[/] [#2ea5e0]--vae-path[/] [red](ViT-VAE).[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Interactive) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("frames", settings.Frames.ToString(CultureInfo.InvariantCulture));
        parameters.Put("steps", settings.Steps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        Dictionary<string, string> aux = new Dictionary<string, string> { ["vae-path"] = settings.VaePath! };
        ModelSpec baseSpec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Interactive);
        ModelSpec spec = baseSpec with { Aux = aux };
        string label = spec.Catalog?.Id ?? settings.Model;

        return CommandRunner.Run(Modality.Interactive, spec, settings.Image, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
