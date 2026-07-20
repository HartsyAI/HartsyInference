using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates an image from a prompt with any registered diffusion family, saving a PNG.</summary>
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
        [Description("Catalog model id (sdxl, flux1, zimage, chroma, qwen-image, krea2, boogu, lens, ...) or a local path. Optional when --model-path is given.")]
        public string Model { get; init; } = "";

        /// <summary>Path to a diffusion checkpoint in any registered family's layout.</summary>
        [CommandOption("--model-path")]
        [Description("Path to a .safetensors/.gguf diffusion checkpoint of any registered family; pair it with -m <family> when the layout is ambiguous.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Negative prompt.</summary>
        [CommandOption("-n|--negative")]
        [Description("Negative prompt.")]
        public string Negative { get; init; } = "";

        /// <summary>Image width in pixels; unset uses the family's native width.</summary>
        [CommandOption("--width")]
        [Description("Image width in pixels (default: the model family's native width).")]
        public int? Width { get; init; }

        /// <summary>Image height in pixels; unset uses the family's native height.</summary>
        [CommandOption("--height")]
        [Description("Image height in pixels (default: the model family's native height).")]
        public int? Height { get; init; }

        /// <summary>Number of denoising steps; unset uses the family's officially recommended count.</summary>
        [CommandOption("--steps")]
        [Description("Number of denoising steps (default: the model family's recommended count).")]
        public int? Steps { get; init; }

        /// <summary>Classifier-free guidance scale; unset uses the family's officially recommended scale.</summary>
        [CommandOption("--cfg")]
        [Description("Guidance scale (default: the model family's recommended scale; 1.0 for distilled/turbo models).")]
        public float? Cfg { get; init; }

        /// <summary>Sampler name; unset uses the family's canonical sampler.</summary>
        [CommandOption("--sampler")]
        [Description("Sampler name, e.g. euler, ddim, dpmpp_2m, lcm (default: the model family's sampler).")]
        public string? Sampler { get; init; }

        /// <summary>Scheduler / sigma-schedule name; unset uses the family's canonical schedule.</summary>
        [CommandOption("--scheduler")]
        [Description("Scheduler / sigma-schedule name (default: the model family's schedule).")]
        public string? Scheduler { get; init; }

        /// <summary>Flow-match sigma shift; unset uses the family's trained shift.</summary>
        [CommandOption("--sigma-shift")]
        [Description("Flow-match sigma shift (default: the model family's trained shift).")]
        public double? SigmaShift { get; init; }

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
            AnsiConsole.MarkupLine("[red]Specify a model with[/] [#2ea5e0]--model[/] [red]or[/] [#2ea5e0]--model-path[/][red].[/]");
            return 1;
        }

        // Only flags the user actually passed are forwarded; anything omitted stays unset so the engine applies the
        // resolved family's official defaults instead of a generic guess.
        ParamState parameters = new ParamState(Modality.Image) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("negative", settings.Negative);
        PutIfSet(parameters, "width", settings.Width);
        PutIfSet(parameters, "height", settings.Height);
        PutIfSet(parameters, "steps", settings.Steps);
        PutIfSet(parameters, "cfg", settings.Cfg);
        PutIfSet(parameters, "sampler", settings.Sampler);
        PutIfSet(parameters, "scheduler", settings.Scheduler);
        PutIfSet(parameters, "sigma-shift", settings.SigmaShift);
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Image);
        string label = spec.Catalog?.Id ?? (settings.ModelPath is { Length: > 0 } mp ? Path.GetFileName(mp) : settings.Model);
        // Images always persist; default to the output root when -o is omitted.
        string outputDir = settings.Output ?? RepoPaths.OutputRoot();

        return CommandRunner.Run(Modality.Image, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            outputDir, label, showResponseRule: false);
    }

    /// <summary>Forwards a tunable only when the user actually passed the flag; an omitted flag leaves the key empty so it reaches the engine as null.</summary>
    private static void PutIfSet(ParamState parameters, string key, IFormattable? value)
    {
        if (value is not null)
        {
            parameters.Put(key, value.ToString(null, CultureInfo.InvariantCulture));
        }
    }

    /// <summary>Forwards a string tunable only when the user actually passed a non-empty value.</summary>
    private static void PutIfSet(ParamState parameters, string key, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            parameters.Put(key, value);
        }
    }
}
