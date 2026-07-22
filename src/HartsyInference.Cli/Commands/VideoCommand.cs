using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates a video (frame sequence) from a prompt with any registered video family. CUDA-only,
/// validation-pending per family — see <c>docs/Checklists/MODEL_STATUS_VIDEO.md</c>.</summary>
public sealed class VideoCommand : Command<VideoCommand.Settings>
{
    /// <summary>Options for <c>hartsy video</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The video description.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The video description.")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id (catalog) or path. Optional when <c>--model-path</c> is given.</summary>
        [CommandOption("-m|--model")]
        [Description("Catalog model id (ltx-video, wan, lance-video, ...) or a local path. Optional when --model-path is given.")]
        public string Model { get; init; } = "";

        /// <summary>Path to a video checkpoint in any registered family's layout.</summary>
        [CommandOption("--model-path")]
        [Description("Path to a checkpoint (file or folder) of any registered video family; pair it with -m <family> when the layout is ambiguous.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend (must be cuda for video).</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend (video requires cuda).")]
        public string Backend { get; init; } = "cuda";

        /// <summary>Negative prompt.</summary>
        [CommandOption("-n|--negative")]
        [Description("Negative prompt.")]
        public string Negative { get; init; } = "";

        /// <summary>Frame width in pixels; unset uses the family's native width.</summary>
        [CommandOption("--width")]
        [Description("Frame width in pixels (default: the model family's native width).")]
        public int? Width { get; init; }

        /// <summary>Frame height in pixels; unset uses the family's native height.</summary>
        [CommandOption("--height")]
        [Description("Frame height in pixels (default: the model family's native height).")]
        public int? Height { get; init; }

        /// <summary>Number of frames; unset uses the family's officially recommended count.</summary>
        [CommandOption("--frames")]
        [Description("Number of frames (default: the model family's recommended count).")]
        public int? Frames { get; init; }

        /// <summary>Denoising steps; unset uses the family's officially recommended count.</summary>
        [CommandOption("--steps")]
        [Description("Denoising steps (default: the model family's recommended count).")]
        public int? Steps { get; init; }

        /// <summary>Guidance scale; unset uses the family's officially recommended scale.</summary>
        [CommandOption("--cfg")]
        [Description("Guidance scale (default: the model family's recommended scale).")]
        public float? Cfg { get; init; }

        /// <summary>Frames per second; unset uses the family's native frame rate.</summary>
        [CommandOption("--fps")]
        [Description("Frame rate (default: the model family's native rate).")]
        public int? Fps { get; init; }

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
        ParamState parameters = new ParamState(Modality.Video) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("negative", settings.Negative);
        PutIfSet(parameters, "width", settings.Width);
        PutIfSet(parameters, "height", settings.Height);
        PutIfSet(parameters, "frames", settings.Frames);
        PutIfSet(parameters, "steps", settings.Steps);
        PutIfSet(parameters, "cfg", settings.Cfg);
        PutIfSet(parameters, "fps", settings.Fps);
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Video);
        string label = spec.Catalog?.Id ?? (settings.ModelPath is { Length: > 0 } mp ? Path.GetFileName(mp) : settings.Model);

        return CommandRunner.Run(Modality.Video, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }

    /// <summary>Forwards a tunable only when the user actually passed the flag; an omitted flag leaves the key empty so it reaches the engine as null.</summary>
    private static void PutIfSet(ParamState parameters, string key, IFormattable? value)
    {
        if (value is not null)
        {
            parameters.Put(key, value.ToString(null, CultureInfo.InvariantCulture));
        }
    }
}
