using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates a video (frame sequence) from a prompt with any registered video family. CUDA-only.</summary>
/// <remarks>Validation-pending per family — see <c>docs/Checklists/MODEL_STATUS_VIDEO.md</c>.</remarks>
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

        /// <summary>Optional SeedVR2 restore pass over the generated frames.</summary>
        [CommandOption("--restore [MODEL]")]
        [Description("Restore/upscale the generated frames with SeedVR2 in the same run (default model seedvr2-3b). Generate small, restore up.")]
        public FlagValue<string> Restore { get; init; } = new();

        /// <summary>Restore target-area width (with --restore); default 1280.</summary>
        [CommandOption("--restore-width")]
        [Description("Restore target-area width (default 1280).")]
        public int? RestoreWidth { get; init; }

        /// <summary>Restore target-area height (with --restore); default 720.</summary>
        [CommandOption("--restore-height")]
        [Description("Restore target-area height (default 720).")]
        public int? RestoreHeight { get; init; }

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
        if (!CommandRunner.RequireNonEmpty(settings.Prompt, "A prompt is required.", out int exitCode))
            return exitCode;

        if (!CommandRunner.RequireModelOrPath(settings.Model, settings.ModelPath, "--model", "--model-path", out exitCode))
            return exitCode;

        // Only flags the user actually passed are forwarded; anything omitted stays unset so the engine applies the
        // resolved family's official defaults instead of a generic guess.
        ParamState parameters = new ParamState(Modality.Video) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("negative", settings.Negative);
        parameters.PutIfSet("width", settings.Width);
        parameters.PutIfSet("height", settings.Height);
        parameters.PutIfSet("frames", settings.Frames);
        parameters.PutIfSet("steps", settings.Steps);
        parameters.PutIfSet("cfg", settings.Cfg);
        parameters.PutIfSet("fps", settings.Fps);
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        if (settings.Restore.IsSet)
        {
            parameters.Put("restore", string.IsNullOrWhiteSpace(settings.Restore.Value) ? "seedvr2-3b" : settings.Restore.Value);
            parameters.PutIfSet("restore-width", settings.RestoreWidth);
            parameters.PutIfSet("restore-height", settings.RestoreHeight);
        }

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Video);
        string label = CommandRunner.ResolveLabel(spec, settings.Model, settings.ModelPath);

        return CommandRunner.Run(Modality.Video, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
