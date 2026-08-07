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
    public sealed class Settings : PlacementCliSettings
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

        /// <summary>Start frame for image-to-video; families without keyframe conditioning ignore it.</summary>
        [CommandOption("--init-image")]
        [Description("Path to an image the clip starts on (image-to-video).")]
        public string? InitImage { get; init; }

        /// <summary>End frame; supplying both anchors the clip at each end.</summary>
        [CommandOption("--end-frame")]
        [Description("Path to an image the clip ends on. Combine with --init-image for first-and-last-frame control.")]
        public string? EndFrame { get; init; }

        /// <summary>Reference images for families that carry subject/style from references rather than pinning frames.</summary>
        [CommandOption("--ref-image")]
        [Description("Reference image to carry subject or style from; repeat for more (MiniMax-H3 takes up to 9).")]
        public string[]? ReferenceImages { get; init; }

        /// <summary>Reference video clips for families that carry subject/motion from reference footage.</summary>
        [CommandOption("--ref-video")]
        [Description("Reference video to carry subject or motion from; repeat for more (MiniMax-H3 takes up to 3).")]
        public string[]? ReferenceVideos { get; init; }

        /// <summary>Soundtracks paired by position to --ref-video.</summary>
        [CommandOption("--ref-video-audio")]
        [Description("Soundtrack (WAV) for the same-position --ref-video; repeat to pair more.")]
        public string[]? ReferenceVideoAudios { get; init; }

        /// <summary>Reference audio clips (WAV).</summary>
        [CommandOption("--ref-audio")]
        [Description("Reference audio clip (WAV) to condition on; repeat for more (MiniMax-H3 takes up to 3).")]
        public string[]? ReferenceAudios { get; init; }

        /// <summary>LoRAs to merge into the denoiser; families that don't declare LoRA support refuse them.</summary>
        [CommandOption("--lora")]
        [Description("LoRA to merge, by name or path; repeat for more. Needs an unquantized (bf16) checkpoint — a merge cannot rewrite quantized weights.")]
        public string[]? Loras { get; init; }

        /// <summary>Per-LoRA strengths, positionally matched to --lora; missing entries default to 1.0.</summary>
        [CommandOption("--lora-weight")]
        [Description("Strength for the same-position --lora (default 1.0).")]
        public double[]? LoraWeights { get; init; }

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
        parameters.PutIfSet("init-image", settings.InitImage);
        parameters.PutIfSet("end-frame", settings.EndFrame);
        // Repeatable options collapse to one newline-joined value: the parameter bag is flat strings, and a path
        // cannot contain a newline.
        if (settings.ReferenceImages is { Length: > 0 })
        {
            parameters.Put("ref-images", string.Join('\n', settings.ReferenceImages));
        }
        if (settings.ReferenceVideos is { Length: > 0 })
        {
            parameters.Put("ref-videos", string.Join('\n', settings.ReferenceVideos));
        }
        if (settings.ReferenceVideoAudios is { Length: > 0 })
        {
            parameters.Put("ref-video-audios", string.Join('\n', settings.ReferenceVideoAudios));
        }
        if (settings.ReferenceAudios is { Length: > 0 })
        {
            parameters.Put("ref-audios", string.Join('\n', settings.ReferenceAudios));
        }
        if (settings.Loras is { Length: > 0 })
        {
            parameters.Put("loras", string.Join('\n', settings.Loras));
            parameters.Put("lora-weights", string.Join('\n',
                settings.Loras.Select((_, i) => (settings.LoraWeights is not null && i < settings.LoraWeights.Length
                    ? settings.LoraWeights[i] : 1.0).ToString(CultureInfo.InvariantCulture))));
        }
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        if (settings.Restore.IsSet)
        {
            parameters.Put("restore", string.IsNullOrWhiteSpace(settings.Restore.Value) ? "seedvr2-3b" : settings.Restore.Value);
            parameters.PutIfSet("restore-width", settings.RestoreWidth);
            parameters.PutIfSet("restore-height", settings.RestoreHeight);
        }

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Video);
        string label = CommandRunner.ResolveLabel(spec, settings.Model, settings.ModelPath);

        (int? gpu, EngineOptions? engineOptions) = PlacementCli.Build(settings, settings.Backend,
            HartsyInference.Engine.Modality.Video, PlacementCli.TryModelBytes(settings.Model));
        return CommandRunner.Run(Modality.Video, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false, gpu, engineOptions);
    }
}
