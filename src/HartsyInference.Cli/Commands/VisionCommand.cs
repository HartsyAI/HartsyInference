using System.ComponentModel;
using System.Globalization;
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
        /// <summary>Path to the input image, or a folder of images for frame and multi-view modes.</summary>
        [CommandArgument(0, "<image>")]
        [Description("Path to the input image, or a folder of .png/.bmp images (sorted by name): the frames for track/interpolate, or the views for multi-view depth/metric-depth/geometry.")]
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

        /// <summary>Vision operation; inferred from the model id when omitted.</summary>
        [CommandOption("--mode")]
        [Description("Operation: embed, detect, segment, depth, edge, lineart, normal, segmap, removebg, upscale, metric-depth, geometry, pose, matting, face, intrinsic, track, or interpolate. Inferred from the model id when omitted. Structured results write .npy arrays, PNG previews and a result.json.")]
        public string? Mode { get; init; }

        /// <summary>Frame modes: the input folder's frame rate.</summary>
        [CommandOption("--fps")]
        [Description("track/interpolate: the input frames' rate (frame i is at i / fps seconds).")]
        public double? Fps { get; init; }

        /// <summary>Interpolate: output frames per input interval.</summary>
        [CommandOption("--factor")]
        [Description("interpolate: output frames per input interval (2 doubles the rate). Use this or --target-fps.")]
        public int? Factor { get; init; }

        /// <summary>Interpolate: the output frame rate.</summary>
        [CommandOption("--target-fps")]
        [Description("interpolate: the output frame rate, above the input rate. Use this or --factor.")]
        public double? TargetFps { get; init; }

        /// <summary>Upscale only: fit the result to this width (see <see cref="Height"/>).</summary>
        [CommandOption("--width")]
        [Description("Upscale only: output width to fit to. Omit both sizes to enlarge by the model's own factor.")]
        public int? Width { get; init; }

        /// <summary>Upscale only: fit the result to this height.</summary>
        [CommandOption("--height")]
        [Description("Upscale only: output height to fit to (derived from --width and the source aspect when omitted).")]
        public int? Height { get; init; }

        /// <summary>Text query for open-vocabulary detect/segment (Grounding DINO, CLIPSeg).</summary>
        [CommandOption("-p|--prompt")]
        [Description("Text query for open-vocabulary detect/segment (e.g. \"a fox\").")]
        public string? Prompt { get; init; }

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
        if (!CommandRunner.RequireNonEmpty(settings.Image, "An input image path is required.", out int exitCode))
            return exitCode;

        if (!CommandRunner.RequireModelOrPath(settings.Model, settings.ModelPath, "-m", "--model-path", out exitCode))
            return exitCode;

        ParamState parameters = new ParamState(Modality.Vision) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("confidence", settings.Confidence.ToString(CultureInfo.InvariantCulture));
        parameters.Put("mode", settings.Mode ?? InferMode(settings.Model));
        if (!string.IsNullOrWhiteSpace(settings.Prompt))
        {
            parameters.Put("query", settings.Prompt);
        }
        if (settings.Width is int width)
        {
            parameters.Put("width", width.ToString(CultureInfo.InvariantCulture));
        }
        if (settings.Height is int height)
        {
            parameters.Put("height", height.ToString(CultureInfo.InvariantCulture));
        }
        if (settings.Fps is double fps)
        {
            parameters.Put("fps", fps.ToString(CultureInfo.InvariantCulture));
        }
        if (settings.Factor is int factor)
        {
            parameters.Put("factor", factor.ToString(CultureInfo.InvariantCulture));
        }
        if (settings.TargetFps is double targetFps)
        {
            parameters.Put("target-fps", targetFps.ToString(CultureInfo.InvariantCulture));
        }

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Vision);
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        return CommandRunner.Run(Modality.Vision, spec, settings.Image, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }

    /// <summary>Picks the operation a model id implies: detect, segment, embed, or one of the single-image annotator modes.</summary>
    private static string InferMode(string model)
    {
        string id = model.ToLowerInvariant();
        if (id.Contains("clipseg") || id.Contains("sam"))
        {
            return "segment";
        }
        if (id.Contains("yolo") || id.Contains("rtdetr") || id.Contains("rt-detr")
            || (id.Contains("dino") && !id.StartsWith("dinov", StringComparison.Ordinal)))
        {
            return "detect";
        }
        if (id.Contains("depth"))
        {
            return "depth";
        }
        if (id == "hed")
        {
            return "edge";
        }
        if (id.Contains("lineart"))
        {
            return "lineart";
        }
        if (id.Contains("normalbae") || id.Contains("normal-bae"))
        {
            return "normal";
        }
        if (id.Contains("upernet") || id.Contains("segmap"))
        {
            return "segmap";
        }
        if (id.Contains("rmbg"))
        {
            return "removebg";
        }
        if (id.Contains("esrgan") || id.Contains("upscale"))
        {
            return "upscale";
        }
        return "embed";
    }
}
