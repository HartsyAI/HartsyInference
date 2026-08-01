using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Restores a degraded video or image with SeedVR2, saving PNG frames plus an MP4 for video
/// input. The target is an AREA (--width × --height, aspect preserved by the model's bicubic
/// area-resize) — SeedVR2 has no scale factor; --scale is sugar that computes the area from the input.</summary>
public sealed class RestoreCommand : Command<RestoreCommand.Settings>
{
    /// <summary>Options for <c>hartsy restore</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the video or image to restore.</summary>
        [CommandArgument(0, "<input>")]
        [Description("Path to the video (mp4/avi/…) or image (png/jpg/bmp) to restore.")]
        public string Input { get; init; } = "";

        /// <summary>Model id or path.</summary>
        [CommandOption("-m|--model")]
        [Description("Catalog model id (seedvr2-3b, seedvr2-7b) or a local path.")]
        public string Model { get; init; } = "seedvr2-3b";

        /// <summary>Explicit checkpoint path override.</summary>
        [CommandOption("--model-path")]
        [Description("Local DiT safetensors path (VAE + embeddings resolve as siblings).")]
        public string? ModelPath { get; init; }

        /// <summary>Target-area width; default 1280.</summary>
        [CommandOption("--width")]
        [Description("Target-area width (default 1280). With --height defines an AREA — aspect is preserved.")]
        public int? Width { get; init; }

        /// <summary>Target-area height; default 720.</summary>
        [CommandOption("--height")]
        [Description("Target-area height (default 720).")]
        public int? Height { get; init; }

        /// <summary>Scale sugar over the area knob.</summary>
        [CommandOption("--scale")]
        [Description("Compute the target area as input-area × scale². Mutually exclusive with --width/--height.")]
        public float? Scale { get; init; }

        /// <summary>Frames per chunk.</summary>
        [CommandOption("--clip-frames")]
        [Description("Frames per processing chunk (default 25; rounded to the (n-1)%4==0 contract). Lower on tight VRAM.")]
        public int? ClipFrames { get; init; }

        /// <summary>Chunk overlap.</summary>
        [CommandOption("--overlap")]
        [Description("Frame overlap between chunks, cross-faded (default 4).")]
        public int? Overlap { get; init; }

        /// <summary>Wavelet transplant weight.</summary>
        [CommandOption("--strength")]
        [Description("Restoration strength 0..1 (default 1.0 = pure model output; lower keeps input low frequencies — guards oversharpening).")]
        public float? Strength { get; init; }

        /// <summary>Output fps override.</summary>
        [CommandOption("--fps")]
        [Description("MP4 output fps; defaults to the source rate.")]
        public double? Fps { get; init; }

        /// <summary>RNG seed; negative uses the reference default.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative uses the model default (666).")]
        public int Seed { get; init; } = -1;

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Directory to save output to.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save frames/MP4 (defaults to the output root).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress output.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress output.")]
        public bool Quiet { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (!CommandRunner.RequireNonEmpty(settings.Input, "An input video or image path is required.", out int exitCode))
            return exitCode;
        if (settings.Scale is not null && (settings.Width is not null || settings.Height is not null))
        {
            Spectre.Console.AnsiConsole.MarkupLine("[red]--scale is mutually exclusive with --width/--height.[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Restore)
        {
            Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output,
        };
        (int? width, int? height) = ResolveArea(settings);
        if (width is { } w)
            parameters.Put("width", w.ToString(CultureInfo.InvariantCulture));
        if (height is { } h)
            parameters.Put("height", h.ToString(CultureInfo.InvariantCulture));
        if (settings.ClipFrames is { } cf)
            parameters.Put("clip-frames", cf.ToString(CultureInfo.InvariantCulture));
        if (settings.Overlap is { } ov)
            parameters.Put("overlap", ov.ToString(CultureInfo.InvariantCulture));
        if (settings.Strength is { } st)
            parameters.Put("strength", st.ToString(CultureInfo.InvariantCulture));
        if (settings.Fps is { } fps)
            parameters.Put("fps", fps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Restore);
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        return CommandRunner.Run(Modality.Restore, spec, settings.Input, parameters, settings.Backend,
            settings.Quiet, settings.Output, label, showResponseRule: false);
    }

    private static (int? Width, int? Height) ResolveArea(Settings settings)
    {
        if (settings.Scale is not { } scale)
            return (settings.Width, settings.Height);
        string path = settings.Input.Trim().Trim('"');
        (int w, int h) = MediaProbe.Dimensions(path);
        return ((int)Math.Round(w * scale), (int)Math.Round(h * scale));
    }
}
