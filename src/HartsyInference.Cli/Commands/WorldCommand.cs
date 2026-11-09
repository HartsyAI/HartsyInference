using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Rolls out an interactive world model from a first-frame image with an action plan.</summary>
/// <remarks>Oasis does a one-shot batch rollout; DIAMOND runs a genuinely per-frame loop, one real denoise per action.
/// Every option is checked by the engine's world planner before a checkpoint is loaded.
/// Camera file format: a JSON file <c>{"poses": [[16 numbers], ...], "intrinsics": {"fx", "fy", "cx", "cy"}}</c> or a
/// NumPy <c>.npy</c> float32/float64 array shaped (N, 16) or (N, 4, 4). Each pose is a camera-to-world matrix, row-major,
/// OpenCV convention (+X right, +Y down, +Z forward), one per frame; frame 0 is the seed image.</remarks>
public sealed class WorldCommand : Command<WorldCommand.Settings>
{
    /// <summary>Options for <c>hartsy world</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the first-frame image (Oasis expects 640x360; DIAMOND expects 64x64).</summary>
        [CommandArgument(0, "<image>")]
        [Description("Path to the first-frame PNG (Oasis: 640x360; DIAMOND: 64x64).")]
        public string Image { get; init; } = "";

        /// <summary>Model id.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id: oasis or diamond.")]
        public string Model { get; init; } = "oasis";

        /// <summary>Path to the model checkpoint (Oasis DiT, or the DIAMOND denoiser).</summary>
        [CommandOption("--model-path")]
        [Description("Path to the model checkpoint (Oasis DiT .safetensors, or the DIAMOND denoiser .safetensors).")]
        public string? ModelPath { get; init; }

        /// <summary>Path to the Oasis ViT-VAE .safetensors (Oasis only).</summary>
        [CommandOption("--vae-path")]
        [Description("Path to the Oasis ViT-VAE .safetensors (Oasis only; DIAMOND has no VAE).")]
        public string? VaePath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Number of frames to roll out, counting the seed frame; null uses the model default.</summary>
        [CommandOption("--frames")]
        [Description("Frames to roll out, counting the seed frame (default: the model's own count, 16 for Oasis and DIAMOND).")]
        public int? Frames { get; init; }

        /// <summary>Output frame rate; null uses the model default.</summary>
        [CommandOption("--fps")]
        [Description("Output frame rate (default: the model's own rate, 24 for Oasis and DIAMOND).")]
        public int? Fps { get; init; }

        /// <summary>Path to a per-frame camera path (.json or .npy); see the command remarks for the format.</summary>
        [CommandOption("--camera")]
        [Description("Per-frame camera path, one camera-to-world pose per frame (.json or .npy). Models without camera input refuse it.")]
        public string? Camera { get; init; }

        /// <summary>Pinhole intrinsics as fx,fy,cx,cy in pixels of the seed image; needs --camera.</summary>
        [CommandOption("--intrinsics")]
        [Description("Intrinsics as fx,fy,cx,cy in pixels of the seed image. Overrides the camera file's intrinsics; needs --camera.")]
        public string? Intrinsics { get; init; }

        /// <summary>Comma-separated action tokens, cycled to fill the frame count; model-specific default when omitted.</summary>
        /// <remarks>Oasis defaults to "forward"; DIAMOND defaults to a fire/left/right demo cycle.</remarks>
        [CommandOption("--actions")]
        [Description("Comma-separated action tokens, cycled to fill --frames (default: a model-specific demo plan).")]
        public string? Actions { get; init; }

        /// <summary>DDIM/denoising steps per frame; null uses the model default (10 for Oasis).</summary>
        [CommandOption("--steps")]
        [Description("Denoising steps per frame (Oasis only; default 10). DIAMOND fixes its own step count.")]
        public int? Steps { get; init; }

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
        if (!CommandRunner.RequireNonEmpty(settings.Image, "A first-frame image path is required.", out int exitCode))
            return exitCode;

        if (string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            AnsiConsole.MarkupLine("[red]A model checkpoint is required via[/] [#2ea5e0]--model-path[/][red].[/]");
            return 1;
        }

        bool isOasis = string.Equals(settings.Model, "oasis", StringComparison.OrdinalIgnoreCase);
        if (isOasis && string.IsNullOrWhiteSpace(settings.VaePath))
        {
            AnsiConsole.MarkupLine("[red]Oasis needs[/] [#2ea5e0]--model-path[/] [red](DiT) and[/] [#2ea5e0]--vae-path[/] [red](ViT-VAE).[/]");
            return 1;
        }

        if (settings.Intrinsics is not null && settings.Camera is null)
        {
            AnsiConsole.MarkupLine("[red]--intrinsics applies to a camera path; add[/] [#2ea5e0]--camera[/][red].[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.World) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        PutIfSet(parameters, "frames", settings.Frames);
        PutIfSet(parameters, "fps", settings.Fps);
        PutIfSet(parameters, "steps", settings.Steps);
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        parameters.Put("actions", settings.Actions ?? "");
        PutIfSet(parameters, "camera", settings.Camera);
        PutIfSet(parameters, "intrinsics", settings.Intrinsics);

        Dictionary<string, string> aux = isOasis ? new Dictionary<string, string> { ["vae-path"] = settings.VaePath! }
            : new Dictionary<string, string>();
        ModelSpec baseSpec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.World);
        ModelSpec spec = baseSpec with { Aux = aux };
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        return CommandRunner.Run(Modality.World, spec, settings.Image, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }

    /// <summary>Stores an optional option so an unset one reads back as null and the model default applies.</summary>
    private static void PutIfSet(ParamState parameters, string key, int? value)
    {
        if (value is int v)
        {
            parameters.Put(key, v.ToString(CultureInfo.InvariantCulture));
        }
    }

    /// <summary>Stores an optional text option so an unset one reads back as null.</summary>
    private static void PutIfSet(ParamState parameters, string key, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            parameters.Put(key, value);
        }
    }
}
