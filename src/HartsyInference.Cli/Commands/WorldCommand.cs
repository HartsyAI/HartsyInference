using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine.Worlds;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Rolls out an interactive world model from a first-frame image with an action plan or a camera path.</summary>
/// <remarks>Oasis does a one-shot batch rollout; DIAMOND runs a genuinely per-frame loop, one real denoise per action.
/// SANA-WM renders a whole camera-controlled clip in one run:
/// <c>hartsy world -m sana-wm --image demo.png --prompt-file demo.txt (--actions "w-80,dw-40,aw-40" | --camera poses.npy)
/// --intrinsics intrinsics.npy [--frames 161] [--steps 60] [--cfg-scale 5] [--flow-shift 9.8] [--seed 42]</c>. For a model that
/// takes an action-script camera (SANA-WM), <c>--actions</c> is that script; for the others it is the per-frame action plan.
/// SANA-WM streaming generates chunk by chunk and can be driven while it runs:
/// <c>hartsy world -m sana-wm-streaming --image demo.png --prompt-file demo.txt --intrinsics fx,fy,cx,cy
/// [--actions "w-48"] [--frames 241 | --unbounded] [--interactive] [--refiner]</c>. There <c>--actions</c> is the opening move, and with
/// <c>--interactive</c> the keys <c>wasdijkl</c> toggle the held controls, space releases them and <c>q</c> ends the stream.
/// <c>--refiner</c> passes each chunk through the LTX-2 refiner before it is decoded; its files default to the streaming
/// release's layout beside the DiT folder (<c>refiner_diffusers/</c>, <c>gemma3_12b/</c>) and its memory use is set by the
/// <c>vram.sanaWmRefiner*</c> settings.
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
        [CommandArgument(0, "[image]")]
        [Description("Path to the first-frame PNG (Oasis: 640x360; DIAMOND: 64x64; SANA-WM: any size, cover-cropped to 1280x704).")]
        public string? Image { get; init; }

        /// <summary>The first-frame image, as an option instead of the positional argument.</summary>
        [CommandOption("--image")]
        [Description("Path to the first-frame PNG (same as the positional argument).")]
        public string? ImageOption { get; init; }

        /// <summary>Model id.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id: oasis, diamond, sana-wm or sana-wm-streaming.")]
        public string Model { get; init; } = "oasis";

        /// <summary>The prompt the world is seeded with.</summary>
        [CommandOption("--prompt")]
        [Description("Prompt text (SANA-WM). Use --prompt-file for a long prompt.")]
        public string? Prompt { get; init; }

        /// <summary>A UTF-8 text file holding the prompt; surrounding whitespace is trimmed.</summary>
        [CommandOption("--prompt-file")]
        [Description("UTF-8 text file holding the prompt (SANA-WM).")]
        public string? PromptFile { get; init; }

        /// <summary>Path to the model checkpoint (Oasis DiT, or the DIAMOND denoiser).</summary>
        [CommandOption("--model-path")]
        [Description("Path to the model checkpoint (Oasis DiT .safetensors, or the DIAMOND denoiser .safetensors).")]
        public string? ModelPath { get; init; }

        /// <summary>Path to the VAE .safetensors: Oasis's ViT-VAE, or an override of SANA-WM's LTX-2 VAE.</summary>
        [CommandOption("--vae-path")]
        [Description("Path to the VAE .safetensors (Oasis: required ViT-VAE; SANA-WM: overrides vae/ beside the DiT; DIAMOND has none).")]
        public string? VaePath { get; init; }

        /// <summary>Override of SANA-WM's Gemma-2-2B text encoder .safetensors.</summary>
        [CommandOption("--text-encoder-path")]
        [Description("SANA-WM: Gemma-2-2B .safetensors (default: text_encoder/gemma-2-2b-it/ beside the DiT folder).")]
        public string? TextEncoderPath { get; init; }

        /// <summary>Runs the model's refiner stage; refused by a model without one.</summary>
        [CommandOption("--refiner")]
        [Description("Run the model's refiner stage (sana-wm-streaming: the chunk-causal LTX-2 refiner; needs its checkpoints "
            + "and far more VRAM than stage 1).")]
        public bool Refiner { get; init; }

        /// <summary>Override of the refiner transformer checkpoint.</summary>
        [CommandOption("--refiner-path")]
        [Description("Refiner transformer .safetensors (default: refiner_diffusers/transformer/ beside the DiT folder).")]
        public string? RefinerPath { get; init; }

        /// <summary>Override of the refiner's text connectors checkpoint.</summary>
        [CommandOption("--refiner-connectors-path")]
        [Description("Refiner text connectors .safetensors (default: refiner_diffusers/connectors/ beside the DiT folder).")]
        public string? RefinerConnectorsPath { get; init; }

        /// <summary>Override of the refiner's Gemma-3-12B text encoder: a file or a folder of shards.</summary>
        [CommandOption("--refiner-text-encoder-path")]
        [Description("Refiner Gemma-3-12B text encoder: one .safetensors or a folder of shards (default: gemma3_12b/ beside the DiT folder).")]
        public string? RefinerTextEncoderPath { get; init; }

        /// <summary>Override of the refiner text encoder's tokenizer model.</summary>
        [CommandOption("--refiner-tokenizer-path")]
        [Description("Refiner text encoder tokenizer.model (default: beside the refiner text encoder).")]
        public string? RefinerTokenizerPath { get; init; }

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

        /// <summary>Pinhole intrinsics in pixels of the seed image: fx,fy,cx,cy or an intrinsics file.</summary>
        [CommandOption("--intrinsics")]
        [Description("Intrinsics in pixels of the seed image: fx,fy,cx,cy, or a .npy ((4,), (3,3), (F,4), (F,3,3)) or .json file. "
            + "Overrides the camera file's intrinsics; needs --camera or an action-script --actions.")]
        public string? Intrinsics { get; init; }

        /// <summary>Oasis/DIAMOND: comma-separated action tokens cycled over the frames. SANA-WM: the camera action script.</summary>
        /// <remarks>Oasis defaults to "forward"; DIAMOND defaults to a fire/left/right demo cycle.</remarks>
        [CommandOption("--actions")]
        [Description("Oasis/DIAMOND: comma-separated action tokens, cycled to fill --frames (default: a demo plan). SANA-WM: "
            + "the camera script, <keys>-<frames> segments over wasdijkl or none (e.g. w-80,dw-40,aw-40).")]
        public string? Actions { get; init; }

        /// <summary>Classifier-free guidance scale (SANA-WM).</summary>
        [CommandOption("--cfg-scale")]
        [Description("Classifier-free guidance scale (SANA-WM; default 5.0).")]
        public double? CfgScale { get; init; }

        /// <summary>Flow-matching schedule shift (SANA-WM).</summary>
        [CommandOption("--flow-shift")]
        [Description("Flow-matching schedule shift (SANA-WM; default 9.8).")]
        public double? FlowShift { get; init; }

        /// <summary>Upstream's action overlay; not ported, so the flag is refused.</summary>
        [CommandOption("--action-overlay")]
        [Description("Composite upstream's action overlay onto the video (not ported yet; refused).")]
        public bool ActionOverlay { get; init; }

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

        /// <summary>Run guidance as two sequential passes per step; refused by a model that does not declare it.</summary>
        [CommandOption("--sequential-cfg")]
        [Description("Run guidance as two sequential passes per step (only models that support it).")]
        public bool SequentialCfg { get; init; }

        /// <summary>Ask for chunk-by-chunk streaming generation; refused by a model that has no streaming mode.</summary>
        [CommandOption("--streaming")]
        [Description("Generate chunk by chunk, delivering frames as each chunk completes (models with a streaming mode; "
            + "sana-wm-streaming always streams).")]
        public bool Streaming { get; init; }

        /// <summary>Run a streaming session until it is stopped instead of for a frame count.</summary>
        [CommandOption("--unbounded")]
        [Description("Streaming: run until stopped (q in --interactive, or Ctrl+C) instead of for --frames.")]
        public bool Unbounded { get; init; }

        /// <summary>Drive a streaming session from the keyboard while it runs.</summary>
        [CommandOption("--interactive")]
        [Description("Streaming: steer from the keyboard while generating. wasdijkl toggle the held controls, space releases "
            + "them, q or Esc ends the stream and saves what was generated.")]
        public bool Interactive { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (settings.Image is not null && settings.ImageOption is not null)
        {
            AnsiConsole.MarkupLine("[red]Give the first-frame image once: as the argument or with[/] [#2ea5e0]--image[/][red].[/]");
            return 1;
        }
        string image = settings.Image ?? settings.ImageOption ?? "";
        if (!CommandRunner.RequireNonEmpty(image, "A first-frame image path is required.", out int exitCode))
            return exitCode;

        if (settings.ActionOverlay)
        {
            AnsiConsole.MarkupLine("[red]--action-overlay is not supported yet: upstream's overlay compositor is not ported.[/]");
            return 1;
        }

        ModelSpec baseSpec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.World);
        bool fromCatalogAssets = baseSpec.Catalog is { Assets.Count: > 0 };
        if (string.IsNullOrWhiteSpace(settings.ModelPath) && !fromCatalogAssets)
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

        if (settings.Prompt is not null && settings.PromptFile is not null)
        {
            AnsiConsole.MarkupLine("[red]Give the prompt once: with[/] [#2ea5e0]--prompt[/] [red]or[/] [#2ea5e0]--prompt-file[/][red].[/]");
            return 1;
        }
        string? prompt = settings.Prompt;
        if (settings.PromptFile is not null)
        {
            if (!File.Exists(settings.PromptFile))
            {
                AnsiConsole.MarkupLine($"[red]Prompt file not found:[/] {Markup.Escape(settings.PromptFile)}");
                return 1;
            }
            prompt = File.ReadAllText(settings.PromptFile).Trim();
            if (prompt.Length == 0)
            {
                AnsiConsole.MarkupLine($"[red]Prompt file is empty:[/] {Markup.Escape(settings.PromptFile)}");
                return 1;
            }
        }

        ParamState parameters = new ParamState(Modality.World) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        PutIfSet(parameters, "frames", settings.Frames);
        PutIfSet(parameters, "fps", settings.Fps);
        PutIfSet(parameters, "steps", settings.Steps);
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        parameters.Put("actions", settings.Actions ?? "");
        PutIfSet(parameters, "camera", settings.Camera);
        PutIfSet(parameters, "intrinsics", settings.Intrinsics);
        PutIfSet(parameters, "world-prompt", prompt);
        PutIfSet(parameters, "cfg-scale", settings.CfgScale);
        PutIfSet(parameters, "flow-shift", settings.FlowShift);
        parameters.Put("sequential-cfg", settings.SequentialCfg ? "true" : "false");
        parameters.Put("streaming", settings.Streaming ? "true" : "false");
        parameters.Put("unbounded", settings.Unbounded ? "true" : "false");
        parameters.Put("interactive", settings.Interactive ? "true" : "false");
        parameters.Put("refiner", settings.Refiner ? "true" : "false");

        Dictionary<string, string> aux = new Dictionary<string, string>();
        if (!string.IsNullOrWhiteSpace(settings.VaePath))
            aux[WorldComponentResolver.VaeAuxKey] = settings.VaePath;
        if (!string.IsNullOrWhiteSpace(settings.TextEncoderPath))
            aux[WorldComponentResolver.TextEncoderAuxKey] = settings.TextEncoderPath;
        if (!string.IsNullOrWhiteSpace(settings.RefinerPath))
            aux[WorldComponentResolver.RefinerAuxKey] = settings.RefinerPath;
        if (!string.IsNullOrWhiteSpace(settings.RefinerConnectorsPath))
            aux[WorldComponentResolver.RefinerConnectorsAuxKey] = settings.RefinerConnectorsPath;
        if (!string.IsNullOrWhiteSpace(settings.RefinerTextEncoderPath))
            aux[WorldComponentResolver.RefinerTextEncoderAuxKey] = settings.RefinerTextEncoderPath;
        if (!string.IsNullOrWhiteSpace(settings.RefinerTokenizerPath))
            aux[WorldComponentResolver.RefinerTokenizerAuxKey] = settings.RefinerTokenizerPath;
        ModelSpec spec = baseSpec with { Aux = aux };
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        return CommandRunner.Run(Modality.World, spec, image, parameters, settings.Backend, settings.Quiet,
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

    /// <summary>Stores an optional number so an unset one reads back as null and the model default applies.</summary>
    private static void PutIfSet(ParamState parameters, string key, double? value)
    {
        if (value is double v)
        {
            parameters.Put(key, v.ToString("R", CultureInfo.InvariantCulture));
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
