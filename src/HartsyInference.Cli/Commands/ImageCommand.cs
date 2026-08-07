using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates an image from a prompt with any registered diffusion family, saving a PNG.</summary>
public sealed class ImageCommand : Command<ImageCommand.Settings>
{
    /// <summary>Options for <c>hartsy image</c> (inherits the shared multi-GPU placement flags).</summary>
    public sealed class Settings : PlacementCliSettings
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

        /// <summary>Init image for image-to-image; unset runs pure text-to-image.</summary>
        [CommandOption("--init-image")]
        [Description("Path to a PNG/BMP init image; turns the run into image-to-image. Only families whose recipe declares img2img accept it.")]
        public string? InitImage { get; init; }

        /// <summary>Denoise strength over the init image; unset uses the engine's default.</summary>
        [CommandOption("--creativity")]
        [Description("How much of the init image to overwrite, 0..1 (0 keeps it, 1 ignores it). Requires --init-image.")]
        public double? Creativity { get; init; }

        /// <summary>Image-guidance (InstructPix2Pix) scale for dual-CFG edit models; unset uses the family default.</summary>
        [CommandOption("--ip2p-cfg")]
        [Description("Image-guidance scale for dual-CFG edit models (OmniGen2 default 2.0, Boogu default 1.0). Requires --init-image on an edit family.")]
        public double? Ip2pCfg { get; init; }

        /// <summary>Inpaint mask; white regenerates, black preserves.</summary>
        [CommandOption("--mask")]
        [Description("Path to a PNG/BMP inpaint mask (white = regenerate, black = preserve). Requires --init-image.")]
        public string? Mask { get; init; }

        /// <summary>Mask dilation in pixels.</summary>
        [CommandOption("--mask-grow")]
        [Description("Grow the mask by this many pixels before use.")]
        public int? MaskGrow { get; init; }

        /// <summary>Mask blur in pixels, for a softer inpaint transition.</summary>
        [CommandOption("--mask-blur")]
        [Description("Blur the mask by this many pixels for a softer transition.")]
        public int? MaskBlur { get; init; }

        /// <summary>"Inpaint only masked": generate at full resolution over just the masked region.</summary>
        [CommandOption("--mask-shrink-grow")]
        [Description("Inpaint only masked: crop to the mask's bounds grown by this many pixels, generate there at full resolution, and composite back. 0 disables.")]
        public int? MaskShrinkGrow { get; init; }

        /// <summary>FLUX.1 Redux style model (projector) id or path; requires --prompt-image.</summary>
        [CommandOption("--style-model")]
        [Description("FLUX.1 Redux style model id or path (searched under style_models); requires --prompt-image. Flux only.")]
        public string? StyleModel { get; init; }

        /// <summary>Style reference image for the Redux image-prompt slot.</summary>
        [CommandOption("--prompt-image")]
        [Description("Path to the style reference image consumed by --style-model (first image is encoded, matching ComfyUI).")]
        public string[]? PromptImages { get; init; }

        /// <summary>Redux token multiplier.</summary>
        [CommandOption("--redux-multiply")]
        [Description("Redux style-token multiplier (default 1.0).")]
        public double? ReduxMultiply { get; init; }

        /// <summary>Redux merge strength 0..1.</summary>
        [CommandOption("--redux-merge")]
        [Description("Redux merge strength 0..1 (default 1.0; low values like 0.1 recommended).")]
        public double? ReduxMerge { get; init; }

        /// <summary>Schedule fraction at which Redux starts applying.</summary>
        [CommandOption("--redux-apply-start")]
        [Description("Fraction of the schedule before which Redux is not applied (default 0).")]
        public double? ReduxApplyStart { get; init; }

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
        if (!CommandRunner.RequireNonEmpty(settings.Prompt, "A prompt is required.", out int exitCode))
            return exitCode;

        if (!CommandRunner.RequireModelOrPath(settings.Model, settings.ModelPath, "--model", "--model-path", out exitCode))
            return exitCode;

        // Only flags the user actually passed are forwarded; anything omitted stays unset so the engine applies the
        // resolved family's official defaults instead of a generic guess.
        ParamState parameters = new ParamState(Modality.Image) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("negative", settings.Negative);
        parameters.PutIfSet("width", settings.Width);
        parameters.PutIfSet("height", settings.Height);
        parameters.PutIfSet("steps", settings.Steps);
        parameters.PutIfSet("cfg", settings.Cfg);
        parameters.PutIfSet("sampler", settings.Sampler);
        parameters.PutIfSet("scheduler", settings.Scheduler);
        parameters.PutIfSet("sigma-shift", settings.SigmaShift);
        parameters.PutIfSet("init-image", settings.InitImage);
        parameters.PutIfSet("creativity", settings.Creativity);
        parameters.PutIfSet("ip2p-cfg", settings.Ip2pCfg);
        parameters.PutIfSet("mask", settings.Mask);
        parameters.PutIfSet("mask-grow", settings.MaskGrow);
        parameters.PutIfSet("mask-blur", settings.MaskBlur);
        parameters.PutIfSet("mask-shrink-grow", settings.MaskShrinkGrow);
        parameters.PutIfSet("style-model", settings.StyleModel);
        if (settings.PromptImages is { Length: > 0 })
        {
            parameters.Put("prompt-images", string.Join('\n', settings.PromptImages));
        }
        parameters.PutIfSet("redux-multiply", settings.ReduxMultiply);
        parameters.PutIfSet("redux-merge", settings.ReduxMerge);
        parameters.PutIfSet("redux-apply-start", settings.ReduxApplyStart);
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Image);
        string label = CommandRunner.ResolveLabel(spec, settings.Model, settings.ModelPath);

        (int? gpu, EngineOptions? engineOptions) = PlacementCli.Build(settings, settings.Backend,
            HartsyInference.Engine.Modality.Image, PlacementCli.TryModelBytes(settings.Model));
        return CommandRunner.Run(Modality.Image, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false, gpu, engineOptions);
    }
}
