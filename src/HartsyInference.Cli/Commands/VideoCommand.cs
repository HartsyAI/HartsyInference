using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates a video (BMP frame sequence) from a prompt with LTX-Video. CUDA-only, validation-pending.</summary>
public sealed class VideoCommand : Command<VideoCommand.Settings>
{
    /// <summary>Options for <c>hartsy video</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The video description.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The video description.")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id (ltx-video).")]
        public string Model { get; init; } = "ltx-video";

        /// <summary>Path to the LTX .safetensors checkpoint.</summary>
        [CommandOption("--model-path")]
        [Description("Path to the LTX .safetensors checkpoint.")]
        public string? ModelPath { get; init; }

        /// <summary>Path to a standalone T5-XXL .safetensors.</summary>
        [CommandOption("--text-encoder-path")]
        [Description("Path to a standalone T5-XXL .safetensors.")]
        public string? TextEncoderPath { get; init; }

        /// <summary>Path to the T5 SentencePiece model (spiece.model).</summary>
        [CommandOption("--tokenizer-path")]
        [Description("Path to the T5 SentencePiece model (spiece.model).")]
        public string? TokenizerPath { get; init; }

        /// <summary>Compute backend (must be cuda for video).</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend (video requires cuda).")]
        public string Backend { get; init; } = "cuda";

        /// <summary>Negative prompt.</summary>
        [CommandOption("-n|--negative")]
        [Description("Negative prompt.")]
        public string Negative { get; init; } = "";

        /// <summary>Frame width (divisible by 32).</summary>
        [CommandOption("--width")]
        [Description("Frame width (divisible by 32).")]
        public int Width { get; init; } = 704;

        /// <summary>Frame height (divisible by 32).</summary>
        [CommandOption("--height")]
        [Description("Frame height (divisible by 32).")]
        public int Height { get; init; } = 480;

        /// <summary>Number of frames ((n-1) divisible by 8).</summary>
        [CommandOption("--frames")]
        [Description("Number of frames ((n-1) divisible by 8).")]
        public int Frames { get; init; } = 25;

        /// <summary>Denoising steps.</summary>
        [CommandOption("--steps")]
        [Description("Denoising steps.")]
        public int Steps { get; init; } = 30;

        /// <summary>Frames per second for playback naming.</summary>
        [CommandOption("--fps")]
        [Description("Frame rate.")]
        public int Fps { get; init; } = 25;

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

        if (string.IsNullOrWhiteSpace(settings.ModelPath) || string.IsNullOrWhiteSpace(settings.TextEncoderPath) || string.IsNullOrWhiteSpace(settings.TokenizerPath))
        {
            AnsiConsole.MarkupLine("[red]Video needs[/] [mediumpurple2]--model-path[/][red],[/] [mediumpurple2]--text-encoder-path[/] [red]and[/] [mediumpurple2]--tokenizer-path[/][red].[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Video) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("negative", settings.Negative);
        parameters.Put("width", settings.Width.ToString(CultureInfo.InvariantCulture));
        parameters.Put("height", settings.Height.ToString(CultureInfo.InvariantCulture));
        parameters.Put("frames", settings.Frames.ToString(CultureInfo.InvariantCulture));
        parameters.Put("steps", settings.Steps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("fps", settings.Fps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        Dictionary<string, string> aux = new Dictionary<string, string>
        {
            ["text-encoder-path"] = settings.TextEncoderPath!,
            ["tokenizer-path"] = settings.TokenizerPath!,
        };
        ModelSpec baseSpec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Video);
        ModelSpec spec = baseSpec with { Aux = aux };
        string label = spec.Catalog?.Id ?? settings.Model;

        return CommandRunner.Run(Modality.Video, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
