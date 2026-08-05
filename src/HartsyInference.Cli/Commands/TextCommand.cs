using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates text from a prompt with a local LLM (GGUF or safetensors), streaming tokens as they arrive.</summary>
public sealed class TextCommand : Command<TextCommand.Settings>
{
    /// <summary>Options for <c>hartsy text</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The prompt to complete.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The prompt to complete.")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id (catalog) or path. Optional when <c>--model-path</c> is given.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id (e.g. qwen2, qwen3) or a local path. Optional when --model-path is given.")]
        public string Model { get; init; } = "";

        /// <summary>Explicit checkpoint path (.gguf file or safetensors directory).</summary>
        [CommandOption("--model-path")]
        [Description("Path to a .gguf file or a safetensors model directory.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Device selector for the text service, including the layer-split composite form.</summary>
        [CommandOption("--device")]
        [Description("Device for the LLM: 'cuda:0' pins an ordinal; 'cuda:0+cuda:1' splits the layers across both GPUs by free VRAM (VRAM pooling for models that don't fit one card).")]
        public string? Device { get; init; }

        /// <summary>Maximum new tokens to generate.</summary>
        [CommandOption("--max-tokens")]
        [Description("Maximum number of tokens to generate.")]
        public int MaxTokens { get; init; } = 256;

        /// <summary>Sampling temperature; &lt;= 0 selects greedy decoding.</summary>
        [CommandOption("--temperature")]
        [Description("Sampling temperature; 0 or less means greedy.")]
        public float Temperature { get; init; } = 0.7f;

        /// <summary>Nucleus sampling cutoff.</summary>
        [CommandOption("--top-p")]
        [Description("Nucleus (top-p) sampling cutoff.")]
        public float TopP { get; init; } = 0.95f;

        /// <summary>RNG seed; &lt; 0 uses a reproducible default.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative uses a reproducible default.")]
        public int Seed { get; init; } = -1;

        /// <summary>System prompt applied ahead of the user turn; unset lets the model/template default apply.</summary>
        [CommandOption("--system")]
        [Description("System prompt applied ahead of the user turn.")]
        public string? System { get; init; }

        /// <summary>Image(s) to attach to the prompt, for vision-language models. Repeat the flag for more than one.</summary>
        [CommandOption("-i|--image")]
        [Description("Attach an image (PNG/BMP) for vision-language models. Repeat for multiple images.")]
        public string[] Image { get; init; } = [];

        /// <summary>Top-K sampling cutoff; unset uses the model default.</summary>
        [CommandOption("--top-k")]
        [Description("Top-K sampling cutoff; unset uses the model default.")]
        public int? TopK { get; init; }

        /// <summary>Min-P sampling cutoff; unset uses the model default.</summary>
        [CommandOption("--min-p")]
        [Description("Min-P sampling cutoff; unset uses the model default.")]
        public float? MinP { get; init; }

        /// <summary>Repetition penalty; unset uses the model default.</summary>
        [CommandOption("--repetition-penalty")]
        [Description("Repetition penalty; unset uses the model default.")]
        public float? RepetitionPenalty { get; init; }

        /// <summary>Enable the model's reasoning/"thinking" mode (Qwen3-family); unset lets the model/template default apply.</summary>
        [CommandOption("--thinking")]
        [Description("Enable chat-template reasoning/\"thinking\" mode (Qwen3-family). Unset lets the model/template decide.")]
        public bool? Thinking { get; init; }

        /// <summary>Disable the model's reasoning/"thinking" mode. Mutually exclusive with <see cref="Thinking"/>.</summary>
        [CommandOption("--no-thinking")]
        [Description("Disable chat-template reasoning/\"thinking\" mode.")]
        public bool NoThinking { get; init; }

        /// <summary>Keep weights compressed at their on-disk quant instead of dequantizing to full precision.</summary>
        [CommandOption("--low-vram-quant")]
        [Description("Keep weights compressed at their on-disk quant to reduce VRAM use.")]
        public bool LowVramQuant { get; init; }

        /// <summary>Free the model's device memory after this request completes.</summary>
        [CommandOption("--always-free-memory")]
        [Description("Free the model's device memory after this request completes.")]
        public bool AlwaysFreeMemory { get; init; }

        /// <summary>Directory to save the generated text to (as a .txt file).</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the generated text (.txt).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress and stream output; print only the final text.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress staging/streaming; print only the final text.")]
        public bool Quiet { get; init; }

        /// <summary>Force CUDA-graph decode on; requires greedy sampling and an eligible dense GQA/RoPE model.</summary>
        /// <remarks>Forces --temperature to 0 if a positive temperature was also given.</remarks>
        [CommandOption("--graph-decode")]
        [Description("Force CUDA-graph decode (forces greedy sampling). Falls back silently if the model/backend isn't eligible.")]
        public bool GraphDecode { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (!CommandRunner.RequireNonEmpty(settings.Prompt, "A prompt is required.", out int exitCode))
            return exitCode;

        if (!CommandRunner.RequireModelOrPath(settings.Model, settings.ModelPath, "--model", "--model-path", out exitCode))
            return exitCode;

        if (settings.Thinking == true && settings.NoThinking)
        {
            AnsiConsole.MarkupLine("[red]--thinking and --no-thinking are mutually exclusive.[/]");
            return 1;
        }

        foreach (string imagePath in settings.Image)
        {
            if (!File.Exists(imagePath))
            {
                AnsiConsole.MarkupLine($"[red]Image not found:[/] {imagePath}");
                return 1;
            }
        }

        float temperature = settings.Temperature;
        if (settings.GraphDecode && temperature > 0f)
        {
            AnsiConsole.MarkupLine("[yellow]--graph-decode requires greedy sampling; forcing --temperature to 0.[/]");
            temperature = 0f;
        }

        ParamState parameters = new ParamState(Modality.Text) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        if (settings.Device is { Length: > 0 } device) parameters.Put("device", device);
        parameters.Put("max-tokens", settings.MaxTokens.ToString(CultureInfo.InvariantCulture));
        parameters.Put("temperature", temperature.ToString(CultureInfo.InvariantCulture));
        parameters.Put("top-p", settings.TopP.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        parameters.Put("graph-decode", settings.GraphDecode ? "true" : "false");
        if (settings.System is { Length: > 0 } system)
            parameters.Put("system", system);
        if (settings.Image.Length > 0)
            parameters.Put("image", string.Join('\n', settings.Image));
        if (settings.TopK is { } topK)
            parameters.Put("top-k", topK.ToString(CultureInfo.InvariantCulture));
        if (settings.MinP is { } minP)
            parameters.Put("min-p", minP.ToString(CultureInfo.InvariantCulture));
        if (settings.RepetitionPenalty is { } repetitionPenalty)
            parameters.Put("repetition-penalty", repetitionPenalty.ToString(CultureInfo.InvariantCulture));
        if (settings.Thinking is { } thinking)
            parameters.Put("thinking", thinking ? "true" : "false");
        else if (settings.NoThinking)
            parameters.Put("thinking", "false");
        parameters.Put("low-vram-quant", settings.LowVramQuant ? "true" : "false");
        parameters.Put("always-free-memory", settings.AlwaysFreeMemory ? "true" : "false");

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Text);
        string label = CommandRunner.ResolveLabel(spec, settings.Model, settings.ModelPath);

        return CommandRunner.Run(Modality.Text, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: true);
    }
}
