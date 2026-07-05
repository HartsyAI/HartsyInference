using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
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

        /// <summary>Directory to save the generated text to (as a .txt file).</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the generated text (.txt).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress and stream output; print only the final text.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress staging/streaming; print only the final text.")]
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
            AnsiConsole.MarkupLine("[red]Specify a model with[/] [mediumpurple2]--model[/] [red]or[/] [mediumpurple2]--model-path[/][red].[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Text) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("max-tokens", settings.MaxTokens.ToString(CultureInfo.InvariantCulture));
        parameters.Put("temperature", settings.Temperature.ToString(CultureInfo.InvariantCulture));
        parameters.Put("top-p", settings.TopP.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Text);
        string label = spec.Catalog?.Id ?? (settings.ModelPath is { Length: > 0 } mp ? Path.GetFileName(mp) : settings.Model);

        return CommandRunner.Run(Modality.Text, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: true);
    }
}
