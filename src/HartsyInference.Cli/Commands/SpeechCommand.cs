using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Synthesizes speech from text with Piper, saving a WAV.</summary>
public sealed class SpeechCommand : Command<SpeechCommand.Settings>
{
    /// <summary>Options for <c>hartsy speak</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The text to speak.</summary>
        [CommandArgument(0, "<text>")]
        [Description("The text to speak.")]
        public string Text { get; init; } = "";

        /// <summary>Speech model id, optionally with a variant (e.g. piper:en_US-lessac-medium, kokoro:af_heart).</summary>
        [CommandOption("-m|--model|--voice")]
        [Description("Speech model, optionally 'id:variant' (e.g. piper:en_US-lessac-medium, kokoro:af_heart). Empty uses the default.")]
        public string Model { get; init; } = "";

        /// <summary>Path to a local Piper .onnx voice.</summary>
        [CommandOption("--model-path")]
        [Description("Path to a local Piper .onnx voice (with its .onnx.json beside it).")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Speaking rate multiplier (1.0 = normal; higher = faster).</summary>
        [CommandOption("--speed")]
        [Description("Speaking rate (1.0 = normal, higher = faster).")]
        public float Speed { get; init; } = 1.0f;

        /// <summary>Directory to save the WAV to.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the WAV (defaults to the output root).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress output.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress output.")]
        public bool Quiet { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.Text))
        {
            AnsiConsole.MarkupLine("[red]Text to speak is required.[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.Speech) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("speed", settings.Speed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Speech);
        string label = settings.ModelPath is { Length: > 0 } mp ? Path.GetFileName(mp) : (settings.Model.Length > 0 ? settings.Model : "en_US-lessac-medium");
        string outputDir = settings.Output ?? RepoPaths.OutputRoot();

        return CommandRunner.Run(Modality.Speech, spec, settings.Text, parameters, settings.Backend, settings.Quiet,
            outputDir, label, showResponseRule: false);
    }
}
