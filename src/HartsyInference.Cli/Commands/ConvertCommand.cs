using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Re-voices a source clip with RVC (a user-placed trained voice) or OpenVoice (tone-color transfer from a
/// target reference), saving a WAV.</summary>
public sealed class ConvertCommand : Command<ConvertCommand.Settings>
{
    /// <summary>Options for <c>hartsy convert</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the source WAV to re-voice.</summary>
        [CommandArgument(0, "<audio>")]
        [Description("Path to the source WAV to re-voice.")]
        public string Audio { get; init; } = "";

        /// <summary>Model id, optionally with a variant (e.g. rvc:myvoice).</summary>
        [CommandOption("-m|--model")]
        [Description("Model: rvc:<voice-name> (needs --model-path for the voice), or openvoice.")]
        public string Model { get; init; } = "openvoice";

        /// <summary>Path to the RVC voice checkpoint (.pth/.safetensors) or an OpenVoice override.</summary>
        [CommandOption("--model-path")]
        [Description("Path to the RVC voice checkpoint (.pth/.safetensors); ignored by OpenVoice.")]
        public string? ModelPath { get; init; }

        /// <summary>Target voice reference WAV, for tone-color transfer models (OpenVoice); ignored by RVC.</summary>
        [CommandOption("--target")]
        [Description("Target voice reference WAV (OpenVoice tone-color transfer); ignored by RVC.")]
        public string? Target { get; init; }

        /// <summary>Pitch shift in semitones (RVC).</summary>
        [CommandOption("--pitch-shift")]
        [Description("Pitch shift in semitones (RVC); 0 = no shift.")]
        public double PitchShift { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

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
        if (string.IsNullOrWhiteSpace(settings.Audio))
        {
            AnsiConsole.MarkupLine("[red]A source audio file path is required.[/]");
            return 1;
        }

        ParamState parameters = new ParamState(Modality.VoiceConvert) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        if (settings.Target is { Length: > 0 })
            parameters.Put("target-path", settings.Target);
        parameters.Put("pitch-shift", settings.PitchShift.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.VoiceConvert);
        string label = spec.Catalog?.Id ?? settings.Model;
        string outputDir = settings.Output ?? RepoPaths.OutputRoot();

        return CommandRunner.Run(Modality.VoiceConvert, spec, settings.Audio, parameters, settings.Backend, settings.Quiet,
            outputDir, label, showResponseRule: false);
    }
}
