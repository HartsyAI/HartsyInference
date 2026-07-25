using System.ComponentModel;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Transcribes a WAV file to text with Whisper (downloaded and cached on first use).</summary>
public sealed class TranscribeCommand : Command<TranscribeCommand.Settings>
{
    /// <summary>Options for <c>hartsy transcribe</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the WAV file to transcribe.</summary>
        [CommandArgument(0, "<audio>")]
        [Description("Path to a WAV file (auto-resampled to 16 kHz mono).")]
        public string Audio { get; init; } = "";

        /// <summary>Whisper model id or HF repo (e.g. whisper-tiny, openai/whisper-large-v3).</summary>
        [CommandOption("-m|--model")]
        [Description("Whisper model: whisper-tiny/base/small/medium/large-v3, or any HF repo id.")]
        public string Model { get; init; } = "whisper";

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Two-letter source language code.</summary>
        [CommandOption("-l|--language")]
        [Description("Source language (two-letter code).")]
        public string Language { get; init; } = "en";

        /// <summary>Translate to English instead of transcribing.</summary>
        [CommandOption("-t|--translate")]
        [Description("Translate to English instead of transcribing verbatim.")]
        public bool Translate { get; init; }

        /// <summary>Emit segment timestamps.</summary>
        [CommandOption("--timestamps")]
        [Description("Include segment timestamps.")]
        public bool Timestamps { get; init; }

        /// <summary>Directory to save the transcript to.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the transcript (.txt).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress output; print only the transcript.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress; print only the transcript.")]
        public bool Quiet { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (!CommandRunner.RequireNonEmpty(settings.Audio, "An audio file path is required.", out int exitCode))
            return exitCode;

        ParamState parameters = new ParamState(Modality.Transcribe) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("language", settings.Language);
        parameters.Put("translate", settings.Translate ? "true" : "false");
        parameters.Put("timestamps", settings.Timestamps ? "true" : "false");

        // Transcribe models are HF repo ids resolved by the handler, not local paths.
        ModelSpec spec = new ModelSpec { Requested = settings.Model, Modality = Modality.Transcribe, Catalog = ModelCatalog.Find(settings.Model) };

        return CommandRunner.Run(Modality.Transcribe, spec, settings.Audio, parameters, settings.Backend, settings.Quiet,
            settings.Output, settings.Model, showResponseRule: false);
    }
}
