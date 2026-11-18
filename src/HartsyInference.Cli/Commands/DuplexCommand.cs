using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Plays a recording of the user to a full-duplex speech-to-speech model (Haru-4B) frame by frame and saves what it answers.</summary>
public sealed class DuplexCommand : Command<DuplexCommand.Settings>
{
    /// <summary>Options for <c>hartsy duplex</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The user's recording.</summary>
        [CommandArgument(0, "<input>")]
        [Description("Audio of the user (any rate, mixed down to mono).")]
        public string Input { get; init; } = "";

        /// <summary>Model id with an optional precision, e.g. haru:q8.</summary>
        [CommandOption("-m|--model")]
        [Description("Full-duplex model, optionally 'id:precision' (haru, haru:q8). q8 fits a 12 GB card.")]
        public string Model { get; init; } = "haru";

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>The character card as text.</summary>
        [CommandOption("--card")]
        [Description("Character card of the model, as text.")]
        public string? Card { get; init; }

        /// <summary>A file holding the character card.</summary>
        [CommandOption("--card-file")]
        [Description("Read the character card from a file.")]
        public string? CardFile { get; init; }

        /// <summary>The voice the model speaks with.</summary>
        [CommandOption("--reference")]
        [Description("Voice the model speaks with: 3-180 s of one speaker. Only use voices you have the right to use.")]
        public string? Reference { get; init; }

        /// <summary>The model opens the conversation.</summary>
        [CommandOption("--speak-first")]
        [Description("The model opens the conversation.")]
        public bool SpeakFirst { get; init; }

        /// <summary>Write the user on the left channel and the model on the right.</summary>
        [CommandOption("--stereo")]
        [Description("Write the user on the left channel and the model on the right.")]
        public bool Stereo { get; init; }

        /// <summary>Frames of silence after the recording, so the model can finish.</summary>
        [CommandOption("--tail-seconds")]
        [Description("Seconds of silence appended after the recording so the model can finish speaking (default 0; a 14 s question needed about 20).")]
        public double TailSeconds { get; init; }

        /// <summary>Audio guidance.</summary>
        [CommandOption("--guidance-scale")]
        [Description("Audio guidance scale, 1 to 2 (default 1.5).")]
        public float? Guidance { get; init; }

        /// <summary>Text sampling temperature.</summary>
        [CommandOption("--temperature")]
        [Description("Text sampling temperature; 0 takes the most likely token (default 0.7).")]
        public float? Temperature { get; init; }

        /// <summary>Top-k.</summary>
        [CommandOption("--top-k")]
        [Description("Text sampling top-k (default 25).")]
        public int? TopK { get; init; }

        /// <summary>Top-p.</summary>
        [CommandOption("--top-p")]
        [Description("Text sampling top-p; 1 keeps every token (default 1).")]
        public float? TopP { get; init; }

        /// <summary>Longest conversation.</summary>
        [CommandOption("--max-frames")]
        [Description("Most frames the conversation may run, 7.5 per second (default 27000, an hour). A longer recording is refused.")]
        public int? MaxFrames { get; init; }

        /// <summary>Turn off the repetition penalty.</summary>
        [CommandOption("--no-dry")]
        [Description("Turn off the repetition penalty of the text stream.")]
        public bool NoDry { get; init; }

        /// <summary>Bias towards answering sooner.</summary>
        [CommandOption("--eop-bias")]
        [Description("Added to the logit that makes the model start to speak; above 0 it answers sooner.")]
        public float? EopBias { get; init; }

        /// <summary>Sampler seed.</summary>
        [CommandOption("--seed")]
        [Description("Sampler seed; the same seed and recording repeat the conversation (default 1).")]
        public int? Seed { get; init; }

        /// <summary>Write the words the model spoke, as UTF-8 text.</summary>
        [CommandOption("--transcript")]
        [Description("Write the words the model spoke to this UTF-8 text file.")]
        public string? Transcript { get; init; }

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
        if (!CommandRunner.RequireNonEmpty(settings.Input, "A recording of the user is required.", out int exitCode))
            return exitCode;

        if (!string.IsNullOrEmpty(settings.Card) && !string.IsNullOrEmpty(settings.CardFile))
        {
            AnsiConsole.MarkupLine("[red]Give the character card with --card or --card-file, not both.[/]");
            return 1;
        }
        string card = settings.CardFile is { Length: > 0 } file ? File.ReadAllText(file).Trim() : settings.Card ?? "";
        ParamState parameters = new ParamState(Modality.Speech) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("duplex", "true");
        parameters.Put("input", settings.Input);
        parameters.Put("card", card);
        parameters.Put("tail-seconds", settings.TailSeconds.ToString(CultureInfo.InvariantCulture));
        if (settings.Reference is { Length: > 0 })
            parameters.Put("reference", settings.Reference);
        if (settings.SpeakFirst)
            parameters.Put("speak-first", "true");
        if (settings.Stereo)
            parameters.Put("stereo", "true");
        if (settings.NoDry)
            parameters.Put("no-dry", "true");
        parameters.PutIfSet("guidance-scale", settings.Guidance);
        parameters.PutIfSet("temperature", settings.Temperature);
        parameters.PutIfSet("top-k", settings.TopK);
        parameters.PutIfSet("top-p", settings.TopP);
        parameters.PutIfSet("max-frames", settings.MaxFrames);
        parameters.PutIfSet("eop-bias", settings.EopBias);
        parameters.PutIfSet("seed", settings.Seed);
        if (settings.Transcript is { Length: > 0 })
            parameters.Put("transcript", settings.Transcript);

        ModelSpec spec = ModelResolver.Resolve(settings.Model, null, Modality.Speech);
        string label = CommandRunner.ResolveLabel(spec, settings.Model, null, "haru");
        return CommandRunner.Run(Modality.Speech, spec, Path.GetFileNameWithoutExtension(settings.Input), parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
