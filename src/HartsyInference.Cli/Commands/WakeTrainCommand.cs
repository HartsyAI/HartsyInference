using System.ComponentModel;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.Engine.Services;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Trains a new wake word from its text alone by synthesizing it across the engine's own TTS voices and fitting a small head on the frozen shared backbone.</summary>
public sealed class WakeTrainCommand : AsyncCommand<WakeTrainCommand.Settings>
{
    /// <summary>Options for <c>hartsy wake train</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The phrase to detect.</summary>
        [CommandArgument(0, "<phrase>")]
        [Description("Wake phrase to train, e.g. \"hey hartsy\".")]
        public string Phrase { get; init; } = "";

        [CommandOption("-n|--name")]
        [Description("Head file name; defaults to the phrase slugified.")]
        public string? Name { get; init; }

        [CommandOption("--speech-model")]
        [Description("TTS model used to synthesize positives.")]
        public string SpeechModel { get; init; } = "kokoro";

        [CommandOption("--voices")]
        [Description("Comma-separated TTS voices. More voices generalize better past any one speaker.")]
        public string? Voices { get; init; }

        [CommandOption("--negative-audio")]
        [Description("Directory of WAV recordings to use as negatives — podcasts, TV, room ambience. The single biggest lever on false accepts.")]
        public string? NegativeAudio { get; init; }

        [CommandOption("--negative-phrases")]
        [Description("Comma-separated phrases to synthesize as negatives; phonetically similar ones help most.")]
        public string? NegativePhrases { get; init; }

        [CommandOption("--epochs")]
        [Description("Training epochs.")]
        public int Epochs { get; init; } = 120;

        [CommandOption("--route")]
        [Description("Opaque tag echoed on the detection event, for sending different words to different agents.")]
        public string? Route { get; init; }

        [CommandOption("-b|--backend")]
        [Description("Backend for the TTS stage: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.Phrase))
        {
            AnsiConsole.MarkupLine("[red]A wake phrase is required.[/]");
            return 1;
        }

        string modelRoot = Path.Combine(RepoPaths.ModelsRoot(), "audio", "wake");
        if (!Directory.Exists(Path.Combine(modelRoot, "backbone")))
        {
            AnsiConsole.MarkupLine($"[red]Wake backbone not found under[/] {modelRoot}/backbone");
            return 1;
        }

        using IInferenceEngine engine = new InferenceEngine(settings.Backend);
        WakeTrainingJob job = new(engine, modelRoot);

        WakeTrainingOptions options = new()
        {
            Phrase = settings.Phrase,
            Name = settings.Name,
            SpeechModel = settings.SpeechModel,
            Epochs = settings.Epochs,
            NegativeAudioDirectory = settings.NegativeAudio,
            NegativePhrases = Split(settings.NegativePhrases),
            Voices = Split(settings.Voices) is { Count: > 0 } voices ? voices : new WakeTrainingOptions { Phrase = settings.Phrase }.Voices,
        };

        Progress<string> progress = new(message => AnsiConsole.MarkupLine($"[grey]{message.EscapeMarkup()}[/]"));
        WakeTrainingResult result = await job.RunAsync(options, progress);

        // Record the measured threshold beside the head. Without this the number is printed once and lost,
        // and the word falls back to a global default that was never measured for it.
        WakeWordConfigStore store = new(modelRoot);
        store.Load();
        store.Set(result.Name, new WakeWordConfig
        {
            Threshold = result.SuggestedThreshold,
            Route = settings.Route,
        });

        AnsiConsole.MarkupLine($"[green]Trained[/] '{result.Name.EscapeMarkup()}' → {result.HeadPath.EscapeMarkup()}");
        AnsiConsole.MarkupLine($"  settings written to {store.Path.EscapeMarkup()}");
        AnsiConsole.MarkupLine($"  windows: {result.PositiveWindows} positive / {result.NegativeWindows} negative");
        AnsiConsole.MarkupLine($"  held-out recall [bold]{result.Recall:P1}[/] at threshold [bold]{result.SuggestedThreshold:F2}[/]");
        AnsiConsole.MarkupLine($"  false accepts [bold]{result.FalseAcceptRate:P2}[/] per window = [bold]{result.FalseAcceptsPerHour:F0}[/] per hour");
        if (result.NegativeWindows == 0)
            AnsiConsole.MarkupLine("[yellow]No negatives were supplied, so the false-accept figure is meaningless. Re-run with --negative-audio pointing at real recordings.[/]");
        else if (result.FalseAcceptsPerHour > 5f)
            AnsiConsole.MarkupLine($"[yellow]{result.FalseAcceptsPerHour:F0} false accepts/hour is too high to live with (openWakeWord targets under 0.5). Your negative set is too small — point --negative-audio at hours of real room audio, TV or podcasts.[/]");
        return 0;
    }

    private static List<string> Split(string? value) =>
        string.IsNullOrWhiteSpace(value) ? [] : [.. value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
}
