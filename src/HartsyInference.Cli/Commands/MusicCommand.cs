using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates music from a text prompt with MusicGen, saving a WAV.</summary>
public sealed class MusicCommand : Command<MusicCommand.Settings>
{
    /// <summary>Options for <c>hartsy music</c>. Inherits the shared placement options so multi-GPU layer
    /// splits (<c>--lm-shard-gpu</c> — YuE Stage-1 un-quantized across two cards) work from the CLI.</summary>
    public sealed class Settings : PlacementCliSettings
    {
        /// <summary>The music description.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The music description (e.g. \"upbeat electronic dance music\").")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id, optionally with a variant (e.g. acestep:turbo, yue:en-cot).</summary>
        [CommandOption("-m|--model")]
        [Description("Model: musicgen, audiogen, acestep, yue, stableaudio, heartmula (optionally 'id:variant', e.g. acestep:turbo). Empty uses musicgen.")]
        public string Model { get; init; } = "musicgen";

        /// <summary>Optional override to a local checkpoint; every catalog model otherwise self-downloads.</summary>
        [CommandOption("--model-path")]
        [Description("Optional local checkpoint override; every catalog model self-downloads when omitted.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Duration in seconds.</summary>
        [CommandOption("-d|--duration")]
        [Description("Duration in seconds.")]
        public int Duration { get; init; } = 10;

        /// <summary>RNG seed; &lt; 0 uses 0.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed.")]
        public int Seed { get; init; } = -1;

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
        if (!CommandRunner.RequireNonEmpty(settings.Prompt, "A prompt is required.", out int exitCode))
            return exitCode;

        // --model-path is an override for a local checkpoint (or a user-placed ACE-Step/YuE folder); it is NOT
        // required — every catalog music model (musicgen, audiogen, acestep, yue, stableaudio, heartmula)
        // self-downloads through the engine's audio cache, same as the speech/transcribe commands.
        ParamState parameters = new ParamState(Modality.Music) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("duration", settings.Duration.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Music);
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        (int? gpu, EngineOptions? engineOptions) = PlacementCli.Build(settings, settings.Backend);
        return CommandRunner.Run(Modality.Music, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false, gpu, engineOptions);
    }
}
