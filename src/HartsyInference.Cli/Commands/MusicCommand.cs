using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates music from a text prompt with MusicGen, saving a WAV.</summary>
public sealed class MusicCommand : Command<MusicCommand.Settings>
{
    /// <summary>Options for <c>hartsy music</c>; inherits placement options so multi-GPU layer splits (<c>--lm-shard-gpu</c>) work from the CLI.</summary>
    public sealed class Settings : PlacementCliSettings
    {
        /// <summary>The music description.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The music description (e.g. \"upbeat electronic dance music\").")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id, optionally with a variant (e.g. acestep:turbo, yue:en-cot).</summary>
        [CommandOption("-m|--model")]
        [Description("Model: musicgen, audiogen, acestep, yue, yue2, stableaudio, heartmula, minimaxmusic3 (optionally 'id:variant', e.g. acestep:turbo). Empty uses musicgen.")]
        public string Model { get; init; } = "musicgen";

        /// <summary>Optional override to a local checkpoint; every catalog model otherwise self-downloads.</summary>
        [CommandOption("--model-path")]
        [Description("Optional local checkpoint override; every catalog model self-downloads when omitted.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Genre/style tags, separate from the prompt (which YuE reads as LYRICS).</summary>
        [CommandOption("-g|--genre")]
        [Description("Genre/style tags (ACE-Step style prompt; YuE/YuE2 genre tags, e.g. \"uplifting pop female vocal electronic\"). For YuE, YuE2 and MiniMax the PROMPT is the lyrics (use [verse]/[chorus] markers) and this carries the style.")]
        public string Genre { get; init; } = "";

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

        /// <summary>Sampler/solver knobs. Left unset, each model applies its own release default.</summary>
        [CommandOption("--steps")]
        [Description("Denoiser/solver steps (YuE2: acoustic ODE steps, default 32).")]
        public int? Steps { get; init; }

        [CommandOption("--cfg-scale")]
        [Description("Classifier-free guidance scale.")]
        public double? CfgScale { get; init; }

        [CommandOption("--temperature")]
        [Description("Sampling temperature (YuE2: the semantic pass; the score planner has its own --abc-temperature).")]
        public double? Temperature { get; init; }

        [CommandOption("--top-k")]
        [Description("Top-k sampling cutoff.")]
        public int? TopK { get; init; }

        [CommandOption("--top-p")]
        [Description("Nucleus sampling threshold.")]
        public double? TopP { get; init; }

        [CommandOption("--repetition-penalty")]
        [Description("Repetition penalty.")]
        public double? RepetitionPenalty { get; init; }

        /// <summary>YuE2's symbolic planning stage.</summary>
        [CommandOption("--cot")]
        [Description("YuE2 planning mode: full (melody + chords, default), melody (melody only, best for covers), or off (straight to audio).")]
        public string Cot { get; init; } = "";

        [CommandOption("--abc")]
        [Description("YuE2: an ABC score to render verbatim, skipping the planning pass. Accepts a file path or inline notation.")]
        public string Abc { get; init; } = "";

        [CommandOption("--abc-temperature")]
        [Description("YuE2 score planner temperature (default 0.7 — far cooler than the semantic pass).")]
        public double? AbcTemperature { get; init; }

        [CommandOption("--abc-top-p")]
        [Description("YuE2 score planner nucleus threshold (default 0.9).")]
        public double? AbcTopP { get; init; }

        [CommandOption("--abc-top-k")]
        [Description("YuE2 score planner top-k (default 30).")]
        public int? AbcTopK { get; init; }

        [CommandOption("--abc-repetition-penalty")]
        [Description("YuE2 score planner repetition penalty (default 1.005 — a score repeats by design).")]
        public double? AbcRepetitionPenalty { get; init; }

        [CommandOption("--abc-max-tokens")]
        [Description("YuE2 score planner token budget (default 4096).")]
        public int? AbcMaxTokens { get; init; }

        [CommandOption("--penalty-window")]
        [Description("YuE2: how many recent tokens the semantic repetition penalty counts over (default 50).")]
        public int? PenaltyWindow { get; init; }

        [CommandOption("--min-tokens")]
        [Description("YuE2: tokens the semantic pass must emit before it may stop (default 200).")]
        public int? MinTokens { get; init; }

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
        parameters.Put("genre", settings.Genre);
        PutIfSet(parameters, "steps", settings.Steps);
        PutIfSet(parameters, "cfg-scale", settings.CfgScale);
        PutIfSet(parameters, "temperature", settings.Temperature);
        PutIfSet(parameters, "top-k", settings.TopK);
        PutIfSet(parameters, "top-p", settings.TopP);
        PutIfSet(parameters, "repetition-penalty", settings.RepetitionPenalty);
        parameters.Put("cot", settings.Cot);
        // An --abc that names a readable file is the score itself; anything else is inline notation.
        parameters.Put("abc", settings.Abc.Length > 0 && File.Exists(settings.Abc) ? File.ReadAllText(settings.Abc) : settings.Abc);
        PutIfSet(parameters, "abc-temperature", settings.AbcTemperature);
        PutIfSet(parameters, "abc-top-p", settings.AbcTopP);
        PutIfSet(parameters, "abc-top-k", settings.AbcTopK);
        PutIfSet(parameters, "abc-repetition-penalty", settings.AbcRepetitionPenalty);
        PutIfSet(parameters, "abc-max-tokens", settings.AbcMaxTokens);
        PutIfSet(parameters, "penalty-window", settings.PenaltyWindow);
        PutIfSet(parameters, "min-tokens", settings.MinTokens);

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Music);
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        (int? gpu, EngineOptions? engineOptions) = PlacementCli.Build(settings, settings.Backend,
            HartsyInference.Engine.Modality.Music, PlacementCli.TryModelBytes(settings.Model));
        return CommandRunner.Run(Modality.Music, spec, settings.Prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false, gpu, engineOptions);
    }

    /// <summary>Writes an optional flag only when the user actually passed it, so an unset knob stays null in the
    /// request and the model keeps its own default.</summary>
    private static void PutIfSet(ParamState parameters, string key, int? value)
    {
        if (value is { } set) parameters.Put(key, set.ToString(CultureInfo.InvariantCulture));
    }

    private static void PutIfSet(ParamState parameters, string key, double? value)
    {
        if (value is { } set) parameters.Put(key, set.ToString(CultureInfo.InvariantCulture));
    }
}
