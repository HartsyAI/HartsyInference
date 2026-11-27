using System.ComponentModel;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Splits a mixed audio clip into stems with Demucs, saving one WAV per stem.</summary>
public sealed class FxSeparateCommand : Command<FxSeparateCommand.Settings>
{
    /// <summary>Options for <c>hartsy fx separate</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to the mixed-audio WAV to separate.</summary>
        [CommandArgument(0, "<audio>")]
        [Description("Path to the mixed-audio WAV to separate.")]
        public string Audio { get; init; } = "";

        /// <summary>Model id, optionally with a variant (e.g. demucs:htdemucs_6s).</summary>
        [CommandOption("-m|--model")]
        [Description("Model: demucs, optionally 'demucs:<variant>' (htdemucs, htdemucs_6s, htdemucs_ft); kim-vocal-2; voice-isolator or voice-isolator-realtime (voice and background stems).")]
        public string Model { get; init; } = "demucs";

        /// <summary>Path to a local htdemucs checkpoint (.th/.safetensors).</summary>
        [CommandOption("--model-path")]
        [Description("Path to a local htdemucs checkpoint (.th/.safetensors).")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector. Honoured by kim-vocal-2 (MDX-Net) and the voice isolator; Demucs always runs on the CPU.</summary>
        /// <remarks>Demucs' STFT/ISTFT has no CUDA/Vulkan implementation yet. The voice isolator runs on the engine's backend (CUDA or Vulkan), with a CPU rerun when the card is full.</remarks>
        [CommandOption("-b|--backend")]
        [Description("Backend selector for kim-vocal-2 and the voice isolator (auto, cpu, cuda, vulkan). auto, the default, uses the GPU when there is one; pass cpu to keep the work off it. Demucs always runs on the CPU (its STFT has no GPU path).")]
        public string Backend { get; init; } = "auto";

        /// <summary>Directory to write the stems into.</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to write the stem WAVs into (defaults to the output root).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress output.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress progress output.")]
        public bool Quiet { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (!CommandRunner.RequireNonEmpty(settings.Audio, "An audio file path is required.", out int exitCode))
            return exitCode;

        // Demucs' STFT/ISTFT (DemucsSpec.Spec) only has a CPU implementation (CudaBackend.Stft throws
        // NotSupportedException), so Demucs forces CPU whatever -b says. The MDX-Net separator and the voice isolator take -b:
        // the isolator's STFT stays on the host, but its network runs on the requested backend.
        bool voiceIsolator = string.Equals(settings.Model, "voice-isolator", StringComparison.OrdinalIgnoreCase)
            || string.Equals(settings.Model, "voice-isolator-realtime", StringComparison.OrdinalIgnoreCase);
        bool usesBackend = voiceIsolator || ModelCatalog.Find(settings.Model)?.Features.HasFlag(CatalogFeatures.MdxNetSeparator) == true;
        string backend = usesBackend ? settings.Backend : "cpu";
        ParamState parameters = new ParamState(Modality.Fx) { Backend = backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("mode", "separate");

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Fx);
        string label = CommandRunner.ResolveLabel(spec, settings.Model);

        return CommandRunner.Run(Modality.Fx, spec, settings.Audio, parameters, backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
