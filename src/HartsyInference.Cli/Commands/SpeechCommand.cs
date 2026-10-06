using System.ComponentModel;
using System.Globalization;
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

        /// <summary>Speech model id, optionally with a variant (e.g. piper:en_US-lessac-medium, whisper:medium.en).</summary>
        [CommandOption("-m|--model")]
        [Description("Speech model, optionally 'id:variant' (e.g. piper:en_US-lessac-medium). Empty uses the default.")]
        public string Model { get; init; } = "";

        /// <summary>Built-in voice/speaker name within the selected model; distinct from --model.</summary>
        /// <remarks>E.g. a Kokoro voice pack, a PocketTTS voice, or Spark-TTS gender male/female. Empty uses the model's own default.</remarks>
        [CommandOption("--voice")]
        [Description("Built-in voice/speaker name (Kokoro voice pack e.g. af_heart, PocketTTS voice e.g. alba, Spark-TTS gender male/female). Empty uses the model default.")]
        public string? Voice { get; init; }

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

        /// <summary>Path to a voice-reference WAV, for cloning models (Zonos, F5-TTS, CosyVoice, Chatterbox, …).</summary>
        [CommandOption("--reference")]
        [Description("Voice-reference WAV for cloning models (Zonos, F5-TTS, CosyVoice, Chatterbox, VibeVoice, …).")]
        public string? Reference { get; init; }

        /// <summary>Transcript of the reference clip, for models that align against it (F5-TTS).</summary>
        [CommandOption("--ref-text")]
        [Description("Transcript of --reference, for models that need it (F5-TTS).")]
        public string? RefText { get; init; }

        /// <summary>Expressiveness knob (Chatterbox).</summary>
        [CommandOption("--exaggeration")]
        [Description("Expressiveness (Chatterbox); model default when unset.")]
        public float? Exaggeration { get; init; }

        /// <summary>Flow-matching steps (F5-TTS NFE).</summary>
        [CommandOption("--nfe-step")]
        [Description("Flow-matching steps (F5-TTS); model default when unset.")]
        public int? NfeStep { get; init; }

        /// <summary>Classifier-free guidance strength.</summary>
        [CommandOption("--cfg-scale")]
        [Description("Classifier-free guidance strength; model default when unset.")]
        public float? CfgScale { get; init; }

        /// <summary>AuK instruction: the edit command when a reference is given, the voice description otherwise.</summary>
        [CommandOption("--instruction")]
        [Description("AuK instruction: with --reference the verbatim edit command, without it the voice description for the text.")]
        public string? Instruction { get; init; }

        /// <summary>Output length in seconds, for models that need it (AuK without a reference).</summary>
        [CommandOption("--duration")]
        [Description("Output length in seconds (AuK without a reference clip; at most 30).")]
        public float? Duration { get; init; }

        /// <summary>IndexTTS-2 emotion: 8 comma-separated weights.</summary>
        [CommandOption("--emotion")]
        [Description("IndexTTS-2 emotion vector: 8 comma-separated weights in IndexTTS-2 order (happy,angry,sad,afraid,disgusted,melancholic,surprised,calm; 0-1.2 each). Other models that read an emotion vector (Zonos) use a different order.")]
        public string? Emotion { get; init; }

        /// <summary>IndexTTS-2 emotion-reference WAV.</summary>
        [CommandOption("--emotion-reference")]
        [Description("IndexTTS-2: a separate WAV whose emotion (not voice) the speech adopts.")]
        public string? EmotionReference { get; init; }

        /// <summary>IndexTTS-2 emotion strength.</summary>
        [CommandOption("--emotion-alpha")]
        [Description("IndexTTS-2: emotion strength 0-1 (default 1).")]
        public float? EmotionAlpha { get; init; }

        /// <summary>IndexTTS-2 free-text emotion.</summary>
        [CommandOption("--emotion-text")]
        [Description("IndexTTS-2: describe the wanted emotion in words (classified by the bundled QwenEmotion model).")]
        public string? EmotionText { get; init; }

        /// <summary>IndexTTS-2 emotion from the spoken text itself.</summary>
        [CommandOption("--emotion-from-text")]
        [Description("IndexTTS-2: infer the emotion from the text being spoken.")]
        public bool EmotionFromText { get; init; }

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
        if (!CommandRunner.RequireNonEmpty(settings.Text, "Text to speak is required.", out int exitCode))
            return exitCode;

        ParamState parameters = new ParamState(Modality.Speech) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        parameters.Put("speed", settings.Speed.ToString(CultureInfo.InvariantCulture));
        if (settings.Voice is { Length: > 0 })
            parameters.Put("voice", settings.Voice);
        if (settings.Reference is { Length: > 0 })
            parameters.Put("reference", settings.Reference);
        if (settings.RefText is { Length: > 0 })
            parameters.Put("ref-text", settings.RefText);
        parameters.PutIfSet("exaggeration", settings.Exaggeration);
        parameters.PutIfSet("nfe-step", settings.NfeStep);
        parameters.PutIfSet("cfg-scale", settings.CfgScale);
        if (settings.Instruction is { Length: > 0 })
            parameters.Put("instruction", settings.Instruction);
        parameters.PutIfSet("duration", settings.Duration);
        if (settings.Emotion is { Length: > 0 })
            parameters.Put("emotion", settings.Emotion);
        if (settings.EmotionReference is { Length: > 0 })
            parameters.Put("emotion-reference", settings.EmotionReference);
        parameters.PutIfSet("emotion-alpha", settings.EmotionAlpha);
        if (settings.EmotionText is { Length: > 0 })
            parameters.Put("emotion-text", settings.EmotionText);
        if (settings.EmotionFromText)
            parameters.Put("emotion-from-text", "true");

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Speech);
        string label = CommandRunner.ResolveLabel(spec, settings.Model, settings.ModelPath, "en_US-lessac-medium");

        return CommandRunner.Run(Modality.Speech, spec, settings.Text, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: false);
    }
}
