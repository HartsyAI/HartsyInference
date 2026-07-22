using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>One prepared text-to-speech job: the request knobs plus the voice reference materialized in the forms
/// the descriptors consume — mono 24 kHz samples, a temp WAV path for pipelines that take a file, and the raw clip
/// for pipelines that need their own sample rate.</summary>
internal sealed record TtsJob
{
    /// <summary>The text to speak.</summary>
    internal required string Text { get; init; }

    /// <summary>Transcript of the reference clip, for models that align against it (e.g. F5).</summary>
    internal string RefText { get; init; } = "";

    /// <summary>The raw voice reference, for descriptors that decode it at their own rate (NeuTTS 16 kHz, GPT-SoVITS 16/32 kHz).</summary>
    internal AudioClip? Reference { get; init; }

    /// <summary>Voice-reference samples at 24 kHz, or null when no reference was supplied.</summary>
    internal float[]? ReferenceMono24k { get; init; }

    /// <summary>Temp 24 kHz mono WAV of the reference, for models that take a file path (e.g. VibeVoice).</summary>
    internal string? ReferenceWavPath { get; init; }

    /// <summary>Built-in voice name; null/empty is the model's default.</summary>
    internal string? Voice { get; init; }

    /// <summary>Speaking-rate multiplier; null is the model default.</summary>
    internal double? Speed { get; init; }

    /// <summary>Chatterbox expressiveness; null is the config default.</summary>
    internal double? Exaggeration { get; init; }

    /// <summary>F5 flow-matching steps (NFE); null is the model default.</summary>
    internal int? NfeStep { get; init; }

    /// <summary>Classifier-free guidance strength; null is the model default.</summary>
    internal double? CfgScale { get; init; }

    /// <summary>Sampling seed; 0 leaves it unset.</summary>
    internal int Seed { get; init; }
}
