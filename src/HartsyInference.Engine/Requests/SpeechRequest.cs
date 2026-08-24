using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Requests;

/// <summary>Native text-to-speech request. Carries the text plus an optional voice reference (for zero-shot cloning) and the per-model knobs pipelines honor when they support them.</summary>
public sealed record SpeechRequest
{
    /// <summary>The text to speak.</summary>
    public required string Text { get; init; }

    /// <summary>Transcript of the reference clip, for models that use it (e.g. F5); empty otherwise.</summary>
    public string RefText { get; init; } = "";

    /// <summary>Optional voice reference for cloning; null uses the model's built-in / default voice.</summary>
    public AudioClip? Reference { get; init; }

    /// <summary>Built-in voice name (e.g. a Kokoro voice pack); null/empty uses the model default.</summary>
    public string? Voice { get; init; }

    /// <summary>Speaking-rate multiplier; null uses the model default.</summary>
    public double? Speed { get; init; }

    /// <summary>Expressiveness (Chatterbox); null uses the config default.</summary>
    public double? Exaggeration { get; init; }

    /// <summary>Flow-matching steps (F5 NFE); null uses the model default.</summary>
    public int? NfeStep { get; init; }

    /// <summary>Classifier-free guidance strength; null uses the model default.</summary>
    public double? CfgScale { get; init; }

    /// <summary>Loudness-normalize the output (Fish-Speech). Null uses the model default.</summary>
    public bool? NormalizeLoudness { get; init; }

    /// <summary>Speaker identity for models that take a numeric speaker slot (CSM's <c>speaker</c>).</summary>
    public int? SpeakerId { get; init; }

    /// <summary>Sampling temperature; null uses the model default (Bark 0.7 semantic, Dia 1.2).</summary>
    public double? Temperature { get; init; }

    /// <summary>Bark's separate waveform/coarse temperature (upstream default 0.7); null uses the model default.</summary>
    public double? WaveformTemperature { get; init; }

    /// <summary>Nucleus sampling threshold; null uses the model default (Dia 0.95).</summary>
    public double? TopP { get; init; }

    /// <summary>Top-K sampling cutoff; for Dia this is its <c>cfg_filter_top_k</c> (upstream default 45).</summary>
    public int? TopK { get; init; }

    /// <summary>Maximum tokens to generate; null uses the model default (Dia 3072).</summary>
    public int? MaxTokens { get; init; }

    /// <summary>Zonos 8-way emotion vector in the reference order (Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral); renormalized to sum 1. Null uses the model default.</summary>
    public IReadOnlyList<double>? Emotion { get; init; }

    /// <summary>Zonos speaking rate in phonemes per second (0-40; 15 default, 30 very fast, 10 slow).</summary>
    public double? SpeakingRate { get; init; }

    /// <summary>Zonos pitch standard deviation (0-400; 20-45 normal, 60-150 expressive).</summary>
    public double? PitchStd { get; init; }

    /// <summary>Sampling seed for reproducibility; 0 leaves it unset.</summary>
    public int Seed { get; init; }

    /// <summary>Per-request VRAM lever overrides; null follows the backend's policy.</summary>
    public VramOverrides? Vram { get; init; }
}
