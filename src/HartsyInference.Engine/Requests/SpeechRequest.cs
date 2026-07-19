namespace HartsyInference.Engine.Requests;

/// <summary>Native text-to-speech request. Carries the text plus an optional voice reference (for zero-shot cloning)
/// and the per-model knobs pipelines honor when they support them.</summary>
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

    /// <summary>Sampling seed for reproducibility; 0 leaves it unset.</summary>
    public int Seed { get; init; }
}
