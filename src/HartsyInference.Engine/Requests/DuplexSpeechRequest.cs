namespace HartsyInference.Engine.Requests;

/// <summary>A full-duplex spoken conversation: a character card, an optional voice reference, and how the model samples. The model
/// listens and speaks at once, so there is no text to read aloud; the user's audio arrives frame by frame.</summary>
public sealed record DuplexSpeechRequest
{
    /// <summary>The character card, as text; empty for none.</summary>
    public string Card { get; init; } = "";

    /// <summary>The voice the model speaks with: one speaker, 3 seconds to 3 minutes of clean speech. Null uses the model's own.</summary>
    public AudioClip? Reference { get; init; }

    /// <summary>Audio guidance weight, 1 to 2.</summary>
    public float Guidance { get; init; } = 1.5f;

    /// <summary>Text sampling temperature; zero or less takes the most likely token.</summary>
    public float Temperature { get; init; } = 0.7f;

    public int TopK { get; init; } = 25;

    public float TopP { get; init; } = 1.0f;

    /// <summary>Penalize repeating a phrase the model has already said.</summary>
    public bool RepetitionPenalty { get; init; } = true;

    /// <summary>Added to the logit that makes the model start to speak: above zero it answers sooner.</summary>
    public float EopBias { get; init; }

    /// <summary>The model opens the conversation.</summary>
    public bool SpeakFirst { get; init; }

    /// <summary>Frames after which the conversation ends: 27,000 is an hour.</summary>
    public int MaxFrames { get; init; } = 27_000;

    /// <summary>Seed of the sampler; the same seed and the same audio repeat the conversation.</summary>
    public int Seed { get; init; } = 1;
}
