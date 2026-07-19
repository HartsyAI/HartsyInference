namespace HartsyInference.Engine.Requests;

/// <summary>Native speech-to-text request: the audio to transcribe plus decode options. The timestamp/diarization
/// toggles request the richer <see cref="TranscriptResult"/> outputs when the model supports them.</summary>
public sealed record AudioRequest
{
    /// <summary>The audio to transcribe.</summary>
    public required AudioClip Audio { get; init; }

    /// <summary>Source language hint (ISO code, e.g. "en"); null/empty lets the model auto-detect.</summary>
    public string Language { get; init; } = "en";

    /// <summary>True to translate to English instead of transcribing in the source language.</summary>
    public bool Translate { get; init; }

    /// <summary>Request word-level start/end timestamps (and word segments) in the result.</summary>
    public bool WordTimestamps { get; init; }

    /// <summary>Request speaker diarization (who-spoke-when) in the result.</summary>
    public bool Diarization { get; init; }
}
