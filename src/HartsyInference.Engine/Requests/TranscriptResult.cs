namespace HartsyInference.Engine.Requests;

/// <summary>The result of a transcription: the full text and detected language, plus optional word-level timestamps and speaker diarization when the request asked for them and the model produced them.</summary>
public sealed record TranscriptResult
{
    /// <summary>The full transcript text.</summary>
    public required string Text { get; init; }

    /// <summary>Detected or requested language (ISO code).</summary>
    public string Language { get; init; } = "en";

    /// <summary>Word-level segments with timestamps; null when not requested or unsupported.</summary>
    public IReadOnlyList<WordSegment>? Words { get; init; }

    /// <summary>Speaker-diarized spans; null when not requested or unsupported.</summary>
    public IReadOnlyList<DiarizedSegment>? Speakers { get; init; }

    /// <summary>Length of the audio in seconds.</summary>
    public double Duration { get; init; }

    /// <summary>How many windows the clip was planned into; 1 for a clip that fits the model's window.</summary>
    public int WindowCount { get; init; } = 1;
}
