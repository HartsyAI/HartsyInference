namespace HartsyInference.LLM.Decision.Clef;

/// <summary>A record tokenized for the backbone, with the span of every question and option.</summary>
public sealed record ClefEncodedRecord
{
    public required int[] InputIds { get; init; }

    public required ClefQuestionSpans[] Questions { get; init; }
}
