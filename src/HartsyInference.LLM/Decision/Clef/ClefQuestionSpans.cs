namespace HartsyInference.LLM.Decision.Clef;

/// <summary>Where one question and its options sit in the encoded record (token spans, end exclusive).</summary>
public sealed record ClefQuestionSpans
{
    public required string QuestionId { get; init; }

    /// <summary>0 = noul, 1 = choice, 2 = score.</summary>
    public required int QuestionType { get; init; }

    public required (int Start, int End) QuestionSpan { get; init; }

    public required (int Start, int End)[] OptionSpans { get; init; }

    public required string[] OptionIds { get; init; }
}
