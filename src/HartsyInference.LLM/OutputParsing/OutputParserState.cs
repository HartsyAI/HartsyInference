namespace HartsyInference.LLM.OutputParsing;

/// <summary>Where a parser starts, so a continuation whose prompt already ended inside <c>&lt;think&gt;</c> resumes in the right section.</summary>
public readonly record struct OutputParserState(OutputParserSection Section)
{
    /// <summary>A fresh non-thinking turn.</summary>
    public static OutputParserState Content => new(OutputParserSection.Content);

    /// <summary>A turn whose prompt ended inside the think block.</summary>
    public static OutputParserState Reasoning => new(OutputParserSection.Reasoning);
}
