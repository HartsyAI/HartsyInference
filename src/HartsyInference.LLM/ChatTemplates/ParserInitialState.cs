using HartsyInference.LLM.OutputParsing;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>What the completion parser needs to know about how the prompt ended.</summary>
public sealed record ParserInitialState(bool ReasoningOpen)
{
    /// <summary>The parser section a completion of this prompt starts in.</summary>
    public OutputParserState ToParserState() => ReasoningOpen ? OutputParserState.Reasoning : OutputParserState.Content;
}
