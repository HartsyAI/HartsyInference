namespace HartsyInference.LLM.ChatTemplates;

/// <summary>What the completion parser needs to know about how the prompt ended; the parser itself is a later change.</summary>
public sealed record ParserInitialState(bool ReasoningOpen);
