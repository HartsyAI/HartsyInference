namespace HartsyInference.LLM.OutputParsing;

/// <summary>One parsed event. <paramref name="Text"/> is the delta, tool name or malformed reason; <paramref name="ToolCallIndex"/> is the zero-based call position (-1 when not call-related); <paramref name="Namespace"/> qualifies a ToolCallBegin.</summary>
public readonly record struct ParsedEvent(ParsedEventKind Kind, string? Text = null, int ToolCallIndex = -1, string? Namespace = null);
