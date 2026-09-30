namespace HartsyInference.Tools.Parsing;

/// <summary>Literal text that opens a tool-call span. <paramref name="TextIsPayload"/> means the marker's own characters are the first characters of the payload (the bare <c>{</c> / <c>[</c> forms); <paramref name="LineStartOnly"/> restricts the match to the start of the message or of a line, so ordinary JSON inside prose is left alone.</summary>
public readonly record struct ToolCallMarker(string Text, ToolCallPayload Payload, bool TextIsPayload = false, bool LineStartOnly = false);
