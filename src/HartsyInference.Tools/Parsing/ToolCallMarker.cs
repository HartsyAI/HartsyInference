namespace HartsyInference.Tools.Parsing;

/// <summary>Literal text that opens a tool-call span. <paramref name="TextIsPayload"/> means the marker's own characters are the first characters of the payload (the bare <c>{</c> / <c>[</c> forms); <paramref name="LineStartOnly"/> restricts the match to the start of the message or of a line; <paramref name="Strict"/> marks a bare form that can also occur in ordinary output, so its span must name a known tool (when the parser has the list) and a JSON payload must open with <c>"name"</c>, else it is forwarded as text at once.</summary>
public readonly record struct ToolCallMarker(string Text, ToolCallPayload Payload, bool TextIsPayload = false, bool LineStartOnly = false, bool Strict = false);
