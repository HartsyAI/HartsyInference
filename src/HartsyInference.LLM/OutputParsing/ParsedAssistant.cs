using HartsyInference.LLM.ChatTemplates;

namespace HartsyInference.LLM.OutputParsing;

/// <summary>The structured assistant turn a parser assembled from its stream; the streamed events replayed in order produce exactly this.</summary>
public sealed record ParsedAssistant(
    string Reasoning,
    string Content,
    IReadOnlyList<ChatToolCall> ToolCalls,
    bool Malformed,
    bool Completed);
