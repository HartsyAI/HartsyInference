namespace HartsyInference.LLM.ChatTemplates;

/// <summary>A rendered-body part: plain text (images already replaced by their placeholder) or a merged tool result.</summary>
internal readonly record struct PromptBlock(bool IsToolResult, string Text, string ToolUseId);
