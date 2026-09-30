namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Working copy of a <see cref="ChatMessage"/> used while merging and rendering; <see cref="Blocks"/> mirrors the reference encoder's <c>content_blocks</c>.</summary>
internal sealed class PromptMessage
{
    public PromptMessage(string role) => Role = role;

    public string Role { get; }

    public string Content { get; set; } = string.Empty;

    public List<PromptBlock>? Blocks { get; set; }

    public IReadOnlyList<ToolSpec>? Tools { get; set; }

    public string? ResponseFormatJson { get; set; }

    public IReadOnlyList<ChatToolCall>? ToolCalls { get; set; }

    public string? ReasoningContent { get; set; }

    public string? Task { get; set; }

    public bool WithoutEos { get; set; }

    public PromptMessage WithoutReasoning()
    {
        PromptMessage copy = (PromptMessage)MemberwiseClone();
        copy.ReasoningContent = null;
        return copy;
    }
}
