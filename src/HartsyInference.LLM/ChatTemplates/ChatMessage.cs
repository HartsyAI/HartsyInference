namespace HartsyInference.LLM.ChatTemplates;

/// <summary>A single conversation turn. <paramref name="Role"/> is "system", "user", "assistant", "tool" or "latest_reminder"; the optional members carry the structured fields used by richer prompt formats.</summary>
public sealed record ChatMessage(string Role, string Content)
{
    /// <summary>Structured body; for a user turn it replaces <see cref="Content"/> when non-empty.</summary>
    public IReadOnlyList<ContentBlock>? Blocks { get; init; }

    /// <summary>Tool calls made by an assistant turn.</summary>
    public IReadOnlyList<ChatToolCall>? ToolCalls { get; init; }

    /// <summary>Id of the assistant tool call a "tool" turn answers.</summary>
    public string? ToolCallId { get; init; }

    /// <summary>Assistant reasoning text that precedes the visible answer.</summary>
    public string? ReasoningContent { get; init; }

    /// <summary>Internal classification task (action, query, authority, domain, title, read_url) requested for this turn.</summary>
    public string? Task { get; init; }

    /// <summary>Tool schemas defined at this turn (a system message renders them).</summary>
    public IReadOnlyList<ToolSpec>? Tools { get; init; }

    /// <summary>JSON response schema a system turn asks the model to follow.</summary>
    public string? ResponseFormatJson { get; init; }

    /// <summary>Renders an assistant turn without its end-of-sentence token (a prefill to continue).</summary>
    public bool WithoutEos { get; init; }

    /// <summary>Creates a "system" turn.</summary>
    public static ChatMessage System(string c) => new("system", c);

    /// <summary>Creates a "user" turn.</summary>
    public static ChatMessage User(string c) => new("user", c);

    /// <summary>Creates an "assistant" turn.</summary>
    public static ChatMessage Assistant(string c) => new("assistant", c);

    /// <summary>Creates a "tool" turn answering <paramref name="toolCallId"/>.</summary>
    public static ChatMessage Tool(string toolCallId, string c) => new("tool", c) { ToolCallId = toolCallId };

    /// <summary>Creates a "latest_reminder" turn.</summary>
    public static ChatMessage LatestReminder(string c) => new("latest_reminder", c);
}
