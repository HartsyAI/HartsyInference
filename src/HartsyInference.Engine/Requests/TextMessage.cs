namespace HartsyInference.Engine.Requests;

/// <summary>One chat message: an author role, its text, any attached images (for vision-language models), and the tool-call fields an assistant or tool turn carries.</summary>
public sealed record TextMessage
{
    /// <summary>The author role.</summary>
    public required TextRole Role { get; init; }

    /// <summary>The message text.</summary>
    public required string Content { get; init; }

    /// <summary>Attached images for multimodal turns; null/empty for text-only.</summary>
    public IReadOnlyList<ImageData>? Images { get; init; }

    /// <summary>Tool calls an <see cref="TextRole.Assistant"/> turn made; null for a plain turn.</summary>
    public IReadOnlyList<NativeToolCall>? ToolCalls { get; init; }

    /// <summary>Id of the assistant tool call a <see cref="TextRole.Tool"/> turn answers.</summary>
    public string? ToolCallId { get; init; }

    /// <summary>Name of the tool a <see cref="TextRole.Tool"/> turn answers for (OpenAI's <c>name</c>).</summary>
    public string? Name { get; init; }

    /// <summary>The model's reasoning for an assistant turn, carried back in a conversation. The prompt builder does not render it yet.</summary>
    public string? ReasoningContent { get; init; }
}
