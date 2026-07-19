namespace HartsyInference.Engine.Requests;

/// <summary>One chat message: an author role, its text, and any attached images (for vision-language models).</summary>
public sealed record TextMessage
{
    /// <summary>The author role.</summary>
    public required TextRole Role { get; init; }

    /// <summary>The message text.</summary>
    public required string Content { get; init; }

    /// <summary>Attached images for multimodal turns; null/empty for text-only.</summary>
    public IReadOnlyList<ImageData>? Images { get; init; }
}
