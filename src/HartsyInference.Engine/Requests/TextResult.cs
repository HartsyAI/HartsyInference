namespace HartsyInference.Engine.Requests;

/// <summary>The non-streaming result of a text generation: the full text, why it stopped, token counts, and an
/// optional native tool call the model emitted instead of finishing.</summary>
public sealed record TextResult
{
    /// <summary>The generated text.</summary>
    public required string Text { get; init; }

    /// <summary>Why generation stopped.</summary>
    public StopReason Stop { get; init; } = StopReason.Stop;

    /// <summary>Prompt token count.</summary>
    public int PromptTokens { get; init; }

    /// <summary>Generated token count.</summary>
    public int CompletionTokens { get; init; }

    /// <summary>A native tool call the model emitted; null when it produced plain text.</summary>
    public NativeToolCall? ToolCall { get; init; }
}
