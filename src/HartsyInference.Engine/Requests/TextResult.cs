namespace HartsyInference.Engine.Requests;

/// <summary>The non-streaming result of a text generation: the full text, why it stopped, token counts, and an optional native tool call the model emitted instead of finishing.</summary>
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

    /// <summary>The last native tool call the model emitted; null when it produced plain text. Equal to the last of <see cref="ToolCalls"/>.</summary>
    public NativeToolCall? ToolCall { get; init; }

    /// <summary>Every native tool call the model emitted, in order; null or empty when it produced plain text.</summary>
    public IReadOnlyList<NativeToolCall>? ToolCalls { get; init; }

    /// <summary>Prefill wall time in milliseconds (prompt processing up to the first token); 0 when the path did not time it.</summary>
    public double PrefillMilliseconds { get; init; }

    /// <summary>Decode wall time in milliseconds (first token onward); 0 when the path did not time it.</summary>
    public double DecodeMilliseconds { get; init; }
}
