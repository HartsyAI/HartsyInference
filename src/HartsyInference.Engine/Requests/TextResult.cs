namespace HartsyInference.Engine.Requests;

/// <summary>The non-streaming result of a text generation: the full text, why it stopped, token counts, and an optional native tool call the model emitted instead of finishing.</summary>
public sealed record TextResult
{
    /// <summary>The generated text: the answer alone when the model reasoned first (see <see cref="Reasoning"/>).</summary>
    public required string Text { get; init; }

    /// <summary>The reasoning the model wrote before its answer when its prompt ended inside a think block; empty otherwise. A reply that ran out
    /// of tokens mid-reasoning keeps its partial reasoning here with <see cref="Stop"/> = <see cref="StopReason.Length"/> and an empty
    /// <see cref="Text"/>.</summary>
    public string Reasoning { get; init; } = "";

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
