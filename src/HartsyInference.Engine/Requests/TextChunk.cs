namespace HartsyInference.Engine.Requests;

/// <summary>One streamed event from text generation. <see cref="Kind"/> selects which payload is meaningful: <see cref="Text"/> for Chunk/Result/Status, <see cref="Stop"/> for StopReason, <see cref="ToolCall"/> for NativeToolCall.</summary>
public sealed record TextChunk
{
    /// <summary>The event kind.</summary>
    public required TextChunkKind Kind { get; init; }

    /// <summary>Text payload (incremental for Chunk, full for Result, status string for Status); null otherwise.</summary>
    public string? Text { get; init; }

    /// <summary>Stop reason, set when <see cref="Kind"/> is <see cref="TextChunkKind.StopReason"/>.</summary>
    public StopReason? Stop { get; init; }

    /// <summary>Tool call, set when <see cref="Kind"/> is <see cref="TextChunkKind.NativeToolCall"/>.</summary>
    public NativeToolCall? ToolCall { get; init; }
}
