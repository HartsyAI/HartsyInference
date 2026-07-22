namespace HartsyInference.Engine.Requests;

/// <summary>The kind of a streamed <see cref="TextChunk"/>, preserving the provider event vocabulary
/// (chunk / result / status / stopReason / native_tool_call).</summary>
public enum TextChunkKind
{
    /// <summary>Incremental text to append to the buffer.</summary>
    Chunk,

    /// <summary>Full text that replaces the buffer.</summary>
    Result,

    /// <summary>A backend status update (e.g. "loading_model").</summary>
    Status,

    /// <summary>The finishing stop reason.</summary>
    StopReason,

    /// <summary>A parsed native tool call.</summary>
    NativeToolCall,
}
