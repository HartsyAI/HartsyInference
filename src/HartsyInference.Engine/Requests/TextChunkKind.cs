namespace HartsyInference.Engine.Requests;

/// <summary>The kind of a streamed <see cref="TextChunk"/>, preserving the provider event vocabulary (chunk / result / status / stopReason / native_tool_call).</summary>
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

    /// <summary>Incremental reasoning text from the model's think block.</summary>
    Reasoning,

    /// <summary>A fragment of a streaming tool call; <see cref="TextChunk.ToolCallIndex"/> says which call.</summary>
    ToolCallDelta,

    /// <summary>The streaming tool call at <see cref="TextChunk.ToolCallIndex"/> was dropped as malformed; discard its deltas.</summary>
    ToolCallAbort,

    /// <summary>Token accounting, carried in <see cref="TextChunk.Usage"/>.</summary>
    Usage,

    /// <summary>The result of a dispatched tool call: <see cref="TextChunk.Text"/> is the result, <see cref="TextChunk.ToolCall"/> the call and <see cref="TextChunk.ToolCallIndex"/> its position in the run.</summary>
    ToolResult,
}
