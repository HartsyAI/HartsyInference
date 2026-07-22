namespace HartsyInference.Engine.Requests;

/// <summary>Why a generation stopped.</summary>
public enum StopReason
{
    /// <summary>Natural end-of-sequence.</summary>
    Stop,

    /// <summary>Hit the max-token budget.</summary>
    Length,

    /// <summary>Stopped to emit a tool call.</summary>
    ToolCall,

    /// <summary>Cancelled by the caller.</summary>
    Cancelled,

    /// <summary>Ended on an error.</summary>
    Error,
}
