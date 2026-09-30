namespace HartsyInference.PhoneLink;

/// <summary>Outcome of a tool request; serialized by name in <see cref="ToolResultMessage"/>.</summary>
public enum LinkToolStatus
{
    /// <summary>The tool ran.</summary>
    Ok,
    /// <summary>The tool ran and failed; <see cref="ToolResultMessage.Message"/> says why.</summary>
    Failed,
    /// <summary>The gateway does not implement this tool.</summary>
    Unsupported,
}
