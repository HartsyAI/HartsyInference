namespace HartsyInference.PhoneLink;

/// <summary>JSON body of <see cref="LinkMessageType.ToolResult"/>, following the <c>u32 requestId</c> prefix that names the
/// request it answers.</summary>
public sealed record ToolResultMessage
{
    public required LinkToolStatus Status { get; init; }

    /// <summary>Human-readable detail, mainly for <see cref="LinkToolStatus.Failed"/>.</summary>
    public string? Message { get; init; }
}
