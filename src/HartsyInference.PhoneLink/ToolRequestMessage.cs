using System.Text.Json;

namespace HartsyInference.PhoneLink;

/// <summary>JSON body of <see cref="LinkMessageType.ToolRequest"/>, following the <c>u32 requestId</c> prefix. The host asks the
/// gateway to run a telephony tool.</summary>
public sealed record ToolRequestMessage
{
    /// <summary>Tool name: <c>hangup</c>, <c>send_dtmf</c>, <c>transfer</c>, <c>hold</c>, <c>unhold</c> or <c>play_prompt</c>.</summary>
    public required string Name { get; init; }

    /// <summary>Tool arguments as a JSON object, or null for tools without arguments. Owned by the message, not by the frame
    /// buffer.</summary>
    public JsonElement? Arguments { get; init; }
}
