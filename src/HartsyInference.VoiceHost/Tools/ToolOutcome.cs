using System.Text.Json;

namespace HartsyInference.VoiceHost.Tools;

/// <summary>What a telephony tool returns to the model: the gateway's status and its explanation.</summary>
internal sealed record ToolOutcome
{
    public required string Status { get; init; }

    public string? Message { get; init; }

    /// <summary>The outcome as the compact JSON the model reads.</summary>
    public static string Write(string status, string? message) =>
        JsonSerializer.Serialize(new ToolOutcome { Status = status, Message = message }, ToolJsonContext.Default.ToolOutcome);
}
