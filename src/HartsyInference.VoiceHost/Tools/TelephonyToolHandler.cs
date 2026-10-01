using System.Text.Json;
using HartsyInference.PhoneLink;
using HartsyInference.Tools;

namespace HartsyInference.VoiceHost.Tools;

/// <summary>A telephony tool the gateway runs: the model's arguments go out as a <c>ToolRequest</c> for the call and the
/// gateway's <c>ToolResult</c> comes back to the model as <c>{"status": …, "message": …}</c>. The request and its
/// timeout belong to <paramref name="request"/>.</summary>
internal sealed class TelephonyToolHandler(string name, string description, string jsonSchema,
    Func<ToolRequestMessage, CancellationToken, Task<ToolResultMessage>> request) : IToolHandler
{
    public string Name { get; } = name;

    public string Description { get; } = description;

    public string JsonSchema { get; } = jsonSchema;

    public async Task<string> InvokeAsync(string argumentsJson, CancellationToken cancel)
    {
        ToolRequestMessage message = new() { Name = Name, Arguments = ParseArguments(argumentsJson) };
        ToolResultMessage result = await request(message, cancel).ConfigureAwait(false);
        return ToolOutcome.Write(result.Status.ToString(), result.Message);
    }

    /// <summary>The model's arguments as a JSON object owned by the request, or null when there are none.</summary>
    internal static JsonElement? ParseArguments(string argumentsJson)
    {
        if (string.IsNullOrWhiteSpace(argumentsJson))
        {
            return null;
        }
        using JsonDocument document = JsonDocument.Parse(argumentsJson);
        if (document.RootElement.ValueKind != JsonValueKind.Object)
        {
            throw new ArgumentException($"Tool arguments must be a JSON object, got {document.RootElement.ValueKind}.", nameof(argumentsJson));
        }
        return document.RootElement.EnumerateObject().Any() ? document.RootElement.Clone() : null;
    }
}
