using System.Text.Json.Serialization;

namespace HartsyInference.PhoneLink;

/// <summary>Source-generated metadata for the control payloads: camelCase keys, enums by name, nulls omitted. No reflection, so
/// the gateway can trim and the host never walks types on a call thread.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    UseStringEnumConverter = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(CallStartMessage))]
[JsonSerializable(typeof(LinkEventMessage))]
[JsonSerializable(typeof(ToolRequestMessage))]
[JsonSerializable(typeof(ToolResultMessage))]
[JsonSerializable(typeof(LinkErrorMessage))]
internal sealed partial class LinkJsonContext : JsonSerializerContext
{
}
