using System.Text.Json;
using System.Text.Json.Serialization;

namespace HartsyInference.PhoneGateway.Config;

/// <summary>Source-generated metadata for <see cref="GatewayConfig"/>: camelCase keys, enums by name, comments and
/// trailing commas tolerated so a hand-edited file loads.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    UseStringEnumConverter = true,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true,
    WriteIndented = true)]
[JsonSerializable(typeof(GatewayConfig))]
internal sealed partial class GatewayJsonContext : JsonSerializerContext
{
}
