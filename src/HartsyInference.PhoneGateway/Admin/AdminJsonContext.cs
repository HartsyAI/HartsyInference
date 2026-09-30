using System.Text.Json.Serialization;

namespace HartsyInference.PhoneGateway.Admin;

/// <summary>Source-generated metadata for the admin endpoint's bodies.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    UseStringEnumConverter = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(HealthStatus))]
[JsonSerializable(typeof(PlaceCallRequest))]
[JsonSerializable(typeof(PlaceCallResponse))]
internal sealed partial class AdminJsonContext : JsonSerializerContext
{
}
