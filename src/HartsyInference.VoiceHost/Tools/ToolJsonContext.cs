using System.Text.Json.Serialization;

namespace HartsyInference.VoiceHost.Tools;

/// <summary>Source-generated metadata for the tool results the model reads: compact, camelCase, nulls omitted.</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(ToolOutcome))]
[JsonSerializable(typeof(ClockReading))]
internal sealed partial class ToolJsonContext : JsonSerializerContext
{
}
