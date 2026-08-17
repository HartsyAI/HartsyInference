using System.Text.Json.Serialization;

namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>Source-generated serializer for the profile sidecars — the engine forbids reflection-based JSON, and a
/// store that runs on the always-on wake thread must not depend on a runtime type walk.</summary>
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(SpeakerProfileDocument))]
internal sealed partial class SpeakerProfileJsonContext : JsonSerializerContext
{
}
