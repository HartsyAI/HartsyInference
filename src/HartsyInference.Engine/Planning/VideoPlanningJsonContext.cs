using System.Text.Json.Serialization;

namespace HartsyInference.Engine.Planning;

/// <summary>Reflection-free JSON metadata for local video-profile sidecars.</summary>
[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true, UseStringEnumConverter = true)]
[JsonSerializable(typeof(VideoProfileSidecar))]
internal sealed partial class VideoPlanningJsonContext : JsonSerializerContext
{
}
