using System.Text.Json;
using System.Text.Json.Serialization;

namespace HartsyInference.VoiceHost.Config;

/// <summary>Source-generated metadata for <see cref="VoiceHostConfig"/>: camelCase keys, comments and trailing commas
/// tolerated so a hand-edited file loads, and an unknown key refused so a misspelt setting fails at start-up instead
/// of being silently ignored.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    WriteIndented = true)]
[JsonSerializable(typeof(VoiceHostConfig))]
internal sealed partial class VoiceHostJsonContext : JsonSerializerContext
{
}
