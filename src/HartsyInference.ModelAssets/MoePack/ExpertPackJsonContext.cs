using System.Text.Json.Serialization;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>Source-generated JSON for the pack manifest; reflection-free per the repo's JSON rule.</summary>
[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(ExpertPackManifest))]
internal sealed partial class ExpertPackJsonContext : JsonSerializerContext
{
}
