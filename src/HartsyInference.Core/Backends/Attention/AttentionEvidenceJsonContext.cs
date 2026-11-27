using System.Text.Json.Serialization;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>Metadata for formatting completed observations outside execution.</summary>
[JsonSerializable(typeof(AttentionEvidenceEvent[]))]
internal sealed partial class AttentionEvidenceJsonContext : JsonSerializerContext;
