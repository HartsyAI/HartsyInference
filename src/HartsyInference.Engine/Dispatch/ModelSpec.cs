using HartsyInference.Engine;

namespace HartsyInference.Engine.Dispatch;

/// <summary>A resolved request to load a model: what the user asked for, the matched catalog entry (if any), and the local path it resolved to. Produced by <see cref="ModelResolver"/> and consumed by a handler's <c>Load</c>.</summary>
public sealed record ModelSpec
{
    /// <summary>The original <c>--model</c> token (catalog id, local path, or HF repo id).</summary>
    public required string Requested { get; init; }

    /// <summary>The modality this model is being loaded for.</summary>
    public required Modality Modality { get; init; }

    /// <summary>The catalog entry matched by id, or null when the request was a raw path.</summary>
    public CatalogEntry? Catalog { get; init; }

    /// <summary>The resolved on-disk path (file or directory), or null when nothing local was found.</summary>
    public string? LocalPath { get; init; }

    /// <summary>Auxiliary load-time paths for multi-file models (e.g. "text-encoder-path", "vae-path", "tokenizer-path"), keyed by role. Empty for single-file models.</summary>
    public IReadOnlyDictionary<string, string> Aux { get; init; } = new Dictionary<string, string>();
}

