using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.Engine;

/// <summary>One derivative of a catalog model (a different quantization or conversion of the same weights), with the components it ships.</summary>
public sealed record CatalogVariant
{
    /// <summary>Short selector within the model, e.g. "official" or "mlx-4bit".</summary>
    public required string Id { get; init; }

    /// <summary>Human-readable name for tables and menus.</summary>
    public required string DisplayName { get; init; }

    /// <summary>The pinned download, or null when no preset download is defined.</summary>
    public required CatalogShardSet Source { get; init; }

    /// <summary>The safetensors companion naming, or null for a non-safetensors format such as GGUF.</summary>
    public QuantFlavor? Flavor { get; init; }

    /// <summary>Components the variant ships (a partial component is not listed; see <see cref="Refusals"/>).</summary>
    public required CatalogComponents Components { get; init; }

    /// <summary>True when the vision tower ships as its own file rather than inside the main checkpoint.</summary>
    public bool VisionSeparateFile { get; init; }

    /// <summary>Why a component that is missing or partial cannot be used, keyed by that component.</summary>
    public IReadOnlyDictionary<CatalogComponents, string> Refusals { get; init; } = new Dictionary<CatalogComponents, string>();

    /// <summary>Free-form provenance and caveats, such as index quirks found in the real repo.</summary>
    public string? Notes { get; init; }
}
