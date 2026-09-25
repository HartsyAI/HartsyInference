namespace HartsyInference.ModelAssets.Metadata;

/// <summary>Who a published artifact says it is: the fields SwarmUI classifies and displays a model by.</summary>
/// <remarks>Separate from the engine's <c>ModelCatalog</c> entry, which describes what the engine can <i>drive</i>.
/// A model can be drivable without being publishable and the reverse, and this type lives in ModelAssets so the
/// conversion sites in Audio and ModelAssets can reach it — Engine depends on both, not the other way round.</remarks>
public sealed record ArtifactIdentity
{
    /// <summary>The engine's own id for this family, e.g. <c>"krea2"</c> or <c>"kokoro"</c>. Also the stem of the
    /// canonical file name.</summary>
    public required string EngineId { get; init; }

    /// <summary>SwarmUI's <c>T2IModelClass.ID</c>, written verbatim as <c>modelspec.architecture</c>. Looked up
    /// case-insensitively by <c>T2IModelClassSorter</c>, but stored in the registered casing so an artifact we
    /// publish reads the same as one SwarmUI stamped itself.</summary>
    public required string SwarmClassId { get; init; }

    /// <summary>Human title, shown on the model card.</summary>
    public required string DisplayName { get; init; }

    /// <summary>Who made the original weights, not who repacked them.</summary>
    public required string Author { get; init; }

    /// <summary>License id as HuggingFace spells it. <c>"unknown"</c> and multi-source descriptions are not valid
    /// ids — such a model uses <c>"other"</c> and states its terms in the bundle's README.</summary>
    public required string License { get; init; }

    /// <summary>Upstream repo the weights originate from, for provenance. Null when the family has no single one.</summary>
    public string? UpstreamRepo { get; init; }

    /// <summary>The class's declared generation size as <c>"{w}x{h}"</c>, or null for a model with no spatial
    /// output.
    /// <para>Null is load-bearing, not merely absent: <c>IdentifyClassFor</c> accepts a model whose resolution
    /// matches the class standard OR is absent, and for anything else it substitutes a clone carrying the stamped
    /// size with its heuristic matcher disabled. The id still resolves, so this is quieter than a wrong
    /// architecture — the model keeps its class and its parameters — but the class then reports a standard size
    /// nobody declared, and can no longer be matched by shape if its id is ever missing. Audio classes declare
    /// 0x0, so for them any value at all is a mismatch.</para></summary>
    public string? StandardResolution { get; init; }

    /// <summary>Tags for <c>modelspec.tags</c>; the first is the category (<c>audio</c>/<c>image</c>/<c>video</c>).</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>Per-variant class ids, keyed case-insensitively by variant, for a family whose variants register
    /// as different SwarmUI classes (Qwen3-TTS base/custom/design). Absent variants use <see cref="SwarmClassId"/>.</summary>
    public IReadOnlyDictionary<string, string> VariantClassIds { get; init; } = new Dictionary<string, string>();

    /// <summary>This identity with <see cref="SwarmClassId"/> resolved for <paramref name="variant"/>.</summary>
    public ArtifactIdentity ForVariant(string? variant)
    {
        if (string.IsNullOrWhiteSpace(variant))
        {
            return this;
        }
        foreach (KeyValuePair<string, string> entry in VariantClassIds)
        {
            if (string.Equals(entry.Key, variant, StringComparison.OrdinalIgnoreCase))
            {
                return this with { SwarmClassId = entry.Value };
            }
        }
        return this;
    }
}
