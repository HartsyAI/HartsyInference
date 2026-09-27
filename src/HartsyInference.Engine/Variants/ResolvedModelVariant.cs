namespace HartsyInference.Engine.Variants;

/// <summary>The variant a checkpoint resolved to, with the evidence that decided it.</summary>
/// <param name="FamilyId">The catalog's family id.</param>
/// <param name="Variant">The selected variant.</param>
/// <param name="Source">Which tier of evidence selected it.</param>
/// <param name="Evidence">Readable reason, e.g. <c>modelspec.architecture=qwen-image-edit</c>.</param>
public sealed record ResolvedModelVariant(string FamilyId, ModelVariant Variant, ModelVariantSource Source, string Evidence)
{
    /// <summary>A caller hint the weights contradicted (only set when <see cref="Source"/> is <see cref="ModelVariantSource.Structure"/>).</summary>
    public string? OverriddenHint { get; init; }

    /// <summary>Whether real evidence (weights, caller, metadata) chose the variant rather than a file-name guess or the
    /// default; a contract a weak signal must not impose (e.g. overriding the caller's step count) checks this.</summary>
    public bool IsDefinitive => Source is ModelVariantSource.Structure or ModelVariantSource.CallerHint or ModelVariantSource.Metadata;

    /// <summary>The selected variant's id.</summary>
    public string Id => Variant.Id;

    /// <summary>Pipeline cache-key fragment: the same file resolved to a different variant must not reuse a pipeline.</summary>
    public string CacheToken => $"variant:{FamilyId}/{Variant.Id};";

    /// <summary>Whether the selected variant is <paramref name="variant"/>.</summary>
    public bool Is(ModelVariant variant) => string.Equals(Variant.Id, variant.Id, StringComparison.Ordinal);

    /// <inheritdoc/>
    public override string ToString() => $"{Variant.Id} ({Source}: {Evidence})";
}
