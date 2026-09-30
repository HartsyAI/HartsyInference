namespace HartsyInference.Engine.Variants;

/// <summary>The variants one family declares, in match order (most specific first), plus the explicit default used
/// when no evidence matches.</summary>
public sealed class ModelVariantCatalog
{
    /// <summary>Validates that ids are unique, that no hint alias names two variants, and that the default is listed.</summary>
    public ModelVariantCatalog(string familyId, ModelVariant defaultVariant, IReadOnlyList<ModelVariant> variants)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(familyId);
        ArgumentNullException.ThrowIfNull(defaultVariant);
        ArgumentNullException.ThrowIfNull(variants);
        Dictionary<string, string> owners = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (ModelVariant variant in variants)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(variant.Id);
            foreach (string name in variant.HintAliases.Prepend(variant.Id))
            {
                if (owners.TryGetValue(name, out string? owner) && !string.Equals(owner, variant.Id, StringComparison.Ordinal))
                {
                    throw new ArgumentException(
                        $"Variant catalog '{familyId}': '{name}' names both '{owner}' and '{variant.Id}'.", nameof(variants));
                }
                owners[name] = variant.Id;
            }
        }
        if (!variants.Any(variant => ReferenceEquals(variant, defaultVariant)))
        {
            throw new ArgumentException(
                $"Variant catalog '{familyId}': default '{defaultVariant.Id}' is not in the variant list.", nameof(defaultVariant));
        }
        FamilyId = familyId;
        Default = defaultVariant;
        Variants = variants;
    }

    /// <summary>The family these variants belong to.</summary>
    public string FamilyId { get; }

    /// <summary>The variant used when no evidence matches.</summary>
    public ModelVariant Default { get; }

    /// <summary>Every variant, in match order.</summary>
    public IReadOnlyList<ModelVariant> Variants { get; }

    /// <summary>The variant with <paramref name="id"/>, or null.</summary>
    public ModelVariant? Find(string id) =>
        Variants.FirstOrDefault(variant => string.Equals(variant.Id, id, StringComparison.OrdinalIgnoreCase));
}
