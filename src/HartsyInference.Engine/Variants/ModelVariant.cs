using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Variants;

/// <summary>One variant a family declares, as data: every signal that may select it. A family lists these in a
/// <see cref="ModelVariantCatalog"/> and <see cref="ModelVariantResolver"/> weighs the signals, so no recipe sniffs
/// filenames or metadata by hand.</summary>
public sealed record ModelVariant
{
    /// <summary>Short stable id (<c>base</c>, <c>edit</c>, <c>turbo</c>); also accepted as a caller hint and matched against <c>hartsy.model_id</c>.</summary>
    public required string Id { get; init; }

    /// <summary>Human-readable name for logs and diagnostics.</summary>
    public required string DisplayName { get; init; }

    /// <summary>Extra caller hints that select this variant, case-insensitive: SwarmUI model-class ids, legacy family ids.</summary>
    public IReadOnlyList<string> HintAliases { get; init; } = [];

    /// <summary><c>modelspec.architecture</c> values that identify this variant.</summary>
    public IReadOnlyList<string> MetadataClassIds { get; init; } = [];

    /// <summary>Tensor names that must all be present for the weights to prove this variant.</summary>
    public IReadOnlyList<string> StructuralMarkers { get; init; } = [];

    /// <summary>Structural rule the marker list cannot express (a key fragment, a shape); combined with the markers by AND.</summary>
    public Func<CheckpointProbe, bool>? StructuralMatch { get; init; }

    /// <summary>Metadata rule beyond <see cref="MetadataClassIds"/> (e.g. a value inside a JSON config string).</summary>
    public Func<CheckpointProbe, bool>? MetadataMatch { get; init; }

    /// <summary>When true only <see cref="ModelVariantSource.Structure"/> can select this variant: its weights carry
    /// modules the variant needs, so a hint or filename naming it on a file without them would load garbage.</summary>
    public bool StructureRequired { get; init; }

    /// <summary>Filename token sets, the last-resort signal: the variant matches when every token of any one set is a
    /// whole token of a file name (split on <c>- _ . space</c>); a token prefixed <c>!</c> must be absent instead.
    /// Empty opts the variant out of filename evidence.</summary>
    public IReadOnlyList<IReadOnlyList<string>> FilenameTokenSets { get; init; } = [];

    /// <summary>A registered recipe family id that serves this variant with its own contract (sampling table, defaults,
    /// catalog entry), e.g. LTX-2.5's <c>ltx-2.5-distilled</c>; null when the declaring recipe serves it.</summary>
    public string? RoutesToFamily { get; init; }

    /// <summary>Whether this variant declares any structural rule.</summary>
    public bool HasStructuralRule => StructuralMarkers.Count > 0 || StructuralMatch is not null;

    /// <summary>Whether <paramref name="hint"/> names this variant by id or alias.</summary>
    public bool MatchesHint(string hint)
    {
        if (string.Equals(Id, hint, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
        foreach (string alias in HintAliases)
        {
            if (string.Equals(alias, hint, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }
}
