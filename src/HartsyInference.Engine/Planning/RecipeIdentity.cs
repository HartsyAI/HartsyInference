using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Planning;

/// <summary>Which recipe, revision and variant a generation was planned against, so a result can be traced back to
/// the exact defaults that produced it after those defaults change.</summary>
public sealed record RecipeIdentity
{
    /// <summary>The recipe family that served the model.</summary>
    public required string FamilyId { get; init; }

    /// <summary>The revision of the defaults the plan used.</summary>
    public required int Revision { get; init; }

    /// <summary>The variant the model resolved to; null for a family that declares none.</summary>
    public string? VariantId { get; init; }

    /// <summary>The evidence that selected <see cref="VariantId"/>; a filename guess or a catalog default is visible here.</summary>
    public ModelVariantSource? VariantSource { get; init; }

    /// <summary>The checkpoint profile whose defaults overlaid the recipe's; null when none applied.</summary>
    public string? ProfileId { get; init; }

    /// <summary>The identity for <paramref name="familyId"/> at <paramref name="revision"/> with its resolved variant.</summary>
    public static RecipeIdentity For(string familyId, int revision, ResolvedModelVariant? variant) => new RecipeIdentity
    {
        FamilyId = familyId,
        Revision = revision,
        VariantId = variant?.Id,
        VariantSource = variant?.Source,
    };

    /// <inheritdoc/>
    public override string ToString() =>
        $"{FamilyId} r{Revision}{(VariantId is null ? "" : $" variant {VariantId} ({VariantSource})")}"
        + (ProfileId is null ? "" : $" profile {ProfileId}");
}
