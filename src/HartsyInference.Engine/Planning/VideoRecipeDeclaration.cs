using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Planning;

/// <summary>What planning needs from a video recipe, resolved for one model: its family, the defaults and features of
/// the resolved variant, and the family's constraints.</summary>
public sealed record VideoRecipeDeclaration
{
    /// <summary>The family id the model is served by, after variant routing.</summary>
    public required string FamilyId { get; init; }

    /// <summary>The resolved variant's defaults.</summary>
    public required VideoDefaults Defaults { get; init; }

    /// <summary>The conditioning the resolved variant applies.</summary>
    public required VideoFeatures Features { get; init; }

    /// <summary>The family's generation limits.</summary>
    public required VideoConstraints Constraints { get; init; }

    /// <summary>The resolved variant; null for a family that declares none.</summary>
    public ResolvedModelVariant? Variant { get; init; }

    /// <summary>The identity a plan against this declaration records.</summary>
    public RecipeIdentity Identity => RecipeIdentity.For(FamilyId, Defaults.Revision, Variant);

    /// <summary>The declaration of <paramref name="recipe"/> for <paramref name="variant"/>.</summary>
    public static VideoRecipeDeclaration For(IVideoRecipe recipe, string familyId, ResolvedModelVariant? variant,
        VideoFeatures features)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        ArgumentException.ThrowIfNullOrWhiteSpace(familyId);
        return new VideoRecipeDeclaration
        {
            FamilyId = familyId,
            Defaults = recipe.DefaultsFor(variant),
            Features = features,
            Constraints = recipe.Constraints,
            Variant = variant,
        };
    }
}
