using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Catalog;

/// <summary>Finds a family's prompt profile by the engine id a published artifact carries, so a conversion site can stamp it
/// into the header without constructing a pipeline.</summary>
public static class RecipePrompts
{
    /// <summary>The <see cref="PromptProfile"/> of the image or video recipe that handles <paramref name="engineId"/>, or null
    /// when no recipe handles it or the recipe declares none. An id that names one of the recipe's variants (a fine-tune such
    /// as <c>iris-z-small</c>) gets that variant's profile; any other id gets the family-level one, which leaves a field
    /// unset where the variants differ, since a file does not say which variant it is.</summary>
    public static PromptProfile? Find(string engineId)
    {
        if (string.IsNullOrWhiteSpace(engineId))
        {
            return null;
        }
        IArchitectureRecipe? image = RecipeRegistry.Resolve(engineId);
        if (image is not null)
        {
            return image.PromptFor(VariantNamed(image.Variants, engineId));
        }
        IVideoRecipe? video = VideoRecipeRegistry.Resolve(engineId);
        return video?.PromptFor(VariantNamed(video.Variants, engineId));
    }

    private static ResolvedModelVariant? VariantNamed(ModelVariantCatalog? catalog, string engineId)
    {
        ModelVariant? variant = catalog?.Variants.FirstOrDefault(candidate => candidate.MatchesHint(engineId));
        return variant is null ? null : new ResolvedModelVariant(catalog!.FamilyId, variant, ModelVariantSource.CallerHint, "engine id");
    }
}
