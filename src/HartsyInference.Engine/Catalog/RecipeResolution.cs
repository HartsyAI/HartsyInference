using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Catalog;

/// <summary>Resolves a model to its family, routed video family, variant and recipe-declared features. Generation and
/// the catalog share these lookups, so a host's answer and the pipeline that runs cannot disagree.</summary>
internal static class RecipeResolution
{
    /// <summary>The variant <paramref name="spec"/> resolves to for its modality, or null when its family declares none.</summary>
    internal static ResolvedModelVariant? ResolveVariant(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string familyId = InferenceEngine.ResolveFamilyId(spec);
        ModelVariantCatalog? catalog = spec.Modality == Modality.Video
            ? VideoRecipeRegistry.Resolve(familyId)?.Variants
            : RecipeRegistry.Resolve(familyId)?.Variants;
        return ResolveVariant(catalog, spec);
    }

    /// <summary>Resolves <paramref name="spec"/> against <paramref name="catalog"/>; null for a family without variants.</summary>
    internal static ResolvedModelVariant? ResolveVariant(ModelVariantCatalog? catalog, ModelSpec spec) =>
        catalog is null ? null : ModelVariantResolver.Resolve(catalog, new ModelVariantEvidence(spec.LocalPath, HintsFor(spec)));

    /// <summary>The recipe that serves <paramref name="spec"/> after variant routing, and the variant it serves.</summary>
    internal static (IVideoRecipe? Recipe, ResolvedModelVariant? Variant) ResolveVideo(ModelSpec spec)
    {
        IVideoRecipe? recipe = VideoRecipeRegistry.Resolve(VideoFamilyIdFor(spec));
        return (recipe, recipe is null ? null : ResolveVariant(recipe.Variants, spec));
    }

    /// <summary>The video family id <paramref name="spec"/> is served by: its own family, or the family its resolved
    /// variant routes to (<see cref="ModelVariant.RoutesToFamily"/>).</summary>
    internal static string VideoFamilyIdFor(ModelSpec spec)
    {
        string familyId = InferenceEngine.ResolveFamilyId(spec);
        ResolvedModelVariant? variant = ResolveVariant(VideoRecipeRegistry.Resolve(familyId)?.Variants, spec);
        return variant?.Variant.RoutesToFamily ?? familyId;
    }

    /// <summary>Features of <paramref name="recipe"/> for <paramref name="variant"/>, plus the prompt-weighting bit its mode implies.</summary>
    internal static ImageFeatures ImageFeaturesFor(IArchitectureRecipe recipe, ResolvedModelVariant? variant)
    {
        ImageFeatures declared = recipe.SupportsFor(variant);
        return AppliesWeighting(recipe.PromptWeighting) ? declared | ImageFeatures.PromptWeighting : declared;
    }

    /// <summary>Features of <paramref name="recipe"/> for <paramref name="variant"/>, plus the prompt-weighting bit its mode
    /// implies. LTX reads its checkpoint's shape, so <paramref name="checkpointPath"/> null means the family-level answer.</summary>
    internal static VideoFeatures VideoFeaturesFor(IVideoRecipe recipe, ResolvedModelVariant? variant, string? checkpointPath)
    {
        VideoFeatures declared = recipe is LtxVideoRecipe ltx ? ltx.SupportsFor(checkpointPath) : recipe.SupportsFor(variant);
        return AppliesWeighting(recipe.PromptWeighting) ? declared | VideoFeatures.PromptWeighting : declared;
    }

    /// <summary>Whether a declared weighting mode means the emphasis grammar must survive prompt flattening.</summary>
    internal static bool AppliesWeighting(PromptWeightingMode mode) => mode != PromptWeightingMode.None;

    /// <summary>The caller's variant hints, strongest first: the explicit variant (SwarmUI's model class), a
    /// <c>family:variant</c> selector suffix, then the requested id itself so a legacy variant-named id still selects.</summary>
    internal static IReadOnlyList<string?> HintsFor(ModelSpec spec)
    {
        ModelSelector selector = ModelSelector.Parse(spec.Requested);
        return [spec.Variant, selector.Variant, spec.Catalog?.Id, selector.Id];
    }
}
