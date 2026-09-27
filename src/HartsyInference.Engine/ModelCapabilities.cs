using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine;

/// <summary>Host-facing capability queries for a model: which variant it is, and what that variant can do. Answered
/// through the same family-id, registry and variant resolution construction uses, so a host (the SwarmUI backend,
/// the CLI) can never be told something the pipeline that runs would disagree with.</summary>
public static class ModelCapabilities
{
    /// <summary>The variant <paramref name="spec"/> resolves to for its modality, or null when its family declares none.</summary>
    public static ResolvedModelVariant? ResolveVariant(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string familyId = InferenceEngine.ResolveFamilyId(spec);
        ModelVariantCatalog? catalog = spec.Modality == Modality.Video
            ? VideoRecipeRegistry.Resolve(familyId)?.Variants
            : RecipeRegistry.Resolve(familyId)?.Variants;
        return ResolveVariant(catalog, spec);
    }

    /// <summary>The composition features the image recipe for <paramref name="spec"/> applies to its resolved variant; none for an unregistered family.</summary>
    public static ImageFeatures ImageFeaturesFor(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        IArchitectureRecipe? recipe = RecipeRegistry.Resolve(InferenceEngine.ResolveFamilyId(spec));
        return recipe is null ? ImageFeatures.None : ImageFeaturesFor(recipe, spec);
    }

    /// <summary>The input-image limits of the image recipe for <paramref name="spec"/>'s resolved variant; text-only for an unregistered family.</summary>
    public static ImageInputLimits ImageInputLimitsFor(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        IArchitectureRecipe? recipe = RecipeRegistry.Resolve(InferenceEngine.ResolveFamilyId(spec));
        return recipe is null ? ImageInputLimits.TextOnly : recipe.InputLimitsFor(ResolveVariant(recipe.Variants, spec));
    }

    /// <summary>The conditioning the video recipe for <paramref name="spec"/> applies to its resolved variant; none for an unregistered family.</summary>
    public static VideoFeatures VideoFeaturesFor(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        (IVideoRecipe? recipe, ResolvedModelVariant? variant) = ResolveVideo(spec);
        if (recipe is null)
        {
            return VideoFeatures.None;
        }
        VideoFeatures declared = recipe is Recipes.Video.LtxVideoRecipe ltx ? ltx.SupportsFor(spec.LocalPath) : recipe.SupportsFor(variant);
        return AppliesWeighting(recipe.PromptWeighting) ? declared | VideoFeatures.PromptWeighting : declared;
    }

    /// <summary>The officially recommended video defaults for <paramref name="spec"/>'s resolved variant.</summary>
    public static VideoDefaults VideoDefaultsFor(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        (IVideoRecipe? recipe, ResolvedModelVariant? variant) = ResolveVideo(spec);
        return recipe?.DefaultsFor(variant) ?? VideoDefaults.Standard;
    }

    /// <summary>The sampler/schedule selection the video recipe for <paramref name="spec"/> accepts for its resolved variant.</summary>
    public static SamplingCapabilities.SamplingSupport VideoSamplingSupportFor(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string familyId = VideoFamilyIdFor(spec);
        (IVideoRecipe? recipe, ResolvedModelVariant? variant) = ResolveVideo(spec);
        return recipe?.SamplingSupportFor(familyId, variant) ?? SamplingCapabilities.Unknown;
    }

    /// <summary>The video family id <paramref name="spec"/> is served by: its own family, or the family its resolved
    /// variant routes to (<see cref="ModelVariant.RoutesToFamily"/>).</summary>
    public static string VideoFamilyIdFor(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string familyId = InferenceEngine.ResolveFamilyId(spec);
        ResolvedModelVariant? variant = ResolveVariant(VideoRecipeRegistry.Resolve(familyId)?.Variants, spec);
        return variant?.Variant.RoutesToFamily ?? familyId;
    }

    /// <summary>Features of <paramref name="recipe"/> for <paramref name="spec"/>'s resolved variant, plus the prompt-weighting bit its mode implies.</summary>
    internal static ImageFeatures ImageFeaturesFor(IArchitectureRecipe recipe, ModelSpec spec)
    {
        ImageFeatures declared = recipe.SupportsFor(ResolveVariant(recipe.Variants, spec));
        return AppliesWeighting(recipe.PromptWeighting) ? declared | ImageFeatures.PromptWeighting : declared;
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

    /// <summary>Whether a declared weighting mode means the emphasis grammar must survive prompt flattening.</summary>
    internal static bool AppliesWeighting(Diffusion.Prompting.PromptWeightingMode mode) =>
        mode != Diffusion.Prompting.PromptWeightingMode.None;

    /// <summary>The caller's variant hints, strongest first: the explicit variant (SwarmUI's model class), a
    /// <c>family:variant</c> selector suffix, then the requested id itself so a legacy variant-named id still selects.</summary>
    internal static IReadOnlyList<string?> HintsFor(ModelSpec spec)
    {
        ModelSelector selector = ModelSelector.Parse(spec.Requested);
        return [spec.Variant, selector.Variant, spec.Catalog?.Id, selector.Id];
    }
}
