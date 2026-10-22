using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Catalog;

/// <summary>The recipe catalog over the built-in registries. Every value comes from the same recipe call the generation
/// path makes for the same variant, so a host is never told a number the pipeline that runs would disagree with.</summary>
public sealed class RecipeCatalog : IRecipeCatalog
{
    /// <inheritdoc/>
    public ModelDescriptor Describe(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        return spec.Modality switch
        {
            Modality.Image => DescribeImage(spec),
            Modality.Video => DescribeVideo(spec),
            _ => throw UnsupportedModality(spec.Modality),
        };
    }

    /// <inheritdoc/>
    public ModelDescriptor DescribeFamily(Modality modality, string familyId, string? variantId = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(familyId);
        return modality switch
        {
            Modality.Image => DescribeImageFamily(familyId, variantId),
            Modality.Video => DescribeVideoFamily(familyId, variantId),
            _ => throw UnsupportedModality(modality),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyList<ModelDescriptor> All(Modality? modality = null)
    {
        List<ModelDescriptor> all = [];
        if (modality is null or Modality.Image)
        {
            foreach (string name in RecipeRegistry.DefaultNames)
            {
                all.Add(DescribeImageFamily(name, null));
            }
        }
        if (modality is null or Modality.Video)
        {
            foreach (string name in VideoRecipeRegistry.DefaultNames)
            {
                all.Add(DescribeVideoFamily(name, null));
            }
        }
        return all;
    }

    private static ModelDescriptor DescribeImage(ModelSpec spec)
    {
        string familyId = InferenceEngine.ResolveFamilyId(spec);
        IArchitectureRecipe recipe = RecipeRegistry.Resolve(familyId)
            ?? throw new RecipeNotRegisteredException(Modality.Image, familyId, RecipeRegistry.RegisteredNames);
        return ImageDescriptor(familyId, recipe, RecipeResolution.ResolveVariant(recipe.Variants, spec));
    }

    private static ModelDescriptor DescribeVideo(ModelSpec spec)
    {
        string familyId = RecipeResolution.VideoFamilyIdFor(spec);
        (IVideoRecipe? recipe, ResolvedModelVariant? variant) = RecipeResolution.ResolveVideo(spec);
        if (recipe is null)
        {
            throw new RecipeNotRegisteredException(Modality.Video, familyId, VideoRecipeRegistry.RegisteredNames);
        }
        return VideoDescriptor(familyId, recipe, variant, spec.LocalPath);
    }

    private static ModelDescriptor DescribeImageFamily(string familyId, string? variantId)
    {
        IArchitectureRecipe recipe = RecipeRegistry.Resolve(familyId)
            ?? throw new RecipeNotRegisteredException(Modality.Image, familyId, RecipeRegistry.RegisteredNames);
        return ImageDescriptor(familyId, recipe, FamilyVariant(familyId, recipe.Variants, variantId));
    }

    private static ModelDescriptor DescribeVideoFamily(string familyId, string? variantId)
    {
        IVideoRecipe recipe = VideoRecipeRegistry.Resolve(familyId)
            ?? throw new RecipeNotRegisteredException(Modality.Video, familyId, VideoRecipeRegistry.RegisteredNames);
        return VideoDescriptor(familyId, recipe, FamilyVariant(familyId, recipe.Variants, variantId), null);
    }

    private static ImageModelDescriptor ImageDescriptor(string familyId, IArchitectureRecipe recipe, ResolvedModelVariant? variant)
    {
        ImageDefaults defaults = recipe.DefaultsFor(variant);
        return new ImageModelDescriptor
        {
            Recipe = RecipeIdentity.For(familyId, defaults.Revision, variant),
            Modality = Modality.Image,
            Variants = VariantIds(recipe.Variants),
            ProfileBound = false,
            Defaults = defaults,
            Features = RecipeResolution.ImageFeaturesFor(recipe, variant),
            InputLimits = recipe.InputLimitsFor(variant),
            PromptWeighting = recipe.PromptWeighting,
        };
    }

    private static VideoModelDescriptor VideoDescriptor(string familyId, IVideoRecipe recipe, ResolvedModelVariant? variant,
        string? checkpointPath)
    {
        VideoDefaults defaults = recipe.DefaultsFor(variant);
        return new VideoModelDescriptor
        {
            Recipe = RecipeIdentity.For(familyId, defaults.Revision, variant),
            Modality = Modality.Video,
            Variants = VariantIds(recipe.Variants),
            ProfileBound = VideoProfileResolver.BindsProfiles(familyId),
            Defaults = defaults,
            Constraints = recipe.Constraints,
            Features = RecipeResolution.VideoFeaturesFor(recipe, variant, checkpointPath),
            Sampling = recipe.SamplingSupportFor(familyId, variant),
        };
    }

    /// <summary>The variant a family-level descriptor answers for: the named one, or the catalog default when none is named.</summary>
    private static ResolvedModelVariant? FamilyVariant(string familyId, ModelVariantCatalog? catalog, string? variantId)
    {
        if (catalog is null)
        {
            return variantId is null ? null : throw UnknownVariant(familyId, variantId, []);
        }
        if (variantId is null)
        {
            return new ResolvedModelVariant(catalog.FamilyId, catalog.Default, ModelVariantSource.Default, "family default");
        }
        ModelVariant variant = catalog.Find(variantId)
            ?? throw UnknownVariant(familyId, variantId, VariantIds(catalog));
        return new ResolvedModelVariant(catalog.FamilyId, variant, ModelVariantSource.CallerHint, $"variant '{variantId}'");
    }

    private static IReadOnlyList<string> VariantIds(ModelVariantCatalog? catalog) =>
        catalog is null ? [] : [.. catalog.Variants.Select(variant => variant.Id)];

    private static ArgumentException UnknownVariant(string familyId, string variantId, IReadOnlyList<string> valid) =>
        new($"Variant '{variantId}' is not declared by family '{familyId}'. Valid variant ids: "
            + $"{(valid.Count == 0 ? "none" : string.Join(", ", valid))}.", nameof(variantId));

    private static ArgumentException UnsupportedModality(Modality modality) =>
        new($"No recipe catalog serves modality '{modality}'; only image and video are described.", nameof(modality));
}
