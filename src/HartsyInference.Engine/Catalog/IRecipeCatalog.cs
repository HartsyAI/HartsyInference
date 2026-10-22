using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine.Catalog;

/// <summary>What the registered recipes declare about each model family, answered without loading weights or
/// constructing a pipeline. Hosts query this instead of re-declaring defaults.</summary>
public interface IRecipeCatalog
{
    /// <summary>The descriptor of the family and variant <paramref name="spec"/> resolves to, through the same routing a generation uses.</summary>
    /// <exception cref="RecipeNotRegisteredException">The model's family has no recipe for its modality.</exception>
    ModelDescriptor Describe(ModelSpec spec);

    /// <summary>The descriptor of a family, for one of its variants or for its default variant when <paramref name="variantId"/> is null.</summary>
    /// <exception cref="RecipeNotRegisteredException">No recipe is registered for the family.</exception>
    /// <exception cref="ArgumentException"><paramref name="variantId"/> is not a variant the family declares.</exception>
    ModelDescriptor DescribeFamily(Modality modality, string familyId, string? variantId = null);

    /// <summary>One family-level descriptor per built-in recipe of <paramref name="modality"/>, or of every image and video recipe when it is null.</summary>
    IReadOnlyList<ModelDescriptor> All(Modality? modality = null);
}
