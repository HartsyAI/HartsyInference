using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Lens standard vs Lens-Turbo: the same transformer, differing only in sampling defaults.</summary>
public static class LensVariants
{
    /// <summary>Lens / Lens-Base (default).</summary>
    public static ModelVariant Standard { get; } = new ModelVariant { Id = "standard", DisplayName = "Lens", HintAliases = ["base"] };

    /// <summary>Lens-Turbo: 4 distilled steps, no CFG.</summary>
    public static ModelVariant Turbo { get; } = new ModelVariant { Id = "turbo", DisplayName = "Lens-Turbo", FilenameTokenSets = [["turbo"]] };

    /// <summary>The catalog <see cref="LensRecipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("lens", Standard, [Turbo, Standard]);
}
