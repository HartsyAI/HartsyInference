using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Anima base / Turbo. Both share the Cosmos-Predict2-2B architecture and SwarmUI's single <c>anima</c> class,
/// so a hint (<c>-m anima:turbo</c>), <c>hartsy.model_id</c> or, as the logged last resort, the file name decides.</summary>
public static class AnimaVariants
{
    /// <summary>Undistilled base and aesthetic builds (default).</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "Anima" };

    /// <summary>Step-distilled build (<c>anima-turbo-v1.x</c>).</summary>
    public static ModelVariant Turbo { get; } = new ModelVariant
    {
        Id = "turbo",
        DisplayName = "Anima Turbo",
        FilenameTokenSets = VariantTokens.Distilled,
    };

    /// <summary>The catalog <see cref="AnimaRecipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("anima", Base, [Turbo, Base]);
}
