using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Krea 2 Base vs Turbo/TDM: architecturally identical, so a hint, <c>hartsy.model_id</c> or (logged) the file name decides.</summary>
public static class Krea2Variants
{
    /// <summary>The guided base build (default).</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "Krea 2" };

    /// <summary>The step-distilled Turbo/TDM build: pinned shift, guidance-free.</summary>
    public static ModelVariant Turbo { get; } = new ModelVariant
    {
        Id = "turbo",
        DisplayName = "Krea 2 Turbo",
        HintAliases = ["tdm"],
        FilenameTokenSets = VariantTokens.Distilled,
    };

    /// <summary>The catalog <see cref="Krea2Recipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("krea2", Base, [Turbo, Base]);
}
