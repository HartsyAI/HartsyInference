using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Boogu-Image base / Turbo and text-to-image / Edit. Every build shares one DiT and SwarmUI's single
/// <c>boogu</c> class, so a hint (<c>-m boogu:turbo</c>), <c>hartsy.model_id</c> or, as the logged last resort, the
/// file name decides.</summary>
public static class BooguImageVariants
{
    /// <summary>Text-to-image base (default).</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "Boogu-Image" };

    /// <summary>Step-distilled text-to-image build.</summary>
    public static ModelVariant Turbo { get; } = new ModelVariant
    {
        Id = "turbo",
        DisplayName = "Boogu-Image Turbo",
        FilenameTokenSets = VariantTokens.Distilled,
    };

    /// <summary>Edit build (the catalog's checkpoint), which also runs plain text-to-image.</summary>
    public static ModelVariant Edit { get; } = new ModelVariant { Id = "edit", DisplayName = "Boogu-Image Edit", FilenameTokenSets = [["edit"]] };

    /// <summary>Step-distilled edit build.</summary>
    public static ModelVariant EditTurbo { get; } = new ModelVariant
    {
        Id = "edit-turbo",
        DisplayName = "Boogu-Image Edit Turbo",
        FilenameTokenSets = VariantTokens.DistilledWith("edit"),
    };

    /// <summary>The catalog <see cref="BooguImageRecipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("boogu", Base, [EditTurbo, Turbo, Edit, Base]);

    /// <summary>Whether <paramref name="variant"/> is a step-distilled build.</summary>
    public static bool IsTurbo(ResolvedModelVariant? variant) => variant is not null && (variant.Is(Turbo) || variant.Is(EditTurbo));
}
