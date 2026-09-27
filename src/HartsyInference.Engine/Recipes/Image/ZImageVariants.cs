using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Z-Image Base vs Turbo: architecturally indistinguishable, and the official BF16 Base file carries no
/// metadata. The official Base release ships under the bare family name (<c>Z-Image</c>, <c>z_image_bf16</c>), so that
/// name without a <c>turbo</c> token is Base evidence too; a name carrying both tokens is ambiguous and falls to the
/// Base default, whose F32-attention policy is numerically safe on either weight set.</summary>
public static class ZImageVariants
{
    /// <summary>The Base build (default): F32 attention, shift 6.</summary>
    public static ModelVariant Base { get; } = new ModelVariant
    {
        Id = "base",
        DisplayName = "Z-Image Base",
        FilenameTokenSets = [["base", "!turbo"], ["zimage", "!turbo"], ["z", "image", "!turbo"]],
    };

    /// <summary>The distilled Turbo build.</summary>
    public static ModelVariant Turbo { get; } = new ModelVariant
    {
        Id = "turbo",
        DisplayName = "Z-Image Turbo",
        FilenameTokenSets = [["turbo", "!base"]],
    };

    /// <summary>The catalog <see cref="ZImageRecipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("zimage", Base, [Turbo, Base]);
}
