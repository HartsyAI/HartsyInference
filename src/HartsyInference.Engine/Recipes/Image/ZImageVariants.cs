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

    /// <summary>NitroFusion2 (ChenDY/NitroFusion2, Apache 2.0): a 1–2 step adversarially distilled fine-tune of Z-Image
    /// Turbo's transformer. Same geometry and keys as Turbo; the file name is the only signal. It is not Turbo, so it
    /// takes the Base policy (F32 attention, shift 6) until its F16 behaviour is validated.</summary>
    public static ModelVariant NitroFusion2 { get; } = new ModelVariant
    {
        Id = "nitrofusion2",
        DisplayName = "Z-Image NitroFusion2",
        HintAliases = ["nf2"],
        FilenameTokenSets = [["nitrofusion2"], ["nitro", "fusion", "2"], ["nf2"]],
    };

    /// <summary>The catalog <see cref="ZImageRecipe"/> declares. NitroFusion2 precedes Turbo so a name carrying both
    /// tokens resolves to the more specific build.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("zimage", Base, [NitroFusion2, Turbo, Base]);
}
