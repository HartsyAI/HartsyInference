using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The two LongCat-Video-Avatar builds the recipe knows. The 1.5 and 1.6 checkpoints have the same tensors and config, so nothing in the
/// weights tells them apart by their files, so the caller names the build: <c>longcat-avatar:1.6</c> routes to the 1.6 row. An unnamed request is the
/// 1.5 build (the default).</summary>
public static class LongCatAvatarVariants
{
    /// <summary>True for a requested 1.6 build name (<c>1.6</c> or <c>avatar-1.6</c>).</summary>
    public static bool IsSixPointOne(string variant) =>
        variant.Equals("1.6", StringComparison.OrdinalIgnoreCase) || variant.Equals("avatar-1.6", StringComparison.OrdinalIgnoreCase);

    /// <summary>The catalog row id of the 1.6 INT8 release. A request names it as <c>longcat-avatar-1.6</c>, or <c>longcat-avatar:1.6</c>, which the CLI
    /// rewrites to it.</summary>
    public const string Avatar16FamilyId = "longcat-avatar-1.6";

    /// <summary>The 1.6 card's example mix: the first four steps run DMD only (alpha 1.0), the last four OPSD only.</summary>
    public const int Avatar16MixSteps = 4;

    /// <summary>The DMD weight during the mixed steps (see <see cref="Avatar16MixSteps"/>).</summary>
    public const float Avatar16MixAlpha = 1f;

    /// <summary>The 1.6 build: the 8-step distilled path with its DMD (loraA) and OPSD (loraB) LoRAs mixed per step, as the 1.6 model card runs it
    /// (<c>--use_distill --mix_steps=4 --mix_alpha=1.0</c>). Its weights are the INT8 row's.</summary>
    public static ModelVariant Avatar16 { get; } = new ModelVariant
    {
        Id = "1.6", DisplayName = "LongCat-Video-Avatar 1.6", HintAliases = ["avatar-1.6", Avatar16FamilyId],
    };

    /// <summary>The 1.5 build: the distilled 8-step path, guidance off, scheduler shift 7. The default build; the recipe runs it on this path.</summary>
    public static ModelVariant Avatar15 { get; } = new ModelVariant
    {
        Id = "1.5", DisplayName = "LongCat-Video-Avatar 1.5", HintAliases = ["avatar-1.5"],
    };

    /// <summary>The catalog <see cref="LongCatAvatarRecipe"/> declares. The default is 1.5: the catalog row carries the 1.5 weights, and a row's
    /// settings must match its weights. The 1.6 build has its own INT8 row (<see cref="Avatar16FamilyId"/>), which the recipe serves.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("longcat-avatar", Avatar15, [Avatar15, Avatar16]);
}
