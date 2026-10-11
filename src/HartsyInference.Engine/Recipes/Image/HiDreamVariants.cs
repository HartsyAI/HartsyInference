using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>HiDream-I1 Full / Dev / Fast. The three share one architecture and SwarmUI's single <c>hidream-i1</c> class,
/// so a hint (<c>-m hidream:full</c>), <c>hartsy.model_id</c> or, as the logged last resort, the file name decides.</summary>
public static class HiDreamVariants
{
    /// <summary>Guidance-distilled Dev build (default: the catalog pins <c>hidream_i1_dev_fp8</c>).</summary>
    public static ModelVariant Dev { get; } = new ModelVariant { Id = "dev", DisplayName = "HiDream-I1 Dev", FilenameTokenSets = [["dev"]] };

    /// <summary>Undistilled Full build, sampled with real CFG.</summary>
    public static ModelVariant Full { get; } = new ModelVariant { Id = "full", DisplayName = "HiDream-I1 Full", FilenameTokenSets = [["full"]] };

    /// <summary>Guidance-distilled Fast build.</summary>
    public static ModelVariant Fast { get; } = new ModelVariant { Id = "fast", DisplayName = "HiDream-I1 Fast", FilenameTokenSets = [["fast"]] };

    /// <summary>The catalog <see cref="HiDreamRecipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("hidream", Dev, [Full, Fast, Dev]);
}
