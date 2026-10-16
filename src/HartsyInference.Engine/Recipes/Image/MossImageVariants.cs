using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>MOSS-Image Base vs SFT. The SFT build is MOSS-Image-Base fine-tuned on curated data: the same architecture,
/// key set and sampling defaults, so nothing in the weights tells them apart and the variant only names the build. It
/// is selected by a hint or by an <c>sft</c> token in the file or folder name (<c>MOSS-Image-SFT/</c>).</summary>
public static class MossImageVariants
{
    /// <summary>The base model (default): no post-training, 50 steps at CFG 3 with layer 22 skipped on the unconditional branch.</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "MOSS-Image Base" };

    /// <summary>The supervised fine-tune of the base model, run with the same defaults.</summary>
    public static ModelVariant Sft { get; } = new ModelVariant
    {
        Id = "sft",
        DisplayName = "MOSS-Image SFT",
        FilenameTokenSets = [["sft"]],
    };

    /// <summary>The catalog <see cref="MossImageRecipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("moss-image", Base, [Sft, Base]);
}
