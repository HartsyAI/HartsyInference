using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Wan-Animate-2 base vs distillation. The two builds are key-for-key identical and both declare only
/// <c>model_type: "animate2"</c>, so a hint or <c>hartsy.model_id</c> decides; upstream's own file name
/// (<c>wan_animate_2_bf16_distillation</c>) is the logged last resort.</summary>
public static class WanAnimate2Variants
{
    /// <summary>The base build (default): unmasked attention, the caller's steps and guidance.</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "Wan-Animate-2" };

    /// <summary>The distillation build, trained with a score bias of <see cref="Diffusion.Models.Denoisers.WanAnimate2Transformer.DistillLogScale"/>.</summary>
    public static ModelVariant Distillation { get; } = new ModelVariant
    {
        Id = "distillation",
        DisplayName = "Wan-Animate-2 Distillation",
        HintAliases = ["distill", "distilled"],
        FilenameTokenSets = [["distillation"], ["distill"], ["distilled"]],
    };

    /// <summary>The catalog <see cref="WanAnimate2Recipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("wan-animate-2", Base, [Distillation, Base]);
}
