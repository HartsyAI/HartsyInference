using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>FLUX.3 Klein full vs step-distilled. The distilled build is a flow map whose <c>ref_time_in</c> embeds the
/// time each step jumps to, so its weights prove it and nothing else may select it.</summary>
public static class Flux3Variants
{
    /// <summary>The full build: guided sampling, ComfyUI template 25 steps at CFG 3.5 (default).</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "FLUX.3 Klein" };

    /// <summary>The step-distilled flow-map build: 4–16 steps at CFG 1 (BFL's <c>_sd_</c> files).</summary>
    public static ModelVariant Distilled { get; } = new ModelVariant
    {
        Id = "distilled",
        DisplayName = "FLUX.3 Klein Distilled",
        HintAliases = ["sd", "step-distilled"],
        StructuralMatch = probe => probe.AnyKeyContains("ref_time_in."),
        StructureRequired = true,
    };

    /// <summary>The catalog <see cref="Flux3Recipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("flux3", Base, [Distilled, Base]);
}
