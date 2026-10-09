using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Qwen-Image 2.1 base vs Turbo. Same architecture and side models; the Turbo build is a step-distilled
/// transformer that runs a fixed 8-step schedule, so a <c>turbo</c> filename token (or a hint) selects it.</summary>
public static class QwenImage21Variants
{
    /// <summary>The 2.1 base build (default): the template's 25-step flow-match schedule at CFG 1.</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "Qwen-Image 2.1" };

    /// <summary>The step-distilled Turbo build: fixed 8-step sigma schedule, guidance-free.</summary>
    public static ModelVariant Turbo { get; } = new ModelVariant
    {
        Id = "turbo",
        DisplayName = "Qwen-Image 2.1 Turbo",
        FilenameTokenSets = VariantTokens.Distilled,
    };

    /// <summary>The sigma schedule Qwen-Image-2.1-Turbo ships in its pipeline config (<c>sample_sigmas</c>). The
    /// checkpoint is distilled to these eight values, so they are used as given; other schedules are unevaluated.</summary>
    public static IReadOnlyList<float> TurboSigmas { get; } =
        [1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f];

    /// <summary>The catalog <see cref="QwenImage21Recipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("qwen-image-2.1", Base, [Turbo, Base]);
}
