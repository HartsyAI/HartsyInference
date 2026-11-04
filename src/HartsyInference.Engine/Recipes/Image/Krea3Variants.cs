using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Krea 3 Base vs Fast. Variant names come from the technical report; the checkpoint names and detection
/// tokens are TODO until weights are published, so <see cref="FilenameTokenSets"/> stays empty.</summary>
public static class Krea3Variants
{
    /// <summary>The default build. TODO(krea3-weights): confirm this is the open-weight build, not Ultra.</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "Krea 3" };

    /// <summary>The Fast build the report names. TODO(krea3-weights): confirm distillation and its step count.</summary>
    public static ModelVariant Fast { get; } = new ModelVariant
    {
        Id = "fast",
        DisplayName = "Krea 3 Fast",
        // TODO(krea3-weights): add HintAliases and FilenameTokenSets once a checkpoint name exists.
    };

    /// <summary>The catalog <see cref="Krea3Recipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("krea3", Base, [Fast, Base]);
}
