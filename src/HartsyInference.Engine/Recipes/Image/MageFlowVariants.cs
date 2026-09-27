using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Mage-Flow base / Turbo and text-to-image / Edit. Every build shares the NR-MMDiT architecture and SwarmUI's
/// single <c>mage-flow</c> class, so a hint (<c>-m mage-flow:edit-turbo</c>), <c>hartsy.model_id</c> or, as the logged
/// last resort, the file name decides.</summary>
public static class MageFlowVariants
{
    /// <summary>Text-to-image base (default).</summary>
    public static ModelVariant Base { get; } = new ModelVariant { Id = "base", DisplayName = "Mage-Flow" };

    /// <summary>Step-distilled text-to-image build.</summary>
    public static ModelVariant Turbo { get; } = new ModelVariant
    {
        Id = "turbo",
        DisplayName = "Mage-Flow Turbo",
        FilenameTokenSets = VariantTokens.Distilled,
    };

    /// <summary>Edit build.</summary>
    public static ModelVariant Edit { get; } = new ModelVariant { Id = "edit", DisplayName = "Mage-Flow Edit", FilenameTokenSets = [["edit"]] };

    /// <summary>Step-distilled edit build.</summary>
    public static ModelVariant EditTurbo { get; } = new ModelVariant
    {
        Id = "edit-turbo",
        DisplayName = "Mage-Flow Edit Turbo",
        FilenameTokenSets = VariantTokens.DistilledWith("edit"),
    };

    /// <summary>The catalog <see cref="MageFlowRecipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("mage-flow", Base, [EditTurbo, Turbo, Edit, Base]);

    /// <summary>Whether <paramref name="variant"/> is a step-distilled build.</summary>
    public static bool IsTurbo(ResolvedModelVariant? variant) => variant is not null && (variant.Is(Turbo) || variant.Is(EditTurbo));
}
