using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Qwen-Image base vs Edit vs Edit-Plus. Base, Edit and Edit-2509 are key-for-key identical, so the SwarmUI
/// model class (<c>qwen-image-edit</c> / <c>qwen-image-edit-plus</c>), a <c>family:variant</c> selector or
/// <c>modelspec.architecture</c> decides; only Edit-2511 proves itself, by ComfyUI's bare
/// <c>__index_timestep_zero__</c> marker tensor.</summary>
public static class QwenImageVariants
{
    /// <summary>SwarmUI model-class id of Qwen-Image-Edit (v1).</summary>
    public const string EditClassId = "qwen-image-edit";

    /// <summary>SwarmUI model-class id of Qwen-Image-Edit-2509/2511.</summary>
    public const string EditPlusClassId = "qwen-image-edit-plus";

    /// <summary>ComfyUI's 2511 marker (the <c>index_timestep_zero</c> reference method).</summary>
    public const string TimestepZeroMarker = "__index_timestep_zero__";

    /// <summary>Text-to-image base (default); offers classic img2img only.</summary>
    public static ModelVariant Base { get; } = new ModelVariant
    {
        Id = "base",
        DisplayName = "Qwen-Image",
        MetadataClassIds = ["qwen-image"],
    };

    /// <summary>Qwen-Image-Edit (v1): one unlabelled reference.</summary>
    public static ModelVariant Edit { get; } = new ModelVariant
    {
        Id = "edit",
        DisplayName = "Qwen-Image-Edit",
        HintAliases = [EditClassId],
        MetadataClassIds = [EditClassId],
        FilenameTokenSets = [["edit"]],
    };

    /// <summary>Qwen-Image-Edit-2509/2511: up to three <c>Picture N:</c> references.</summary>
    public static ModelVariant EditPlus { get; } = new ModelVariant
    {
        Id = "edit-plus",
        DisplayName = "Qwen-Image-Edit-Plus",
        HintAliases = [EditPlusClassId, "2509", "2511"],
        MetadataClassIds = [EditPlusClassId],
        StructuralMatch = probe => probe.Keys.Any(key => key.EndsWith(TimestepZeroMarker, StringComparison.Ordinal)),
        FilenameTokenSets = [["edit", "2509"], ["edit", "2511"], ["edit", "plus"]],
    };

    /// <summary>The catalog <see cref="QwenImageRecipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("qwen-image", Base, [EditPlus, Edit, Base]);

    /// <summary>The reference template <paramref name="variant"/> was trained with; null for the text-to-image base.</summary>
    public static QwenImageEditTemplate? TemplateFor(ResolvedModelVariant variant)
    {
        ArgumentNullException.ThrowIfNull(variant);
        return variant.Is(EditPlus) ? QwenImageEditTemplate.EditPlus : variant.Is(Edit) ? QwenImageEditTemplate.Edit : null;
    }
}
