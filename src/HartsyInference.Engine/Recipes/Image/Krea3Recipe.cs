using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Scaffold for Krea 3 (Krea, single-stream MMDiT with reference-image editing). No checkpoint is released, so it declares no feature
/// and <see cref="Construct"/> refuses by name. Architecture and open questions: <c>docs/Research/KREA3_ARCHITECTURE.md</c>.</summary>
public sealed class Krea3Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "krea3";

    /// <summary>Only what the architecture fixes, since no prompting guide covers Krea 3: the report names a vision-language text encoder.</summary>
    public PromptProfile? Prompt { get; } = new() { Kind = "image", Encoders = ["vlm"] };

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "krea3", StringComparison.OrdinalIgnoreCase);

    // TODO(krea3-weights): these are Krea 2 Base's values, used because the report gives no steps, CFG or resolution.
    /// <summary>Placeholder sampling settings.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 4.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public ModelVariantCatalog? Variants => Krea3Variants.Catalog;

    /// <inheritdoc/>
    public ImageDefaults DefaultsFor(ResolvedModelVariant? variant) => FamilyDefaults;

    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.None;

    // TODO(krea3-weights): build the pipeline, then declare the features, input limits and prompt weighting it applies.
    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">Always: no Krea 3 checkpoint has been released.</exception>
    public IRecipePipeline Construct(RecipeContext context) =>
        throw new NotSupportedException(
            "Krea 3 has no released checkpoint yet, so this engine cannot construct it. Architecture notes and the open "
            + "questions are in docs/Research/KREA3_ARCHITECTURE.md.");
}
