using HartsyInference.Engine.Features;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Krea 3 recipe (Krea, single-stream MMDiT with reference-image editing). Architecture and open questions:
/// <c>docs/Research/KREA3_ARCHITECTURE.md</c>. No checkpoint has been released, so <see cref="Construct"/> refuses by
/// name before reading any file. The declarations below describe the planned wiring; nothing is served until
/// construction exists.</summary>
public sealed class Krea3Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "krea3";

    /// <summary>Declared features. Only <see cref="ImageFeatures.RefEdit"/> is declared now: it is the reference-edit
    /// route the report describes, and Fibo 2 splits it from strength img2img the same way. Img2Img, Inpaint and Lora
    /// stay undeclared until a masked or init-image path and LoRA merge exist, because
    /// <c>ImageFeatureDeclarationTests</c> keeps exact family lists and a declaration without a path is refused.
    /// TODO(krea3-weights): add Img2Img, Inpaint, Lora and the Krea 2-inherited Regional, SeamlessTiling and
    /// VariationSeed flags once their paths are wired against the reference tokens.</summary>
    public ImageFeatures Supports => ImageFeatures.RefEdit;

    /// <summary>The report allows up to 8 references in the product and trains with 7. TODO(krea3-weights): verify the
    /// product cap against a real checkpoint before relying on it.</summary>
    public ImageInputLimits InputLimits => new(8, ReferencesRequireInitImage: false, MaxReferences: 8);

    /// <summary>Planned to match Krea 2's weighting pair. TODO(krea3-weights): confirm the attention-weight path works
    /// with the reference tokens present.</summary>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting =>
        Diffusion.Prompting.PromptWeightingMode.CondScaleWithAttention;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "krea3", StringComparison.OrdinalIgnoreCase);

    /// <summary>Placeholder: inherits Krea 2 Base's sampling settings. TODO(krea3-weights): replace with the published
    /// steps, CFG and resolution. The report gives none of them.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 4.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public ModelVariantCatalog? Variants => Krea3Variants.Catalog;

    /// <inheritdoc/>
    public ImageDefaults DefaultsFor(ResolvedModelVariant? variant) => FamilyDefaults;

    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.None;

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">Always: no Krea 3 checkpoint has been released. The message names the
    /// missing piece so a request fails by name before any weights load.</exception>
    public IRecipePipeline Construct(RecipeContext context) =>
        throw new NotSupportedException(
            "Krea 3 has no released checkpoint yet, so this engine cannot construct it. Architecture notes and the open "
            + "questions are in docs/Research/KREA3_ARCHITECTURE.md. Build the pipeline once the weights are published.");
}
