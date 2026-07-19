using HartsyInference.Diffusion.Pipelines;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Reference recipe: SDXL (dual-CLIP). The architecture every other family is measured against, and the
/// worked example the lifted per-family recipes follow. Constructs via <see cref="PipelineFactory.LoadSdxl"/> and
/// drives generation through <see cref="SdxlRecipePipeline"/>.</summary>
public sealed class SdxlRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "sdxl";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "sdxl", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        SdxlPipeline pipeline = PipelineFactory.LoadSdxl(context.CheckpointPath, context.Backend);
        return new SdxlRecipePipeline(pipeline);
    }
}
