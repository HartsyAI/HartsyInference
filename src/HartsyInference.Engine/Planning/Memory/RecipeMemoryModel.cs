namespace HartsyInference.Engine.Planning.Memory;

/// <summary>What a recipe knows about its own memory that a checkpoint header cannot say: the working memory each phase
/// needs at a given geometry, and which side models it loads beside the checkpoint.</summary>
/// <remarks>Returned from the recipe's <c>DescribeMemory</c> hook. The activation functions must call the SAME
/// formulas the pipeline hands to its planner — a second copy would drift, and an estimate that disagrees with the
/// pipeline routes requests to cards that then run out of memory.</remarks>
public sealed record RecipeMemoryModel
{
    /// <summary>The denoise phase's working memory beside the weights, per geometry.</summary>
    public required Func<MemoryEstimateRequest, long> DenoiserActivationBytes { get; init; }

    /// <summary>The decode phase's working memory beside the VAE weights, per geometry; null uses the generic allowance.</summary>
    public Func<MemoryEstimateRequest, long>? VaeActivationBytes { get; init; }

    /// <summary>The text encoder the recipe loads when the checkpoint does not bundle one.</summary>
    public ModelAsset? TextEncoder { get; init; }

    /// <summary>The VAE the recipe loads when the checkpoint does not bundle one.</summary>
    public ModelAsset? Vae { get; init; }
}
