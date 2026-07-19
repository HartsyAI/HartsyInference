using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>Everything a recipe needs to construct its pipeline: the checkpoint path, the compute backend, and any
/// per-request component overrides (VAE / text encoders). Side models beyond the checkpoint are fetched by the recipe
/// via <see cref="ModelDownloader"/> against the <see cref="SideModels"/> registry.</summary>
public sealed record RecipeContext
{
    /// <summary>Local path to the primary checkpoint (the transformer/UNet file or its directory).</summary>
    public required string CheckpointPath { get; init; }

    /// <summary>Compute backend the constructed pipeline runs on.</summary>
    public required IBackend Backend { get; init; }

    /// <summary>Optional swappable-component overrides; null keeps the recipe's defaults.</summary>
    public ComponentOverrides? Components { get; init; }
}
