using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine;

/// <summary>An image pipeline together with the lease that keeps it from being evicted while the caller uses it.</summary>
internal readonly record struct LeasedRecipe(IRecipePipeline Pipeline, IDisposable Lease);
