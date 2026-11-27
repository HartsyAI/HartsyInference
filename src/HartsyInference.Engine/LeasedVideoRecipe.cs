using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine;

/// <summary>A video pipeline together with the lease that keeps it from being evicted while the caller uses it.</summary>
internal readonly record struct LeasedVideoRecipe(IVideoRecipePipeline Pipeline, IDisposable Lease);
