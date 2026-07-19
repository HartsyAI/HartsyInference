namespace HartsyInference.Engine.Recipes;

/// <summary>One architecture family's construction recipe: it knows which family id it handles (the catalog slug, e.g.
/// "sdxl", "zimage", "chroma", "qwen-image", "krea2") and how to build that family's pipeline from a checkpoint plus
/// any side models. This is the single seam the per-family loaders (lifted from the SwarmUI backend's
/// <c>Generation/*Loader.cs</c>) collapse onto — add a recipe, register it, and the whole engine (CLI, HTTP,
/// extension) can drive that family. The family id keys the registry because most families are not distinguishable by
/// the coarse tensor-signature <c>ModelArchitecture</c> enum; the catalog / caller supplies it.</summary>
public interface IArchitectureRecipe
{
    /// <summary>Short recipe name for diagnostics (typically the primary family id, e.g. "sdxl").</summary>
    string Name { get; }

    /// <summary>Whether this recipe handles the family identified by <paramref name="familyId"/> (a catalog slug).</summary>
    bool Matches(string familyId);

    /// <summary>Builds the pipeline for <paramref name="context"/>. The result is cached and reused across requests.</summary>
    IRecipePipeline Construct(RecipeContext context);
}
