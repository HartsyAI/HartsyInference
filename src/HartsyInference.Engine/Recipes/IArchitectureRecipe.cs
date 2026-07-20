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

    /// <summary>Composition features this recipe actually applies; the default declares none, so a family that has not
    /// been wired rejects every composition object with a precise error instead of silently ignoring it.</summary>
    ImageFeatures Supports => ImageFeatures.None;

    /// <summary>This family's officially recommended sampling settings, used to fill the request tunables the caller
    /// left null. The generic 20-step / CFG-7.5 fallback keeps a recipe that has not declared its own numbers working;
    /// variant-dependent families refine it further via <see cref="IRecipePipeline.VariantDefaults"/>.</summary>
    ImageDefaults Defaults => ImageDefaults.Standard;
}
