namespace HartsyInference.Engine.Recipes;

/// <summary>One video architecture family's construction recipe — the video counterpart of
/// <see cref="IArchitectureRecipe"/>. Keyed on the catalog family slug (e.g. "wan-video", "ltx-video").</summary>
public interface IVideoRecipe
{
    /// <summary>Short recipe name for diagnostics (typically the primary family id).</summary>
    string Name { get; }

    /// <summary>Whether this recipe handles the family identified by <paramref name="familyId"/>.</summary>
    bool Matches(string familyId);

    /// <summary>Builds the video pipeline for <paramref name="context"/>. Cached and reused across requests.</summary>
    IVideoRecipePipeline Construct(RecipeContext context);
}
