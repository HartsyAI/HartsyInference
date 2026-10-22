namespace HartsyInference.Engine.Recipes;

/// <summary>One music family's recipe: the defaults its loader and pipeline run at, and the limits a request must
/// satisfy. The music counterpart of <see cref="IVideoRecipe"/>; the checkpoint loader stays in
/// <c>Audio/Music/MusicCatalog</c>.</summary>
public interface IMusicRecipe
{
    /// <summary>Short recipe name for diagnostics (the catalog family id).</summary>
    string Name { get; }

    /// <summary>Whether this recipe handles the family identified by <paramref name="familyId"/>.</summary>
    bool Matches(string familyId);

    /// <summary>The family-level defaults; for a family with variants, the values none of them override.</summary>
    MusicDefaults Defaults { get; }

    /// <summary>What this family can generate.</summary>
    MusicConstraints Constraints { get; }

    /// <summary>The variant ids whose defaults differ from <see cref="Defaults"/>; empty when the family has one set.</summary>
    IReadOnlyList<string> Variants => [];

    /// <summary>The defaults <paramref name="variant"/> runs at; the default is the family-level <see cref="Defaults"/>.</summary>
    MusicDefaults DefaultsFor(string variant) => Defaults;
}
