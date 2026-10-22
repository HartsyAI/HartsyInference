namespace HartsyInference.Engine.Recipes;

/// <summary>The registry of music recipes, the audio counterpart of <see cref="VideoRecipeRegistry"/>: one place to
/// register every music family, validated when it loads.</summary>
public static class MusicRecipeRegistry
{
    private static readonly List<IMusicRecipe> _recipes = Validated(BuildDefaults());

    /// <summary>The families this build ships, captured before <see cref="Register"/> can add to them.</summary>
    public static IReadOnlyList<string> DefaultNames { get; } = [.. _recipes.Select(r => r.Name)];

    /// <summary>The recipe that handles <paramref name="familyId"/>, or null when none is registered.</summary>
    public static IMusicRecipe? Resolve(string familyId)
    {
        foreach (IMusicRecipe recipe in _recipes)
        {
            if (recipe.Matches(familyId))
                return recipe;
        }
        return null;
    }

    /// <summary>Registers an additional recipe (last-registered wins on ties).</summary>
    public static void Register(IMusicRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        IReadOnlyList<string> problems = RecipeRegistryValidator.Problems(recipe);
        if (problems.Count > 0)
        {
            throw new ArgumentException($"Recipe '{recipe.Name}' is invalid: {string.Join(" ", problems)}", nameof(recipe));
        }
        _recipes.Insert(0, recipe);
    }

    /// <summary>The music families a recipe currently covers.</summary>
    public static IReadOnlyCollection<string> RegisteredNames => _recipes.Select(r => r.Name).ToList();

    /// <summary>Refuses to serve a built-in registry whose declarations break their own contract.</summary>
    private static List<IMusicRecipe> Validated(List<IMusicRecipe> recipes)
    {
        RecipeRegistryValidator.ThrowIfInvalid(recipes);
        return recipes;
    }

    private static List<IMusicRecipe> BuildDefaults() => new List<IMusicRecipe>
    {
        new Music.MusicGenMusicRecipe("musicgen"),
        new Music.MusicGenMusicRecipe("audiogen"),
        new Music.AceStepMusicRecipe(),
        new Music.YueMusicRecipe(),
        new Music.Yue2MusicRecipe(),
        new Music.HeartMulaMusicRecipe(),
        new Music.MiniMaxMusic3MusicRecipe(),
        new Music.StableAudioMusicRecipe(),
        new Music.ControlFoleyMusicRecipe(),
    };
}
