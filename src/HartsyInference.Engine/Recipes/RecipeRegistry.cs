namespace HartsyInference.Engine.Recipes;

/// <summary>The registry of architecture recipes. Services resolve a checkpoint's family id (catalog slug) to the
/// matching recipe here; families whose recipe has been lifted are drivable, the rest resolve to null (and the caller
/// surfaces a clear not-yet-wired error). One place to register every family as it lands.</summary>
public static class RecipeRegistry
{
    private static readonly List<IArchitectureRecipe> _recipes = BuildDefaults();

    /// <summary>The recipe that handles <paramref name="familyId"/>, or null when none is registered yet.</summary>
    public static IArchitectureRecipe? Resolve(string familyId)
    {
        foreach (IArchitectureRecipe recipe in _recipes)
        {
            if (recipe.Matches(familyId))
                return recipe;
        }
        return null;
    }

    /// <summary>Registers an additional recipe (last-registered wins on ties). Used by tests and out-of-tree families.</summary>
    public static void Register(IArchitectureRecipe recipe) => _recipes.Insert(0, recipe);

    /// <summary>The architectures a lifted recipe currently covers.</summary>
    public static IReadOnlyCollection<string> RegisteredNames => _recipes.Select(r => r.Name).ToList();

    private static List<IArchitectureRecipe> BuildDefaults() => new List<IArchitectureRecipe>
    {
        new Image.SdxlRecipe(),
    };
}
