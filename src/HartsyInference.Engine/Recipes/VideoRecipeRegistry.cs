namespace HartsyInference.Engine.Recipes;

/// <summary>The registry of video architecture recipes — the video counterpart of <see cref="RecipeRegistry"/>.
/// One place to register every video family as it lands.</summary>
public static class VideoRecipeRegistry
{
    private static readonly List<IVideoRecipe> _recipes = BuildDefaults();

    /// <summary>The recipe that handles <paramref name="familyId"/>, or null when none is registered yet.</summary>
    public static IVideoRecipe? Resolve(string familyId)
    {
        foreach (IVideoRecipe recipe in _recipes)
        {
            if (recipe.Matches(familyId))
                return recipe;
        }
        return null;
    }

    /// <summary>Registers an additional recipe (last-registered wins on ties).</summary>
    public static void Register(IVideoRecipe recipe) => _recipes.Insert(0, recipe);

    /// <summary>The video families a lifted recipe currently covers.</summary>
    public static IReadOnlyCollection<string> RegisteredNames => _recipes.Select(r => r.Name).ToList();

    private static List<IVideoRecipe> BuildDefaults() => new List<IVideoRecipe>
    {
        // The Wan compat classes are shared by the plain backbone and the VACE / Animate / S2V conditioning variants;
        // WanVideoRecipe sniffs the checkpoint header and delegates, and the variants are also directly addressable.
        new Video.WanVideoRecipe("wan"),
        new Video.WanVideoRecipe(Video.WanVideoRecipe.Wan22_5BCompatClassId),
        new Video.WanVideoRecipe(Video.WanVideoRecipe.Wan21_1_3BCompatClassId),
        new Video.WanVideoRecipe(Video.WanVideoRecipe.Wan21_14BCompatClassId),
        new Video.WanVaceRecipe(),
        new Video.WanAnimateRecipe(),
        new Video.WanS2VRecipe(),
        new Video.LtxVideoRecipe(),
        new Video.LtxVideo2Recipe(),
        // Separate registration because the distilled 2.5 sampling contract is indistinguishable from dev in the
        // checkpoint — the model id is the only thing carrying that intent.
        new Video.LtxVideo2Recipe(distilled: true),
        new Video.LanceVideoRecipe(),
        new Video.Kandinsky5VideoRecipe(),
        new Video.HunyuanVideoRecipe(),
        // Registered ahead of its weights so the family resolves to a real explanation instead of "unknown
        // architecture"; Construct() throws until MiniMax publish the checkpoint (docs/Research/MINIMAX_H3.md).
        new Video.MiniMaxH3Recipe(),
    };
}
