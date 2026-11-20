using HartsyInference.Engine.Recipes;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Catalog;

/// <summary>Finds a family's prompt profile by the engine id a published artifact carries, so a conversion site can stamp it
/// into the header without constructing a pipeline.</summary>
public static class RecipePrompts
{
    /// <summary>The family-level <see cref="PromptProfile"/> of the image or video recipe that handles
    /// <paramref name="engineId"/>, or null when no recipe handles it or the recipe declares none. A family whose variants
    /// differ on a field leaves that field unset here, since a file does not say which variant it is.</summary>
    public static PromptProfile? Find(string engineId)
    {
        if (string.IsNullOrWhiteSpace(engineId))
        {
            return null;
        }
        return RecipeRegistry.Resolve(engineId)?.Prompt ?? VideoRecipeRegistry.Resolve(engineId)?.Prompt;
    }
}
