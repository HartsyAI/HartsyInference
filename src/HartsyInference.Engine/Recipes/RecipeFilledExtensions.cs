using System.Runtime.CompilerServices;

namespace HartsyInference.Engine.Recipes;

/// <summary>Reads a request field that a recipe fills, failing by name when the recipe was never applied.</summary>
internal static class RecipeFilledExtensions
{
    /// <summary>The value the recipe filled into <paramref name="value"/>; throws when it is still unset.</summary>
    internal static T Filled<T>(this T? value, [CallerArgumentExpression(nameof(value))] string? field = null)
        where T : struct =>
        value ?? throw new InvalidOperationException($"{field} was not filled from the music recipe; apply MusicDefaults first.");
}
