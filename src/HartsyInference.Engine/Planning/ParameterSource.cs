namespace HartsyInference.Engine.Planning;

/// <summary>Where a resolved generation parameter's value came from.</summary>
public enum ParameterSource
{
    /// <summary>The caller left it unset and the family recipe supplied it.</summary>
    RecipeDefault,

    /// <summary>The caller left it unset and an exact-checkpoint profile supplied it in place of the recipe's.</summary>
    ProfileDefault,

    /// <summary>The caller set it and it runs as set.</summary>
    UserOverride,

    /// <summary>The caller set it and planning changed it to what the model can run; the note says from what.</summary>
    Normalized,

    /// <summary>Computed rather than chosen: a random seed, or a rate pinned by an input clip.</summary>
    Derived,
}
