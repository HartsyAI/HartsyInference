namespace HartsyInference.Engine.Recipes;

/// <summary>What one music family can generate, as opposed to what it recommends (<see cref="MusicDefaults"/>). A
/// recipe's defaults must satisfy these limits.</summary>
public sealed record MusicConstraints
{
    /// <summary>The longest clip the family's model can render in seconds; null when the family has no cap of its own.</summary>
    /// <remarks>Validates the recipe's own default. A caller's longer duration is still clamped by the family's loader, as
    /// before this record existed; refusing it by name is a behaviour change left to the defaults-alignment step.</remarks>
    public double? MaxDurationSeconds { get; init; }
}
