namespace HartsyInference.Engine.Recipes;

/// <summary>What one music family can generate, as opposed to what it recommends (<see cref="MusicDefaults"/>). A
/// recipe's defaults must satisfy these limits.</summary>
public sealed record MusicConstraints
{
    /// <summary>The longest clip the family's model can render in seconds; null when the family has no cap of its own.</summary>
    public double? MaxDurationSeconds { get; init; }
}
