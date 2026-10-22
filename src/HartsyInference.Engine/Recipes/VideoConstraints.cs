namespace HartsyInference.Engine.Recipes;

/// <summary>What one video family can generate, as opposed to what it recommends (<see cref="VideoDefaults"/>): limits
/// planning validates a request against and a recipe's own defaults must satisfy.</summary>
public sealed record VideoConstraints
{
    /// <summary>The frame counts the family's VAE encodes.</summary>
    public required FrameGrid Frames { get; init; }
}
