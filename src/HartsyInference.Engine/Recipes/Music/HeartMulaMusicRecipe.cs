namespace HartsyInference.Engine.Recipes.Music;

/// <summary>HeartMuLa: a 10 s clip; temperature, top-k and CFG stay with the pipeline when unset.</summary>
public sealed class HeartMulaMusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => "heartmula";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults { Duration = 10 };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints { MaxDurationSeconds = 300 };
}
