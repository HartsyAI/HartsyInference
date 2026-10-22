namespace HartsyInference.Engine.Recipes.Music;

/// <summary>MusicGen and AudioGen share one loader and one window: 30 s per call, sampling knobs left to the model.</summary>
public sealed class MusicGenMusicRecipe : IMusicRecipe
{
    private readonly string _familyId;

    /// <summary>Creates the recipe for one AudioCraft family id (<c>musicgen</c> or <c>audiogen</c>).</summary>
    public MusicGenMusicRecipe(string familyId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(familyId);
        _familyId = familyId;
    }

    /// <inheritdoc/>
    public string Name => _familyId;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, _familyId, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults { Duration = 10 };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints { MaxDurationSeconds = 30 };
}
