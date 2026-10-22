using HartsyInference.Engine.Audio;

namespace HartsyInference.Engine.Recipes.Music;

/// <summary>YuE: a reference excerpt runs to its 30 s cap unless the caller narrows it; the sampling knobs are the pipeline's.</summary>
public sealed class YueMusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => AudioWeightsCatalog.YueId;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults
    {
        Duration = 10,
        ReferenceStartSeconds = 0,
        ReferenceEndSeconds = 30,
    };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints { MaxDurationSeconds = 300 };
}
