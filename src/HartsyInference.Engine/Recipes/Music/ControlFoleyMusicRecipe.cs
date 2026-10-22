namespace HartsyInference.Engine.Recipes.Music;

/// <summary>ControlFoley: 25 flow steps at CFG 4.5 when unset; a reference clip is read to its 30 s cap.</summary>
public sealed class ControlFoleyMusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => "controlfoley";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults
    {
        Duration = 10,
        Steps = 25,
        CfgScale = 4.5,
        ReferenceStartSeconds = 0,
        ReferenceEndSeconds = 30,
    };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints();
}
