using HartsyInference.Audio.Models.Music;

namespace HartsyInference.Engine.Recipes.Music;

/// <summary>YuE2: the ODE step count is the model's own; the guidance scale depends on the planning mode, so the pipeline owns it.</summary>
public sealed class Yue2MusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => "yue2";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults
    {
        Duration = 10,
        Steps = Yue2Config.V1.OdeSteps,
    };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints { MaxDurationSeconds = Yue2Protocol.MaxDurationSeconds };
}
