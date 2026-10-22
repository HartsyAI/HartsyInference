using HartsyInference.Audio.Models.Music;

namespace HartsyInference.Engine.Recipes.Music;

/// <summary>YuE2: the ODE step count is the model's own; the guidance scale depends on the planning mode, so the pipeline owns it.
/// The 300 s duration is SwarmUI's (<c>Audio Model Support.md</c>, YuE2; <c>WorkflowGenerator.cs</c> <c>max_duration</c>).</summary>
public sealed class Yue2MusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => "yue2";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults
    {
        Revision = 2,
        Duration = 300,
        Steps = Yue2Config.V1.OdeSteps,
    };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints { MaxDurationSeconds = Yue2Protocol.MaxDurationSeconds };
}
