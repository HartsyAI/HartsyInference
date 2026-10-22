using HartsyInference.Audio.Pipelines;
using HartsyInference.Diffusion.Pipelines;

namespace HartsyInference.Engine.Recipes.Music;

/// <summary>MiniMax Music 3: a 25 Hz frame grid caps the clip at <see cref="MiniMaxMusic3ArPipeline.MaxAudioFrames"/> frames.</summary>
public sealed class MiniMaxMusic3MusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => "minimaxmusic3";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults
    {
        Duration = 10,
        Steps = MiniMaxMusic3FlowPipeline.DefaultSteps,
        // The pipeline's float default, written as a decimal so the declared value is exact.
        CfgScale = 1.7,
    };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints
    {
        MaxDurationSeconds = MiniMaxMusic3ArPipeline.MaxAudioFrames / MiniMaxMusic3ArPipeline.FrameRate,
    };
}
