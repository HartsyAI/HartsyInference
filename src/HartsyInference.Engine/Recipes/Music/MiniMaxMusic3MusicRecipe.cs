using HartsyInference.Audio.Pipelines;
using HartsyInference.Diffusion.Pipelines;

namespace HartsyInference.Engine.Recipes.Music;

/// <summary>MiniMax Music 3: a 25 Hz frame grid caps the clip at <see cref="MiniMaxMusic3ArPipeline.MaxAudioFrames"/> frames.
/// Duration 120 s and CFG 1.5 are SwarmUI core's (<c>WorkflowGenerator.cs</c>: 120 s, <c>cfg_scale</c> 1.5 on
/// <c>MiniMaxMusic3TextEncode</c>); the docs and the pipeline's own default say CFG 1.7.</summary>
public sealed class MiniMaxMusic3MusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => "minimaxmusic3";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults
    {
        Revision = 2,
        Duration = 120,
        Steps = MiniMaxMusic3FlowPipeline.DefaultSteps,
        CfgScale = 1.5,
    };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints
    {
        MaxDurationSeconds = MiniMaxMusic3ArPipeline.MaxAudioFrames / MiniMaxMusic3ArPipeline.FrameRate,
    };
}
