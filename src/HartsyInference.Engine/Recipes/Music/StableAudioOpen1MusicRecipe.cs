using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Engine.Recipes.Music;

/// <summary>Stable Audio Open 1.0: v-prediction EDM with classifier-free guidance over DPM++ 3M SDE. The defaults are the
/// model card's (100 steps, CFG 7); its trained window caps the clip at about 47.55 s.</summary>
public sealed class StableAudioOpen1MusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => "stable-audio-open-1.0";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults { Duration = 30, Steps = 100, CfgScale = 7 };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints
    {
        MaxDurationSeconds = StableAudioDitConfig.Open1_0.MaxLatentTokens
            * (double)StableAudioDitConfig.Open1_0.VaeDownsample / StableAudioDitConfig.Open1_0.SampleRate,
    };
}
